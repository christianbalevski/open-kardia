"""kardia6l desktop GUI — a Tkinter front-end for the KardiaMobile 6L streaming client.

Research tool. NOT a medical device, NOT for diagnosis.

Everything runs through :class:`kardia6l.appcore.AppController`; this module never touches
asyncio or bleak directly. The GUI drains controller events and reads waveform/stat
snapshots on Tk ``after`` timers. Slow controller calls (analysis, report export) run on a
worker thread and marshal their result back through a queue drained on the GUI thread.

Run:
    python -m kardia6l.app            # real device
    python -m kardia6l.app --connect  # real device, auto-connect on launch
    python -m kardia6l.app --sim      # simulated 6L, no hardware
    python -m kardia6l.app --sim --seconds 4   # auto-close after 4 s (smoke test)
    python -m kardia6l.app --sim --selftest    # sim: auto gated-record, print result, close
"""
from __future__ import annotations

import math
import queue
import subprocess
import sys
import threading
import time

import matplotlib

matplotlib.use("TkAgg")  # must precede pyplot / backend import

import numpy as np  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from kardia6l import print_disclaimer  # noqa: E402
from kardia6l import dsp  # noqa: E402
from kardia6l.appcore import AppController  # noqa: E402
from kardia6l.leads import LEAD_NAMES_6  # noqa: E402

PLOT_SECONDS = 6.0
PLOT_FILTER_PAD = 3.0   # fetch this much extra history so display_filter edges are clean
REDRAW_MS = 60          # ~16 Hz
EVENT_POLL_MS = 120
STAT_MS = 400
WORKER_MS = 120

VIEW_1LEAD = "Lead I"
VIEW_2LEAD = "Lead I + II"
VIEW_6LEAD = "All 6 leads"
VIEWS = {VIEW_1LEAD: ["I"], VIEW_2LEAD: ["I", "II"], VIEW_6LEAD: list(LEAD_NAMES_6)}

BG = "#101418"
PANEL = "#151b21"
FG = "#e8edf2"
ACCENT = "#39d0d8"
GRID = "#243038"
GREEN = "#33d17a"
ORANGE = "#f6a609"
RED = "#f04747"
GREY = "#5a6673"


def _isnum(v) -> bool:
    try:
        return v is not None and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _fmt(v, unit: str = "", digits: int = 0) -> str:
    """Format a possibly-NaN number; em-dash when missing."""
    if not _isnum(v):
        return "—"
    v = float(v)
    s = f"{v:.{digits}f}" if digits else f"{v:.0f}"
    return f"{s}{(' ' + unit) if unit else ''}"


class App:
    def __init__(self, root: tk.Tk, sim: bool = False, mains: float = 60.0, loop=None):
        self.root = root
        self.mains = mains
        self.sim_default = sim
        self._loop = loop               # shared main-thread asyncio loop (macOS-safe BLE)
        self._closing = False
        self.ctrl: AppController | None = None
        self._lines: list = []
        self._axes: list = []
        self._leads: list = []
        self._recording = False        # manual record-to-file
        self._gating = False           # quality-gated timed recording active
        self._worker_q: queue.Queue = queue.Queue()
        self._selftest = False
        self._selftest_done = False

        root.title("kardia6l — 6-lead ECG (research, NOT a medical device)")
        root.geometry("1220x860")
        root.minsize(920, 640)
        root.configure(bg=BG)

        self._init_style()
        self._build_toolbar()
        self._build_readouts()
        self._build_view_selector()
        self._build_content()      # plot (left) + side panel (right)
        self._build_bottom()

        self._rebuild_axes(VIEW_2LEAD)
        self._refresh_library()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(EVENT_POLL_MS, self._pump_events)
        root.after(STAT_MS, self._pump_stats)
        root.after(REDRAW_MS, self._pump_plot)
        root.after(WORKER_MS, self._pump_worker)

    # ------------------------------------------------------------------ style
    def _init_style(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground="#1a2128")
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Muted.TLabel", background=BG, foreground=GREY)
        st.configure("PanelMuted.TLabel", background=PANEL, foreground=GREY)
        st.configure("Panel.TLabel", background=PANEL, foreground=FG)
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.map("TCheckbutton", background=[("active", BG)])
        st.configure("TButton", background="#1f2933", foreground=FG, padding=6, borderwidth=0)
        st.map("TButton", background=[("active", "#2a3846"), ("disabled", "#161c22")],
               foreground=[("disabled", GREY)])
        st.configure("Accent.TButton", background=ACCENT, foreground="#04242a")
        st.map("Accent.TButton", background=[("active", "#5be0e8"), ("disabled", "#1f2933")])
        st.configure("Rec.TButton", background=RED, foreground="#fff")
        st.map("Rec.TButton", background=[("active", "#ff6b6b")])
        st.configure("TRadiobutton", background=BG, foreground=FG)
        st.map("TRadiobutton", background=[("active", BG)])
        st.configure("TLabelframe", background=PANEL, foreground=FG, borderwidth=1)
        st.configure("TLabelframe.Label", background=PANEL, foreground=ACCENT)
        st.configure("TSpinbox", fieldbackground="#1a2128", foreground=FG, arrowcolor=FG)
        st.configure("TCombobox", fieldbackground="#1a2128", foreground=FG, arrowcolor=FG)
        st.configure("Gate.Horizontal.TProgressbar", background=ACCENT, troughcolor="#0b0f13")
        st.configure("Treeview", background="#0f151b", fieldbackground="#0f151b", foreground=FG,
                     rowheight=22, borderwidth=0)
        st.map("Treeview", background=[("selected", "#22333f")], foreground=[("selected", FG)])
        st.configure("Treeview.Heading", background="#1f2933", foreground=GREY, borderwidth=0)
        st.map("Treeview.Heading", background=[("active", "#2a3846")])

    # ------------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(12, 10))
        bar.pack(side=tk.TOP, fill=tk.X)

        self.sim_var = tk.BooleanVar(value=self.sim_default)
        self.sim_chk = ttk.Checkbutton(bar, text="Simulated device", variable=self.sim_var)
        self.sim_chk.pack(side=tk.LEFT, padx=(0, 12))

        self.single_var = tk.BooleanVar(value=False)
        self.single_chk = ttk.Checkbutton(bar, text="Single lead (hands only)",
                                          variable=self.single_var, command=self._on_single_toggle)
        self.single_chk.pack(side=tk.LEFT, padx=(0, 12))

        self.connect_btn = ttk.Button(bar, text="Connect", style="Accent.TButton",
                                      command=self._on_connect_toggle)
        self.connect_btn.pack(side=tk.LEFT)

        self.dot = tk.Canvas(bar, width=14, height=14, bg=BG, highlightthickness=0)
        self._dot_id = self.dot.create_oval(2, 2, 12, 12, fill=GREY, outline="")
        self.dot.pack(side=tk.LEFT, padx=(14, 6))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(bar, textvariable=self.status_var).pack(side=tk.LEFT)

    # ------------------------------------------------------------------ readouts
    def _build_readouts(self):
        row = ttk.Frame(self.root, padding=(12, 4))
        row.pack(side=tk.TOP, fill=tk.X)

        hr_box = ttk.Frame(row)
        hr_box.pack(side=tk.LEFT, padx=(0, 26))
        self.hr_var = tk.StringVar(value="--")
        tk.Label(hr_box, textvariable=self.hr_var, bg=BG, fg=ACCENT,
                 font=("Helvetica", 34, "bold")).pack(side=tk.LEFT)
        tk.Label(hr_box, text="bpm", bg=BG, fg=GREY, font=("Helvetica", 13)).pack(
            side=tk.LEFT, padx=(6, 0), pady=(14, 0))

        q_box = ttk.Frame(row)
        q_box.pack(side=tk.LEFT, padx=(0, 26))
        ttk.Label(q_box, text="Signal quality", style="Muted.TLabel").pack(anchor="w")
        self.sqi_var = tk.StringVar(value="--")
        self.sqi_lbl = tk.Label(q_box, textvariable=self.sqi_var, bg=BG, fg=GREY,
                                font=("Helvetica", 15, "bold"))
        self.sqi_lbl.pack(anchor="w")

        t_box = ttk.Frame(row)
        t_box.pack(side=tk.LEFT, padx=(0, 26))
        ttk.Label(t_box, text="Elapsed", style="Muted.TLabel").pack(anchor="w")
        self.elapsed_var = tk.StringVar(value="00:00")
        tk.Label(t_box, textvariable=self.elapsed_var, bg=BG, fg=FG,
                 font=("Helvetica", 15, "bold")).pack(anchor="w")

        self.meta_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.meta_var, style="Muted.TLabel").pack(
            side=tk.RIGHT, anchor="e")

    # ------------------------------------------------------------------ view selector
    def _build_view_selector(self):
        row = ttk.Frame(self.root, padding=(12, 2))
        row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row, text="View:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        self.view_var = tk.StringVar(value=VIEW_2LEAD)
        self._view_radios = {}
        for name in (VIEW_1LEAD, VIEW_2LEAD, VIEW_6LEAD):
            rb = ttk.Radiobutton(row, text=name, value=name, variable=self.view_var,
                                 command=self._on_view_change)
            rb.pack(side=tk.LEFT, padx=(0, 10))
            self._view_radios[name] = rb

    # ------------------------------------------------------------------ content: plot + side panel
    def _build_content(self):
        content = ttk.Frame(self.root)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # plot on the left
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG)
        self.fig.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.07, hspace=0.35)
        self.canvas = FigureCanvasTkAgg(self.fig, master=content)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                         padx=(10, 6), pady=(4, 4))

        # side panel on the right
        side = ttk.Frame(content, style="Panel.TFrame", padding=10, width=340)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(4, 4))
        side.pack_propagate(False)
        self._build_timed_recording(side)
        self._build_library(side)

    def _build_timed_recording(self, parent):
        lf = ttk.Labelframe(parent, text="Timed recording", padding=10)
        lf.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(lf, style="Panel.TFrame")
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Duration", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.dur_var = tk.StringVar(value="30")
        ttk.Combobox(row1, textvariable=self.dur_var, values=["15", "30", "60", "120", "180", "240", "300"],
                     width=5, state="readonly").grid(row=0, column=1, padx=(6, 16))
        ttk.Label(row1, text="Quality ≥", style="Panel.TLabel").grid(row=0, column=2, sticky="w")
        self.thr_var = tk.IntVar(value=90)
        self.thr_spin = ttk.Spinbox(row1, from_=50, to=99, increment=5, width=5,
                                    textvariable=self.thr_var)
        self.thr_spin.grid(row=0, column=3, padx=(6, 2))
        ttk.Label(row1, text="%", style="PanelMuted.TLabel").grid(row=0, column=4, sticky="w")

        self.gate_btn = ttk.Button(lf, text="● Record (quality-gated)", style="Accent.TButton",
                                   command=self._on_gate_toggle, state=tk.DISABLED)
        self.gate_btn.pack(fill=tk.X, pady=(10, 6))

        self.gate_pb = ttk.Progressbar(lf, style="Gate.Horizontal.TProgressbar",
                                       mode="determinate", maximum=100)
        self.gate_pb.pack(fill=tk.X)
        self.gate_status = tk.Label(lf, text="Connect and hold the electrodes to record.",
                                    bg=PANEL, fg=GREY, font=("Helvetica", 9), wraplength=300,
                                    justify="left", anchor="w")
        self.gate_status.pack(fill=tk.X, pady=(6, 0))

    def _build_library(self, parent):
        lf = ttk.Labelframe(parent, text="Saved readings", padding=8)
        lf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(12, 0))

        cols = ("time", "dur", "hr", "rhythm")
        self.tree = ttk.Treeview(lf, columns=cols, show="headings", height=10, selectmode="browse")
        for c, txt, w, anchor in (("time", "Time", 118, "w"), ("dur", "Dur", 46, "e"),
                                  ("hr", "HR", 46, "e"), ("rhythm", "Rhythm", 110, "w")):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor=anchor, stretch=(c == "rhythm"))
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<Double-1>", lambda _e: self._on_view_analysis())

        actions = ttk.Frame(parent, style="Panel.TFrame")
        actions.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        specs = [("View analysis", self._on_view_analysis), ("Export PDF…", lambda: self._on_export("pdf")),
                 ("Export PNG…", lambda: self._on_export("png")), ("Open folder", self._on_open_folder),
                 ("Delete", self._on_delete), ("Refresh", self._refresh_library)]
        for i, (txt, cmd) in enumerate(specs):
            ttk.Button(actions, text=txt, command=cmd).grid(
                row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------ plot axes
    def _rebuild_axes(self, view: str):
        leads = VIEWS[view]
        self.fig.clf()
        self._axes = []
        self._lines = []
        n = len(leads)
        t = np.linspace(-PLOT_SECONDS, 0, 2)
        for i, lead in enumerate(leads):
            ax = self.fig.add_subplot(n, 1, i + 1, facecolor="#0b0f13")
            ax.set_xlim(-PLOT_SECONDS, 0)
            ax.set_ylim(-1.0, 1.0)
            (ln,) = ax.plot(t, [0, 0], color=ACCENT, lw=0.9)
            ax.set_ylabel(f"{lead}\n(mV)", color=FG, fontsize=9, rotation=0, labelpad=18,
                          va="center")
            ax.tick_params(colors=GREY, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("seconds", color=GREY, fontsize=8)
            self._axes.append(ax)
            self._lines.append(ln)
        self._leads = leads
        self.canvas.draw_idle()

    def _on_view_change(self):
        self._rebuild_axes(self.view_var.get())

    def _on_single_toggle(self):
        """Single-lead (hands-only) mode: only Lead I is valid, so restrict the view to it
        and disable the multi-lead views. Set before connecting."""
        single = self.single_var.get()
        if single:
            self.view_var.set(VIEW_1LEAD)
            self._rebuild_axes(VIEW_1LEAD)
        for name, rb in self._view_radios.items():
            if name in (VIEW_2LEAD, VIEW_6LEAD):
                rb.configure(state=(tk.DISABLED if single else tk.NORMAL))

    # ------------------------------------------------------------------ bottom bar
    def _build_bottom(self):
        bar = ttk.Frame(self.root, padding=(12, 8))
        bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.rec_btn = ttk.Button(bar, text="● Record to file…", command=self._on_record_toggle,
                                  state=tk.DISABLED)
        self.rec_btn.pack(side=tk.LEFT)

        self.snap_btn = ttk.Button(bar, text="Save snapshot…", command=self._on_save_snapshot,
                                   state=tk.DISABLED)
        self.snap_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.rec_status = tk.Label(bar, text="", bg=BG, fg=RED, font=("Helvetica", 10, "bold"))
        self.rec_status.pack(side=tk.LEFT, padx=(12, 0))

        tk.Label(bar, text="Research signal — not a medical device, not for diagnosis.",
                 bg=BG, fg=GREY, font=("Helvetica", 9)).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------ connect / disconnect
    def _on_connect_toggle(self):
        state = self.ctrl.state if self.ctrl else "idle"
        if state in ("connecting", "streaming"):
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        self.ctrl = AppController(sim=self.sim_var.get(), mains=self.mains, derive=True,
                                  single_lead=self.single_var.get())
        if self._loop is not None:
            # macOS: run BLE on the shared main-thread loop, not a worker thread.
            self.ctrl.attach_loop(self._loop)
        self.sim_chk.configure(state=tk.DISABLED)
        self.single_chk.configure(state=tk.DISABLED)
        self.connect_btn.configure(text="Disconnect", style="TButton")
        self.status_var.set("Connecting…")
        self.hr_var.set("--")
        self.sqi_var.set("--")
        try:
            self.ctrl.start()
        except Exception as e:  # noqa: BLE001
            self._show_error(str(e))

    def _disconnect(self):
        if self._gating and self.ctrl:
            try:
                self.ctrl.cancel_gated_recording()
            except Exception:  # noqa: BLE001
                pass
        if self._recording:
            self._stop_recording()
        if self.ctrl:
            try:
                self.ctrl.stop()
            except Exception:  # noqa: BLE001
                pass
        self._reset_idle("Stopped")

    def _reset_idle(self, status: str = "Idle"):
        self.connect_btn.configure(text="Connect", style="Accent.TButton")
        self.sim_chk.configure(state=tk.NORMAL)
        self.single_chk.configure(state=tk.NORMAL)
        self.rec_btn.configure(state=tk.DISABLED)
        self.snap_btn.configure(state=tk.DISABLED)
        self.gate_btn.configure(state=tk.DISABLED)
        self._reset_gate_ui()
        self.status_var.set(status)
        self.dot.itemconfigure(self._dot_id, fill=GREY)

    # ------------------------------------------------------------------ quality-gated recording
    def _on_gate_toggle(self):
        if not self.ctrl:
            return
        if self._gating:
            try:
                self.ctrl.cancel_gated_recording()
            except Exception:  # noqa: BLE001
                pass
            self.gate_status.configure(text="Cancelling…")
            return
        try:
            secs = float(self.dur_var.get())
        except ValueError:
            secs = 30.0
        thr = max(50, min(99, int(self.thr_var.get()))) / 100.0
        self._gating = True
        self.gate_btn.configure(text="Cancel", style="Rec.TButton")
        self.thr_spin.configure(state=tk.DISABLED)
        self.gate_pb.configure(value=0)
        self.gate_status.configure(fg=GREY, text="Waiting for signal quality…")
        try:
            self.ctrl.start_gated_recording(seconds=secs, sqi_threshold=thr, hold_s=4.0)
        except Exception as e:  # noqa: BLE001
            self._reset_gate_ui()
            self._show_error(f"Could not start timed recording:\n{e}")

    def _reset_gate_ui(self):
        self._gating = False
        self.gate_btn.configure(text="● Record (quality-gated)", style="Accent.TButton")
        try:
            self.thr_spin.configure(state=tk.NORMAL)
        except tk.TclError:
            pass
        self.gate_pb.configure(value=0)

    def _on_gate_event(self, kind, payload):
        if kind == "gate":
            state = (payload or {}).get("state", "waiting")
            if state == "analyzing":
                self.gate_pb.configure(value=100)
                self.gate_status.configure(fg=ACCENT, text="Analyzing recording…")
            else:
                sqi = payload.get("sqi", float("nan"))
                thr = payload.get("threshold", 0.9)
                held = payload.get("held_s", 0.0)
                hold = payload.get("hold_s", 1.5) or 1.5
                pct = int(round(sqi * 100)) if _isnum(sqi) else 0
                need = int(round(thr * 100))
                self.gate_pb.configure(value=max(0, min(100, held / hold * 100)))
                if held > 0:
                    self.gate_status.configure(
                        fg=GREEN, text=f"Signal quality {pct}% — hold still, recording starts in {max(0.0, hold - held):.0f} s")
                else:
                    self.gate_status.configure(
                        fg=GREY, text=f"Waiting for signal quality: {pct}%  (need ≥ {need}%)")
        elif kind == "rec_start":
            secs = (payload or {}).get("seconds", 0)
            self.gate_pb.configure(value=0)
            self.gate_status.configure(fg=GREEN, text=f"Recording {secs:.0f} s… 0%")
            self._rec_secs = secs
        elif kind == "rec_progress":
            frac = float(payload or 0.0)
            self.gate_pb.configure(value=max(0, min(100, frac * 100)))
            secs = getattr(self, "_rec_secs", 0)
            self.gate_status.configure(fg=GREEN, text=f"Recording {secs:.0f} s… {int(frac*100)}%")
        elif kind == "reading_complete":
            self._reset_gate_ui()
            self.gate_status.configure(fg=GREEN, text="Reading saved.")
            self._refresh_library()
            rec = (payload or {}).get("record")
            analysis = (payload or {}).get("analysis")
            reports = (payload or {}).get("reports") or {}
            rid = getattr(rec, "id", "reading")
            self._show_analysis(f"Reading {rid}", analysis, reports=reports, record=rec)
            if self._selftest:
                print(f"SELFTEST: reading_complete {rid}", flush=True)
                self._selftest_done = True
                self.root.after(600, self._on_close)
        elif kind == "gate_cancelled":
            self._reset_gate_ui()
            self.gate_status.configure(fg=GREY, text="Cancelled.")
        elif kind == "gate_error":
            self._reset_gate_ui()
            self.gate_status.configure(fg=ORANGE, text=str(payload))

    # ------------------------------------------------------------------ manual recording
    def _on_record_toggle(self):
        if not self.ctrl:
            return
        if self._recording:
            self._stop_recording()
        else:
            path = filedialog.asksaveasfilename(
                title="Record ECG to file", defaultextension="",
                initialfile=time.strftime("kardia6l-%Y%m%d-%H%M%S"),
                filetypes=[("CSV + WFDB", "*")])
            if not path:
                return
            try:
                written = self.ctrl.start_saving(path, ["csv", "wfdb"])
            except Exception as e:  # noqa: BLE001
                self._show_error(f"Could not start recording:\n{e}")
                return
            self._recording = True
            self.rec_btn.configure(text="■ Stop recording", style="Rec.TButton")
            shown = written[0] if written else path
            self.rec_status.configure(text=f"● REC  {shown}")

    def _stop_recording(self):
        try:
            if self.ctrl:
                self.ctrl.stop_saving()
        except Exception:  # noqa: BLE001
            pass
        self._recording = False
        self.rec_btn.configure(text="● Record to file…", style="TButton")
        self.rec_status.configure(text="")

    def _on_save_snapshot(self):
        if not self.ctrl:
            return
        path = filedialog.asksaveasfilename(
            title="Save last 30 s snapshot", defaultextension="",
            initialfile=time.strftime("kardia6l-snap-%Y%m%d-%H%M%S"),
            filetypes=[("CSV + WFDB", "*")])
        if not path:
            return
        try:
            written = self.ctrl.save_buffer(path, 30, ["csv", "wfdb"])
        except Exception as e:  # noqa: BLE001
            self._show_error(f"Could not save snapshot:\n{e}")
            return
        if written:
            messagebox.showinfo("Saved", f"Snapshot written:\n{written[0]}")
        else:
            messagebox.showwarning("Nothing to save", "No samples buffered yet.")

    # ------------------------------------------------------------------ library panel
    def _lib_ctrl(self):
        """A controller for library-only operations (no event loop needed). Uses the live
        controller when connected, else a cached throwaway."""
        if self.ctrl:
            return self.ctrl
        if getattr(self, "_lib_ctrl_cached", None) is None:
            self._lib_ctrl_cached = AppController(sim=self.sim_default)
        return self._lib_ctrl_cached

    def _refresh_library(self):
        if not hasattr(self, "tree"):
            return
        try:
            records = self._lib_ctrl().list_readings()
        except Exception:  # noqa: BLE001
            records = []
        self.tree.delete(*self.tree.get_children())
        self._records = {}
        for rec in records:
            rid = getattr(rec, "id", "?")
            self._records[rid] = rec
            when = str(getattr(rec, "created_at", rid))[:19].replace("T", " ")
            dur = getattr(rec, "duration_s", float("nan"))
            hr = getattr(rec, "hr_bpm", float("nan"))
            rhythm = getattr(rec, "rhythm", None) or "—"
            self.tree.insert("", "end", iid=rid,
                             values=(when, _fmt(dur, "s"), _fmt(hr), rhythm))

    def _selected_id(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _on_view_analysis(self):
        rid = self._selected_id()
        if not rid:
            return
        ctrl = self._lib_ctrl()
        rec = self._records.get(rid)
        self._toast(f"Analyzing {rid}…")
        self._run_bg(lambda: ctrl.analyze_reading(rid),
                     lambda ok, res: self._show_analysis(f"Reading {rid}", res, record=rec)
                     if ok else self._show_error(f"Analysis failed:\n{res}"))

    def _on_export(self, fmt: str):
        rid = self._selected_id()
        if not rid:
            return
        path = filedialog.asksaveasfilename(
            title=f"Export report ({fmt.upper()})", defaultextension=f".{fmt}",
            initialfile=f"{rid}-report.{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path:
            return
        ctrl = self._lib_ctrl()
        self._toast(f"Rendering {fmt.upper()}…")
        self._run_bg(lambda: ctrl.export_report(rid, path, fmt),
                     lambda ok, res: (self._toast("Report exported."), self._open(res))
                     if ok and res else self._show_error(f"Export failed:\n{res}"))

    def _on_open_folder(self):
        rid = self._selected_id()
        rec = getattr(self, "_records", {}).get(rid) if rid else None
        if rec is not None and getattr(rec, "dir", None):
            self._open(rec.dir)

    def _on_delete(self):
        rid = self._selected_id()
        if not rid:
            return
        if not messagebox.askyesno("Delete reading", f"Delete reading {rid}?\nThis cannot be undone."):
            return
        try:
            self._lib_ctrl().delete_reading(rid)
        except Exception as e:  # noqa: BLE001
            self._show_error(f"Delete failed:\n{e}")
        self._refresh_library()

    # ------------------------------------------------------------------ analysis view
    def _show_analysis(self, title, analysis, reports=None, record=None):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        win.geometry("560x680")

        head = tk.Frame(win, bg=BG)
        head.pack(fill=tk.X, padx=16, pady=(14, 6))
        rhythm = getattr(analysis, "rhythm", None)
        label = getattr(rhythm, "label", "Undetermined") if rhythm else "No analysis"
        conf = getattr(rhythm, "confidence", 0.0) if rhythm else 0.0
        rationale = getattr(rhythm, "rationale", "") if rhythm else ""
        col = GREEN if "normal" in str(label).lower() else (ORANGE if rhythm else GREY)
        tk.Label(head, text=label, bg=BG, fg=col, font=("Helvetica", 18, "bold"),
                 wraplength=420, justify="left").pack(anchor="w")
        if rhythm:
            tk.Label(head, text=f"confidence {int(conf*100)}%", bg=BG, fg=GREY,
                     font=("Helvetica", 10)).pack(anchor="w")
        if rationale:
            tk.Label(head, text=rationale, bg=BG, fg=GREY, font=("Helvetica", 10),
                     wraplength=420, justify="left").pack(anchor="w", pady=(2, 0))

        iv = getattr(analysis, "intervals", None)
        hrv = getattr(analysis, "hrv", {}) or {}
        axis_label = getattr(analysis, "axis_label", "") or ""
        # rows: (label, value_text, normal_range_text, status)  status in normal/low/high/''
        rows: list[tuple] = []
        if hasattr(analysis, "rows"):
            for r in analysis.rows():
                val = _fmt(r["value"], r["unit"], r.get("digits", 0))
                if r["key"] == "qrs_axis_deg" and axis_label and _isnum(r["value"]):
                    val = f"{val}  ({axis_label})"
                rows.append((r["label"], val, r.get("range", ""), r.get("status", "")))
        else:  # duck-typed stub / no rows()
            from kardia6l.analysis import classify, range_text
            spec = [("Heart rate", "hr_mean", "bpm", 0), ("PR interval", "pr_ms", "ms", 0),
                    ("QRS duration", "qrs_ms", "ms", 0), ("QT interval", "qt_ms", "ms", 0),
                    ("QTc (Bazett)", "qtc_bazett_ms", "ms", 0),
                    ("QTc (Fridericia)", "qtc_fridericia_ms", "ms", 0),
                    ("P duration", "p_dur_ms", "ms", 0), ("T duration", "t_dur_ms", "ms", 0),
                    ("ST level", "st_level_mv", "mV", 2)]
            for label, key, unit, d in spec:
                v = getattr(iv, key, None)
                rows.append((label, _fmt(v, unit, d), range_text(key), classify(key, v)))
            av = getattr(analysis, "qrs_axis_deg", None)
            rows.append(("QRS axis", f"{_fmt(av, '°')}  ({axis_label})" if axis_label else _fmt(av, "°"),
                         range_text("qrs_axis_deg"), classify("qrs_axis_deg", av)))
        # context rows without a clinical range
        rows += [
            ("RR mean ± SD", f"{_fmt(getattr(iv,'rr_mean_ms',None))} ± "
                             f"{_fmt(getattr(iv,'rr_sd_ms',None))} ms" if iv else "—", "", ""),
            ("HR min–max", f"{_fmt(getattr(iv,'hr_min',None))}–{_fmt(getattr(iv,'hr_max',None))} bpm"
                           if iv else "—", "", ""),
            ("SDNN", _fmt(hrv.get("SDNN"), "ms"), "", ""),
            ("RMSSD", _fmt(hrv.get("RMSSD"), "ms"), "", ""),
            ("pNN50", _fmt(hrv.get("pNN50"), "%", 1), "", ""),
            ("Beats analysed", _fmt(getattr(analysis, "n_beats", None)), "", ""),
            ("Analysis lead", getattr(analysis, "lead_used", "—") or "—", "", ""),
        ]
        # header row + grid
        tk.Label(win, text="Measurements (normal ranges are adult resting references, "
                           "not diagnostic thresholds)", bg=BG, fg=GREY,
                 font=("Helvetica", 8), wraplength=480, justify="left").pack(anchor="w", padx=16)
        grid = tk.Frame(win, bg=BG)
        grid.pack(fill=tk.X, padx=16, pady=(4, 6))
        for i, (k, v, rng, status) in enumerate(rows):
            vcol = GREEN if status == "normal" else (ORANGE if status in ("low", "high") else FG)
            tk.Label(grid, text=k, bg=BG, fg=GREY, font=("Helvetica", 10), anchor="w").grid(
                row=i, column=0, sticky="w", pady=1)
            vtxt = v + (f"  {status.upper()}" if status in ("low", "high") else "")
            tk.Label(grid, text=vtxt, bg=BG, fg=vcol, font=("Helvetica", 11, "bold"),
                     anchor="e").grid(row=i, column=1, sticky="e", padx=(16, 8), pady=1)
            tk.Label(grid, text=(f"normal {rng}" if rng else ""), bg=BG, fg=GREY,
                     font=("Helvetica", 8), anchor="e").grid(row=i, column=2, sticky="e", pady=1)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=0)
        grid.columnconfigure(2, weight=0)

        # rhythm findings (arrhythmia screen: ectopy, pauses, conduction, AF)
        arr = getattr(analysis, "arrhythmia", None)
        findings = list(getattr(arr, "findings", []) or []) if arr is not None else []
        if findings:
            tk.Label(win, text="Rhythm findings (automated screen — not a diagnosis)", bg=BG, fg=GREY,
                     font=("Helvetica", 8)).pack(anchor="w", padx=16, pady=(6, 0))
            fgrid = tk.Frame(win, bg=BG)
            fgrid.pack(fill=tk.X, padx=16, pady=(2, 4))
            sev_col = {"urgent": RED, "notable": ORANGE, "info": GREY}
            for i, f in enumerate(findings[:8]):
                sev = str(getattr(f, "severity", "info"))
                tk.Label(fgrid, text=sev.upper(), bg=BG, fg=sev_col.get(sev, GREY),
                         font=("Helvetica", 8, "bold"), anchor="w", width=8).grid(row=2 * i, column=0, sticky="nw")
                conf = getattr(f, "confidence", None)
                lbl = str(getattr(f, "label", "")) + (f"  ({int(conf * 100)}%)" if _isnum(conf) else "")
                tk.Label(fgrid, text=lbl, bg=BG, fg=FG, font=("Helvetica", 10, "bold"), anchor="w",
                         wraplength=430, justify="left").grid(row=2 * i, column=1, sticky="w")
                ev = str(getattr(f, "evidence", "") or "")
                when = f.when() if hasattr(f, "when") else ""
                if when:
                    ev = (ev + "  —  " if ev else "") + when
                if ev:
                    tk.Label(fgrid, text=ev, bg=BG, fg=GREY, font=("Helvetica", 8), anchor="w",
                             wraplength=430, justify="left").grid(row=2 * i + 1, column=1, sticky="w", pady=(0, 3))
            fgrid.columnconfigure(1, weight=1)

        warnings = list(getattr(analysis, "warnings", []) or [])
        flags = list(getattr(rhythm, "flags", []) or []) if rhythm else []
        notes = warnings + flags
        if notes:
            tk.Label(win, text="⚠ " + "; ".join(str(n) for n in notes), bg=BG, fg=ORANGE,
                     font=("Helvetica", 9), wraplength=420, justify="left").pack(
                anchor="w", padx=16, pady=(4, 0))

        btns = tk.Frame(win, bg=BG)
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=12)
        pdf = (reports or {}).get("pdf")
        if pdf:
            ttk.Button(btns, text="Open report (PDF)", style="Accent.TButton",
                       command=lambda p=pdf: self._open(p)).pack(side=tk.LEFT)
            ttk.Button(btns, text="Reveal in Finder",
                       command=lambda p=pdf: self._open(p, reveal=True)).pack(side=tk.LEFT, padx=(8, 0))
        elif record is not None:
            rid = getattr(record, "id", None)
            ttk.Button(btns, text="Export PDF…", style="Accent.TButton",
                       command=lambda: (win.destroy(), self.tree.selection_set(rid),
                                        self._on_export("pdf"))).pack(side=tk.LEFT)
            if getattr(record, "dir", None):
                ttk.Button(btns, text="Open folder",
                           command=lambda d=record.dir: self._open(d)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)
        tk.Label(win, text="Research signal — not a medical device.", bg=BG, fg=GREY,
                 font=("Helvetica", 8)).pack(side=tk.BOTTOM, pady=(0, 4))

    # ------------------------------------------------------------------ worker thread
    def _run_bg(self, work, on_done):
        """Run `work()` on a thread; queue `on_done(ok, result)` for the GUI thread."""
        def runner():
            try:
                res = work()
                self._worker_q.put(lambda: on_done(True, res))
            except Exception as e:  # noqa: BLE001
                self._worker_q.put(lambda: on_done(False, e))
        threading.Thread(target=runner, daemon=True).start()

    def _pump_worker(self):
        try:
            while True:
                try:
                    cb = self._worker_q.get_nowait()
                except queue.Empty:
                    break
                try:
                    cb()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self.root.after(WORKER_MS, self._pump_worker)

    # ------------------------------------------------------------------ event / stat / plot pumps
    def _pump_events(self):
        try:
            if self.ctrl:
                for kind, payload in self.ctrl.poll_events():
                    self._handle_event(kind, payload)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.root.after(EVENT_POLL_MS, self._pump_events)

    def _handle_event(self, kind, payload):
        if kind in ("gate", "rec_start", "rec_progress", "reading_complete",
                    "gate_cancelled", "gate_error"):
            self._on_gate_event(kind, payload)
            return
        if kind == "status":
            label = {"connecting": "Connecting…", "streaming": "Streaming",
                     "stopped": "Stopped", "error": "Error", "idle": "Idle"}.get(payload, payload)
            self.status_var.set(label)
        elif kind == "connected":
            name = payload[0] if payload else "device"
            self.status_var.set(f"Streaming — {name}")
            self.rec_btn.configure(state=tk.NORMAL)
            self.snap_btn.configure(state=tk.NORMAL)
            self.gate_btn.configure(state=tk.NORMAL)
            self.gate_status.configure(fg=GREY, text="Ready. Hold the electrodes and press record.")
            if self._selftest and not self._selftest_done:
                self.root.after(1200, self._selftest_kick)
        elif kind == "error":
            self._show_error(str(payload))
        elif kind == "stopped":
            self._reset_idle("Stopped")
        elif kind == "stall":
            try:
                self.rec_status.configure(text=f"⚠ signal dropout ({float(payload):.0f} ms)")
            except Exception:  # noqa: BLE001
                pass
        elif kind == "saving":
            self._refresh_library()
        elif kind == "saved":
            self._refresh_library()

    def _selftest_kick(self):
        try:
            self.dur_var.set("6")
            self.ctrl.start_gated_recording(seconds=6, sqi_threshold=0.6, hold_s=0.6)
            self._gating = True
            self.gate_btn.configure(text="Cancel", style="Rec.TButton")
        except Exception as e:  # noqa: BLE001
            print("SELFTEST: error", e, flush=True)
            self.root.after(500, self._on_close)

    def _pump_stats(self):
        try:
            if self.ctrl:
                s = self.ctrl.stats()
                if s.hr_bpm == s.hr_bpm and 20 < s.hr_bpm < 300:  # not NaN, plausible
                    self.hr_var.set(f"{s.hr_bpm:.0f}")
                else:
                    self.hr_var.set("--")
                if s.sqi == s.sqi:  # not NaN
                    pct = int(round(max(0.0, min(1.0, s.sqi)) * 100))
                    label = "Readable" if s.readable else "Noisy / Unreadable"
                    self.sqi_var.set(f"{pct}%  {label}")
                    self.sqi_lbl.configure(fg=GREEN if s.readable else ORANGE)
                else:
                    self.sqi_var.set("--")
                    self.sqi_lbl.configure(fg=GREY)
                self.elapsed_var.set(time.strftime("%M:%S", time.gmtime(max(0, s.elapsed_s))))
                self.meta_var.set(
                    f"{s.samples:,} samples · {s.rate_hz:.0f} Hz · gaps {s.seq_gaps} · drop {s.dropped}")
                self.dot.itemconfigure(self._dot_id, fill=GREEN if s.connected else GREY)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.root.after(STAT_MS, self._pump_stats)

    def _pump_plot(self):
        try:
            if self.ctrl and self._leads:
                # Fetch a longer buffer than shown so the display filter's edge transients
                # fall OUTSIDE the visible window (fixes the "shaky" window ends).
                t, x, names = self.ctrl.latest(PLOT_SECONDS + PLOT_FILTER_PAD)
                if len(x) > 4 and x.shape[1] >= 1:
                    fs = self.ctrl.fs
                    try:
                        xf = dsp.display_filter(x, fs, self.mains, window_s=PLOT_SECONDS)
                    except Exception:  # noqa: BLE001 - fall back to plain windowed clean
                        n = min(len(x), int(PLOT_SECONDS * fs))
                        xf = np.column_stack([dsp.clean(x[-n:, i], fs, self.mains)
                                              for i in range(x.shape[1])])
                    m = len(xf)
                    tt = np.linspace(-m / fs, 0, m)
                    name_to_col = {nm: i for i, nm in enumerate(names)}
                    for ax, ln, lead in zip(self._axes, self._lines, self._leads):
                        idx = name_to_col.get(lead)
                        if idx is None or idx >= xf.shape[1]:
                            ln.set_data([tt[0], 0], [0, 0])   # lead not present (e.g. single-lead)
                            continue
                        col = xf[:, idx]
                        ln.set_data(tt, col)
                        if np.isfinite(col).any():
                            lo, hi = np.nanpercentile(col, [1, 99])
                        else:
                            lo, hi = -1, 1
                        span = max(0.6, hi - lo)
                        pad = 0.25 * span
                        mid = 0.5 * (hi + lo)
                        ax.set_ylim(mid - span / 2 - pad, mid + span / 2 + pad)
                        ax.set_xlim(tt[0], 0)
                    self.canvas.draw_idle()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.root.after(REDRAW_MS, self._pump_plot)

    # ------------------------------------------------------------------ misc
    def _open(self, path, reveal: bool = False):
        if not path:
            return
        try:
            args = ["open", "-R", str(path)] if reveal else ["open", str(path)]
            subprocess.run(args, check=False)
        except Exception:  # noqa: BLE001
            pass

    def _toast(self, msg: str):
        try:
            self.gate_status.configure(fg=GREY, text=msg)
        except Exception:  # noqa: BLE001
            pass

    def _show_error(self, msg: str):
        friendly = msg
        if "not found" in msg.lower() or "kardiamobile" in msg.lower():
            friendly = (msg + "\n\nTips:\n"
                        "• Wake the device by touching both electrodes.\n"
                        "• Close the Kardia phone app (only one device can connect).\n"
                        "• Make sure Bluetooth is on and this app has Bluetooth permission.")
        messagebox.showerror("kardia6l", friendly)
        self._reset_idle("Error")

    def _on_close(self):
        # Signal the integrated loop to stop; actual teardown + destroy happens in main().
        self._closing = True
        try:
            if self.ctrl:
                self.ctrl.stop()
        except Exception:  # noqa: BLE001
            pass
        if self._loop is None:
            # legacy path (no integrated loop): tear down and destroy immediately
            try:
                if self.ctrl:
                    self.ctrl.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self.root.destroy()


def main():
    print_disclaimer()
    sim = "--sim" in sys.argv
    selftest = "--selftest" in sys.argv
    seconds = None
    if "--seconds" in sys.argv:
        try:
            seconds = float(sys.argv[sys.argv.index("--seconds") + 1])
        except (IndexError, ValueError):
            seconds = None

    root = tk.Tk()
    # Run one asyncio loop on the MAIN thread and pump Tk from it. This is required on
    # macOS: bleak/CoreBluetooth deliver callbacks that, when a background BLE thread races
    # the main-thread Tk+matplotlib renderer, corrupt the interpreter's thread state and
    # crash (Fatal Python error: GIL not held). Sharing one main-thread loop avoids that.
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = App(root, sim=sim, loop=loop)
    app._selftest = selftest
    if "--single" in sys.argv:
        app.single_var.set(True)
        app._on_single_toggle()

    # convenience: auto-connect on launch (sim always; real device with --connect)
    if sim or "--connect" in sys.argv:
        root.after(300, app._connect)

    if seconds:
        root.after(int(seconds * 1000), app._on_close)

    async def _tk_pump():
        try:
            while not app._closing:
                root.update()
                await asyncio.sleep(1 / 60)
        except tk.TclError:
            pass  # window destroyed out from under us

    try:
        loop.run_until_complete(_tk_pump())
        # graceful teardown of any live controller before closing the loop
        if app.ctrl is not None:
            try:
                loop.run_until_complete(asyncio.wait_for(app.ctrl.aclose(), 6))
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
        # cancel any tasks bleak/asyncio left pending, let them unwind, then close cleanly
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:  # noqa: BLE001
            pass
        loop.close()


if __name__ == "__main__":
    main()
