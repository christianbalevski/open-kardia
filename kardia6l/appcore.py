"""AppController — a thread-safe bridge between the async BLE pipeline and a GUI.

The BLE stack (bleak + asyncio) runs on its own event-loop thread. The GUI thread only
ever calls the plain methods below and reads snapshots via `latest()` / `stats()`; it never
touches asyncio. All callbacks (on_status/on_error) are marshalled to the GUI via a
thread-safe queue the GUI drains on its own timer (see `poll_events`).

Typical GUI use:
    ctrl = AppController(sim=False)
    ctrl.start()                       # scan -> connect -> handshake -> stream (background)
    ... on a Tk timer:
        for kind, payload in ctrl.poll_events(): ...   # 'status','error','connected','stopped'
        t, x, names = ctrl.latest(6.0)                 # for the live plot
        s = ctrl.stats()                               # hr, sqi, rate, samples, gaps
    ctrl.start_saving("rec", ["csv","wfdb"])           # attach file sinks mid-stream
    ctrl.stop()
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import dsp
from .protocol import KNOWN_FORMAT


@dataclass
class Snapshot:
    state: str = "idle"
    connected: bool = False
    device_name: str | None = None
    device_addr: str | None = None
    samples: int = 0
    rate_hz: float = 0.0
    seq_gaps: int = 0
    dropped: int = 0
    stalls: int = 0
    hr_bpm: float = float("nan")
    sqi: float = float("nan")
    readable: bool = False
    elapsed_s: float = 0.0
    saving_to: str | None = None
    error: str | None = None


class _SliceSink:
    """Wraps a Sink so live pipeline writes are sliced to a single column (single-lead mode).
    `open` is assumed already called on the inner sink; only write/close are delegated."""

    def __init__(self, inner, col: int = 0):
        self._inner = inner
        self._col = col

    def open(self, fs, names):
        self._inner.open(fs, names)

    def write(self, t, x):
        if getattr(x, "ndim", 1) == 2 and x.shape[1] > self._col:
            x = x[:, self._col:self._col + 1]
        self._inner.write(t, x)

    def close(self):
        self._inner.close()


class AppController:
    """Runs the kardia6l pipeline on a background asyncio thread; GUI-safe accessors."""

    def __init__(self, sim: bool = False, mains: float = 60.0, derive: bool = True,
                 sim_hr: float = 72.0, fs: float | None = None, single_lead: bool = False):
        self.sim = sim
        self.mains = mains
        # single_lead: device held in the hands only (no left-leg contact) -> only Lead I is
        # valid, so we never derive the augmented leads or an axis from the (floating) Lead II.
        self.single_lead = single_lead
        self.derive = derive and not single_lead
        self.sim_hr = sim_hr
        self.fmt = KNOWN_FORMAT
        self.fs = fs or KNOWN_FORMAT.sample_rate_hz or 300.0

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._own_thread = True     # flipped by attach_loop() for main-thread integration
        self._pipe = None
        self._conn = None
        self._sim_task = None
        self._events: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._state = "idle"
        self._snap = Snapshot()
        self._file_sinks: list = []
        self._t_start = 0.0
        self._stopping = False
        # quality-gated recording state
        self._library = None
        self._gate_task = None
        self._gate_cancel = False
        self._gate_state = "idle"   # idle|waiting|recording|analyzing|done|cancelled|error

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, payload: Any = None):
        self._events.put((kind, payload))

    def poll_events(self):
        """Drain queued (kind, payload) events. Call from the GUI timer."""
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, s: str):
        self._state = s
        self._emit("status", s)

    # ------------------------------------------------------------------ lifecycle
    def attach_loop(self, loop):
        """Run on an externally-owned event loop (e.g. one integrated with a Tk main loop
        on the MAIN thread). Required on macOS: bleak/CoreBluetooth callbacks racing a
        background thread against a main-thread GUI crash the interpreter, so the GUI runs
        the loop cooperatively on the main thread instead of a worker thread."""
        self._loop = loop
        self._own_thread = False

    def _ensure_loop(self):
        if not getattr(self, "_own_thread", True):
            if self._loop is None:
                raise RuntimeError("attach_loop() was not called")
            return
        if self._thread and self._thread.is_alive():
            return
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, name="kardia6l-ble", daemon=True)
        self._thread.start()
        ready.wait(5)

    def _submit(self, coro):
        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ------------------------------------------------------------------ scan
    def scan(self, secs: float = 8.0):
        """Blocking scan (call from a worker thread, not the GUI thread). Returns list[Found]."""
        self._ensure_loop()
        from .scan import scan as _scan
        fut = self._submit(_scan(secs, only_kardia=False))
        return fut.result(secs + 15)

    # ------------------------------------------------------------------ start/stop
    def start(self, address: str | None = None):
        """Scan/connect/handshake/stream (sim or real) on the background loop."""
        if self._state in ("connecting", "streaming"):
            return
        self._stopping = False
        self._set_state("connecting")
        self._submit(self._run(address))

    async def _run(self, address):
        from .stream import Pipeline
        names = ["I", "II"] if self.fmt.channels >= 2 else ["I"]
        self._pipe = Pipeline(self.fmt, self.fs, [], names, derive_leads=self.derive)
        self._pipe.start()
        self._t_start = time.monotonic()
        try:
            if self.sim:
                from .sim import SimDevice
                src = SimDevice(self.fmt, self.fs, hr_bpm=self.sim_hr, per_notification=25)
                self._snap.device_name = "SIMULATED 6L"
                self._set_state("streaming")
                self._emit("connected", ("SIMULATED 6L", "sim"))
                self._sim_task = asyncio.create_task(src.run(self._pipe.on_notify))
            else:
                from .connect import Connection, KARDIA_6L_START
                from .protocol import KARDIA_6L_CMD_CHAR, KARDIA_6L_DATA_CHAR
                self._conn = Connection(address=address, stall_ms=1000.0,
                                        on_stall=lambda ms: self._emit("stall", ms))
                self._set_state("connecting")
                await self._conn.connect()
                dev = self._conn.state.device
                self._snap.device_name = getattr(dev, "name", None)
                self._snap.device_addr = getattr(dev, "address", None)
                await self._conn.subscribe(KARDIA_6L_DATA_CHAR, self._pipe.on_notify)
                for cmd in KARDIA_6L_START:
                    await self._conn.write(KARDIA_6L_CMD_CHAR, cmd)
                    await asyncio.sleep(0.3)
                self._conn.start_watchdog()
                self._set_state("streaming")
                self._emit("connected", (self._snap.device_name, self._snap.device_addr))
        except Exception as e:  # noqa: BLE001
            self._snap.error = str(e)
            self._set_state("error")
            self._emit("error", str(e))
            await self._teardown()

    def stop(self):
        if self._state in ("idle", "stopped"):
            return
        self._stopping = True
        if not getattr(self, "_own_thread", True):
            # external (main-thread) loop: never block the caller — schedule teardown.
            self._submit(self._teardown_and_flag())
            return
        fut = self._submit(self._teardown())
        try:
            fut.result(8)
        except Exception:  # noqa: BLE001
            pass
        self._set_state("stopped")
        self._emit("stopped", None)

    async def _teardown_and_flag(self):
        await self._teardown()
        self._set_state("stopped")
        self._emit("stopped", None)

    async def _teardown(self):
        if self._sim_task:
            self._sim_task.cancel()
            self._sim_task = None
        if self._conn:
            try:
                await self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        if self._pipe:
            try:
                await self._pipe.stop()
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self):
        """Full stop. On an owned background thread this also stops the loop; on an external
        (main-thread) loop the owner drives shutdown via `aclose()` + loop teardown."""
        if not getattr(self, "_own_thread", True):
            self.stop()
            return
        try:
            self.stop()
        finally:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

    async def aclose(self):
        """Await a full teardown (for the external-loop owner to run before closing the loop)."""
        self._stopping = True
        await self._teardown()

    # ------------------------------------------------------------------ saving
    def start_saving(self, path: str, formats=("csv",)):
        """Attach file sinks to the running pipeline. Returns the list of paths written."""
        if not self._pipe:
            return []
        from pathlib import Path
        p = Path(path)
        out_names = self._out_names()
        sinks, paths = [], []
        for f in formats:
            f = f.lower()
            if f == "csv":
                from .sinks.csv_sink import CSVSink
                sinks.append(CSVSink(p.with_suffix(".csv"))); paths.append(str(p.with_suffix(".csv")))
            elif f == "wfdb":
                from .sinks.wfdb_sink import WFDBSink
                sinks.append(WFDBSink(p.with_suffix(""))); paths.append(str(p.with_suffix(".dat")))
            elif f == "edf":
                from .sinks.wfdb_sink import EDFSink
                sinks.append(EDFSink(p.with_suffix(".edf"))); paths.append(str(p.with_suffix(".edf")))
        for s in sinks:
            s.open(self.fs, out_names)
        # In single-lead mode the pipeline still emits 2 columns; wrap each sink so it only
        # receives Lead I (matching out_names) when written to from the live pipeline.
        attach = [_SliceSink(s) for s in sinks] if self.single_lead else sinks
        with self._lock:
            self._pipe.sinks.extend(attach)
            self._file_sinks.extend(attach)
            self._snap.saving_to = str(p)
        self._emit("saving", str(p))
        return paths

    def stop_saving(self):
        if not self._pipe:
            return
        with self._lock:
            for s in self._file_sinks:
                try:
                    self._pipe.sinks.remove(s)
                except ValueError:
                    pass
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
            self._file_sinks.clear()
            self._snap.saving_to = None
        self._emit("saved", None)

    def save_buffer(self, path: str, seconds: float | None = None, formats=("csv",)):
        """Snapshot-save the current ring buffer (what's on screen) without streaming to disk."""
        if not self._pipe:
            return []
        n = int((seconds or 1e9) * self.fs)
        t, x = self._pipe.ring.latest(n)
        if len(x) == 0:
            return []
        x = self._present(x)
        out_names = self._out_names()
        from pathlib import Path
        p = Path(path); paths = []
        for f in formats:
            f = f.lower()
            if f == "csv":
                from .sinks.csv_sink import CSVSink
                s = CSVSink(p.with_suffix(".csv"))
            elif f == "wfdb":
                from .sinks.wfdb_sink import WFDBSink
                s = WFDBSink(p.with_suffix(""))
            elif f == "edf":
                from .sinks.wfdb_sink import EDFSink
                s = EDFSink(p.with_suffix(".edf"))
            else:
                continue
            s.open(self.fs, out_names); s.write(t, x); s.close()
            paths.append(str(p))
        return paths

    # ------------------------------------------------------------------ quality-gated recording
    @property
    def library(self):
        if self._library is None:
            from .library import ReadingLibrary
            self._library = ReadingLibrary()
        return self._library

    @property
    def gate_state(self) -> str:
        return self._gate_state

    def _out_names(self):
        if self.single_lead:
            return ["I"]
        if self.derive and self.fmt.channels >= 2:
            from .leads import LEAD_NAMES_6
            return list(LEAD_NAMES_6)
        return self._pipe.channel_names if self._pipe else ["I"]

    def _present(self, x):
        """Slice pipeline output to what the controller exposes. In single-lead mode only
        Lead I (column 0) is valid — Lead II is a floating electrode with no leg contact."""
        if self.single_lead and getattr(x, "ndim", 0) == 2 and x.shape[1] >= 1:
            return x[:, :1]
        return x

    def start_gated_recording(self, seconds: float = 30.0, sqi_threshold: float = 0.9,
                              hold_s: float = 4.0, timeout_s: float = 60.0,
                              name: str | None = None, analyze: bool = True,
                              make_report: bool = True):
        """Wait until signal quality is sustained above `sqi_threshold` for `hold_s` seconds
        (a settle period so the capture does not start on the contact transient), then
        capture exactly `seconds` of ECG, save it to the library, analyze it, and render a report. Non-blocking;
        progress arrives as events: 'gate'(dict sqi/needed/state), 'rec_start', 'rec_progress'
        (fraction 0..1), 'reading_complete'(dict), 'gate_error'(msg), 'gate_cancelled'."""
        if self._state != "streaming":
            self._emit("gate_error", "not streaming — connect first")
            return
        if self._gate_task and not self._gate_task.done():
            return
        self._gate_cancel = False
        self._gate_task = self._submit(self._gated(seconds, sqi_threshold, hold_s, timeout_s,
                                                    name, analyze, make_report))

    def cancel_gated_recording(self):
        self._gate_cancel = True

    async def _gated(self, seconds, thr, hold_s, timeout_s, name, analyze, make_report):
        from .stream import Sink

        class _Capture:
            def __init__(self):
                self.chunks = []
            def open(self, fs, channels):
                self.names = list(channels)
            def write(self, t, x):
                self.chunks.append(np.asarray(x))
            def close(self):
                pass
            def data(self):
                return np.vstack(self.chunks) if self.chunks else np.zeros((0, 1))

        try:
            # 1) wait for sustained quality
            self._gate_state = "waiting"
            good_since = None
            t_deadline = time.monotonic() + timeout_s
            while not self._gate_cancel:
                q = self._current_quality()
                sqi = q.sqi if q else float("nan")
                ok = q is not None and q.sqi >= thr and q.readable
                good_since = (good_since or time.monotonic()) if ok else None
                held = (time.monotonic() - good_since) if good_since else 0.0
                self._emit("gate", {"sqi": sqi, "threshold": thr, "held_s": held,
                                    "hold_s": hold_s, "state": "waiting"})
                if held >= hold_s:
                    break
                if time.monotonic() > t_deadline:
                    self._gate_state = "error"
                    self._emit("gate_error", f"signal did not reach {int(thr*100)}% quality "
                                             f"within {int(timeout_s)}s — check electrode contact")
                    return
                await asyncio.sleep(0.25)
            if self._gate_cancel:
                self._gate_state = "cancelled"; self._emit("gate_cancelled", None); return

            # 2) capture exactly `seconds` via a temporary in-memory sink
            self._gate_state = "recording"
            cap = _Capture()
            out_names = self._out_names()
            cap.open(self.fs, out_names)
            with self._lock:
                self._pipe.sinks.append(cap)
            self._emit("rec_start", {"seconds": seconds})
            need = int(seconds * self.fs)
            t_start = time.monotonic()
            while not self._gate_cancel:
                have = sum(len(c) for c in cap.chunks)
                self._emit("rec_progress", min(1.0, have / max(need, 1)))
                if have >= need:
                    break
                # abort if connection drops mid-recording
                if not self.sim and not (self._conn and self._conn.state.connected):
                    with self._lock:
                        if cap in self._pipe.sinks:
                            self._pipe.sinks.remove(cap)
                    self._gate_state = "error"; self._emit("gate_error", "disconnected during recording")
                    return
                await asyncio.sleep(0.2)
            with self._lock:
                if cap in self._pipe.sinks:
                    self._pipe.sinks.remove(cap)
            if self._gate_cancel:
                self._gate_state = "cancelled"; self._emit("gate_cancelled", None); return
            x = self._present(cap.data()[:need])
            if len(x) < need * 0.5:
                self._gate_state = "error"; self._emit("gate_error", "too few samples captured"); return

            # 3) save + analyze + report off the event loop (don't block BLE callbacks)
            self._gate_state = "analyzing"
            self._emit("gate", {"state": "analyzing"})
            result = await self._loop.run_in_executor(
                None, self._finish_reading, x, out_names, name, analyze, make_report)
            self._gate_state = "done"
            self._emit("reading_complete", result)
        except Exception as e:  # noqa: BLE001
            self._gate_state = "error"
            self._emit("gate_error", str(e))

    def _current_quality(self):
        if not self._pipe:
            return None
        t, x = self._pipe.ring.latest(int(5 * self.fs))
        if len(x) < 2 * self.fs:
            return None
        try:
            return dsp.signal_quality(x[:, 0], self.fs)
        except Exception:  # noqa: BLE001
            return None

    def _finish_reading(self, x, out_names, name, analyze, make_report):
        """Runs in a worker thread: library.add -> analysis -> report. Returns a plain dict."""
        meta = {"source": "sim" if self.sim else "live", "mains": self.mains,
                "device": self._snap.device_name}
        res = None
        if analyze:
            try:
                from .analysis import analyze_recording
                res = analyze_recording(x, self.fs, out_names, mains=self.mains)
                meta["hr_bpm"] = res.intervals.hr_mean
                meta["rhythm"] = res.rhythm.label
                meta["sqi"] = res.quality_sqi
            except Exception as e:  # noqa: BLE001
                self._emit("gate_error", f"analysis failed: {e}")
        rec = self.library.add(x, self.fs, out_names, meta=meta, name=name)
        report_paths = {}
        if make_report:
            try:
                from .report import render_report
                from .sinks.csv_sink import read_csv
                fs, names, t, xr = read_csv(rec.csv_path())
                rmeta = {"recorded_at": rec.created_at, "duration_s": rec.duration_s,
                         "device": self._snap.device_name, "mains": self.mains}
                base = str(Path(rec.dir) / "report")
                for fmt in ("png", "pdf"):
                    report_paths[fmt] = render_report(t, xr, names, fs, analysis=res,
                                                       out_path=f"{base}.{fmt}", fmt=fmt, meta=rmeta)
            except Exception as e:  # noqa: BLE001
                self._emit("gate_error", f"report failed: {e}")
        return {"record": rec, "analysis": res, "reports": report_paths,
                "summary": res.summary() if res and hasattr(res, "summary") else ""}

    # ------------------------------------------------------------------ library access (GUI thread)
    def list_readings(self):
        try:
            return self.library.list()
        except Exception:  # noqa: BLE001
            return []

    def load_reading(self, reading_id):
        return self.library.load(reading_id)

    def delete_reading(self, reading_id):
        return self.library.delete(reading_id)

    def analyze_reading(self, reading_id):
        from .analysis import analyze_recording
        fs, names, x = self.library.load(reading_id)
        return analyze_recording(x, fs, names, mains=self.mains)

    def export_report(self, reading_id, out_path, fmt="pdf", analysis=None):
        from .report import render_report_from_csv
        rec = self.library.get(reading_id)
        if rec is None:
            return None
        if analysis is None:
            try:
                analysis = self.analyze_reading(reading_id)
            except Exception:  # noqa: BLE001
                analysis = None
        meta = {"recorded_at": rec.created_at, "duration_s": rec.duration_s, "mains": self.mains}
        return render_report_from_csv(rec.csv_path(), analysis=analysis, out_path=out_path,
                                      fmt=fmt, meta=meta)

    # ------------------------------------------------------------------ data access (GUI thread)
    def latest(self, seconds: float):
        """(t, x[n,ch], names) of the last `seconds`. Safe to call from the GUI thread."""
        if not self._pipe:
            return np.zeros(0), np.zeros((0, 1)), ["I"]
        n = int(seconds * self.fs)
        t, x = self._pipe.ring.latest(n)
        if self.single_lead:
            return t, self._present(x), ["I"]
        names = self._pipe.channel_names
        if self.derive and self.fmt.channels >= 2 and x.shape[1] == 6:
            from .leads import LEAD_NAMES_6
            names = list(LEAD_NAMES_6)
        return t, x, names

    def stats(self) -> Snapshot:
        s = self._snap
        s.state = self._state
        if self._pipe:
            st = self._pipe.stats
            s.samples = st.samples
            s.rate_hz = st.rate_est
            s.seq_gaps = st.seq_gaps
            s.dropped = st.dropped_queue
            s.elapsed_s = (time.monotonic() - self._t_start) if self._t_start else 0.0
            # HR + SQI on the last 6 s of lead I
            t, x = self._pipe.ring.latest(int(6 * self.fs))
            if len(x) >= 2 * self.fs:
                try:
                    q = dsp.signal_quality(x[:, 0], self.fs)
                    s.hr_bpm = q.hr_bpm
                    s.sqi = q.sqi
                    s.readable = q.readable
                except Exception:  # noqa: BLE001
                    pass
        s.connected = bool(self._conn and self._conn.state.connected) or (self.sim and self._state == "streaming")
        return s
