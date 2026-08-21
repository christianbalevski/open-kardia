"""Printable ECG report renderer (PNG / PDF) for open_kardia.

Pure matplotlib (Agg) — no GUI, no BLE. Renders a single-page clinical-style report
from a recording plus an optional analysis result (see open_kardia.analysis). The analysis
object is consumed defensively (duck-typed, nan/None tolerant) so a partial or absent
analysis still produces a valid page.

Public API
----------
render_report(t, x, lead_names, fs, analysis=None, out_path=..., fmt="png",
              meta=None, title=...) -> str
render_report_from_csv(csv_path, analysis=None, out_path=..., fmt="png", meta=None) -> str
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from . import dsp  # noqa: E402
from .leads import LEAD_NAMES_6  # noqa: E402

# --- palette (light, print-friendly) -------------------------------------------------
BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#b3123b"          # ECG red
GRID_MAJOR = "#f4b9c6"      # 5 mm pink
GRID_MINOR = "#f9dde3"      # 1 mm pink
TRACE = "#111111"
PANEL = "#fbfbfc"
HEADER_BG = "#f4f5f7"


# --- defensive accessor --------------------------------------------------------------
def _get(obj, path, default=None):
    """Fetch a dotted attribute / dict-key path from a possibly-None object.

    _get(analysis, "intervals.pr_ms")  ->  analysis.intervals.pr_ms  (or default)
    Works across attribute access and mapping access, tolerating None at any hop.
    """
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        nxt = None
        if isinstance(cur, dict):
            nxt = cur.get(part, None)
        else:
            nxt = getattr(cur, part, None)
        cur = nxt
    return default if cur is None else cur


def _fmt(v, unit="", nd=0):
    """Format a numeric value, showing an em dash for None/NaN."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return "—"
    if nd == 0:
        s = f"{f:.0f}"
    else:
        s = f"{f:.{nd}f}"
    return f"{s}{unit}" if unit else s


def _clean_stable(col, fs, mains, pad_s=1.0):
    """Edge-stable display filter: reflect-pad, clean, then trim. Avoids the shaky
    filter transients at the window ends. Falls back to plain clean on any failure."""
    col = np.asarray(col, dtype=np.float64)
    if col.size == 0 or not np.all(np.isfinite(col)):
        return col
    try:
        pad = min(int(pad_s * fs), max(0, col.size - 1))
        if pad > 0:
            ext = np.concatenate([col[pad:0:-1], col, col[-2:-pad - 2:-1]])
            out = dsp.clean(ext, fs, mains)
            return out[pad:pad + col.size]
        return dsp.clean(col, fs, mains)
    except Exception:  # noqa: BLE001
        try:
            return dsp.clean(col, fs, mains)
        except Exception:  # noqa: BLE001
            return col - np.median(col)


def _as_dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return str(v)


# --- ECG grid ------------------------------------------------------------------------
def _ecg_grid(ax, t0, t1, y_center, y_half_mv):
    """Draw the classic pink ECG grid: 0.2 s / 0.5 mV major, 0.04 s / 0.1 mV minor."""
    ax.set_facecolor(BG)
    # minor: 0.04 s and 0.1 mV
    xmin = math.floor(t0 / 0.04) * 0.04
    for xx in np.arange(xmin, t1 + 1e-9, 0.04):
        ax.axvline(xx, color=GRID_MINOR, lw=0.4, zorder=0)
    ylo, yhi = y_center - y_half_mv, y_center + y_half_mv
    ymin = math.floor(ylo / 0.1) * 0.1
    for yy in np.arange(ymin, yhi + 1e-9, 0.1):
        ax.axhline(yy, color=GRID_MINOR, lw=0.4, zorder=0)
    # major: 0.2 s and 0.5 mV
    xmaj = math.floor(t0 / 0.2) * 0.2
    for xx in np.arange(xmaj, t1 + 1e-9, 0.2):
        ax.axvline(xx, color=GRID_MAJOR, lw=0.7, zorder=0)
    ymaj = math.floor(ylo / 0.5) * 0.5
    for yy in np.arange(ymaj, yhi + 1e-9, 0.5):
        ax.axhline(yy, color=GRID_MAJOR, lw=0.7, zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_ylim(ylo, yhi)
    for s in ax.spines.values():
        s.set_color(GRID_MAJOR)
        s.set_linewidth(0.7)
    ax.set_xticks([])
    ax.set_yticks([])


# --- header --------------------------------------------------------------------------
def _draw_header(fig, gs, title, meta, analysis):
    ax = fig.add_subplot(gs)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=HEADER_BG, edgecolor="none", zorder=0))
    meta = meta or {}
    ax.text(0.012, 0.80, title, transform=ax.transAxes, fontsize=15, fontweight="bold",
            color=INK, va="center")
    patient = meta.get("patient_name")
    rec = _as_dt(meta.get("recorded_at"))
    rec_s = rec.strftime("%Y-%m-%d %H:%M") if isinstance(rec, datetime) else (rec or "—")
    dur = meta.get("duration_s") or _get(analysis, "duration_s")
    dev = meta.get("device") or "KardiaMobile 6L (AC-019)"
    sub = f"{dev}    Recorded: {rec_s}    Duration: {_fmt(dur, ' s')}"
    if patient:
        sub = f"{patient}    " + sub
    ax.text(0.012, 0.40, sub, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="center")

    # HR block (center)
    hr_mean = _get(analysis, "intervals.hr_mean")
    hr_min = _get(analysis, "intervals.hr_min")
    hr_max = _get(analysis, "intervals.hr_max")
    ax.text(0.66, 0.72, "HR", transform=ax.transAxes, fontsize=8, color=MUTED, va="center")
    ax.text(0.66, 0.36, _fmt(hr_mean), transform=ax.transAxes, fontsize=26,
            fontweight="bold", color=ACCENT, va="center")
    ax.text(0.745, 0.30, "bpm", transform=ax.transAxes, fontsize=9, color=MUTED, va="center")
    if hr_min is not None or hr_max is not None:
        ax.text(0.66, 0.10, f"min {_fmt(hr_min)} · max {_fmt(hr_max)}",
                transform=ax.transAxes, fontsize=7.5, color=MUTED, va="center")

    # rhythm + axis (right)
    rhythm = _get(analysis, "rhythm.label", "—")
    conf = _get(analysis, "rhythm.confidence")
    axis_lbl = _get(analysis, "axis_label", "—")
    axis_deg = _get(analysis, "qrs_axis_deg")
    rl = str(rhythm)
    rfs = 12.5 if len(rl) <= 24 else (10.5 if len(rl) <= 32 else 9.0)
    ax.text(0.985, 0.74, rl, transform=ax.transAxes, fontsize=rfs,
            fontweight="bold", color=INK, va="center", ha="right")
    if conf is not None:
        ax.text(0.985, 0.50, f"confidence {_fmt(float(conf) * 100 if conf <= 1 else conf, '%')}",
                transform=ax.transAxes, fontsize=8, color=MUTED, va="center", ha="right")
    ax.text(0.985, 0.26, f"Axis: {axis_lbl}" + (f" ({_fmt(axis_deg, '°')})" if axis_deg is not None else ""),
            transform=ax.transAxes, fontsize=8.5, color=INK, va="center", ha="right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# --- intervals / measurements table --------------------------------------------------
def _draw_measurements(fig, gs, analysis):
    ax = fig.add_subplot(gs)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=PANEL, edgecolor="#e5e5e8", lw=1.0, zorder=0))
    ax.text(0.02, 0.94, "Measurements", transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=INK, va="center")

    hrv = _get(analysis, "hrv", {}) or {}

    def _hrv(*keys):
        if not isinstance(hrv, dict):
            return None
        low = {str(k).lower(): v for k, v in hrv.items()}
        for k in keys:
            if k.lower() in low:
                return low[k.lower()]
        return None

    # Normal-range hints come from the analysis module's single source of truth.
    try:
        from open_kardia.analysis import range_text as _rng
    except Exception:  # noqa: BLE001
        def _rng(_key):
            return ""

    cells = [
        ("PR", _fmt(_get(analysis, "intervals.pr_ms"), " ms"), _rng("pr_ms")),
        ("QRS", _fmt(_get(analysis, "intervals.qrs_ms"), " ms"), _rng("qrs_ms")),
        ("QT", _fmt(_get(analysis, "intervals.qt_ms"), " ms"), "rate-dependent"),
        ("QTc (Bazett)", _fmt(_get(analysis, "intervals.qtc_bazett_ms"), " ms"), _rng("qtc_bazett_ms")),
        ("QTc (Fridericia)", _fmt(_get(analysis, "intervals.qtc_fridericia_ms"), " ms"),
         _rng("qtc_fridericia_ms")),
        ("P duration", _fmt(_get(analysis, "intervals.p_dur_ms"), " ms"), _rng("p_dur_ms")),
        ("ST level", _fmt(_get(analysis, "intervals.st_level_mv"), " mV", nd=2), _rng("st_level_mv")),
        ("RR mean±SD", f"{_fmt(_get(analysis, 'intervals.rr_mean_ms'))}±"
                       f"{_fmt(_get(analysis, 'intervals.rr_sd_ms'))} ms", _rng("rr_mean_ms")),
        ("QRS axis", _fmt(_get(analysis, "qrs_axis_deg"), "°"), _rng("qrs_axis_deg")),
        ("SDNN", _fmt(_hrv("sdnn_ms", "sdnn", "SDNN"), " ms"), ""),
        ("RMSSD", _fmt(_hrv("rmssd_ms", "rmssd", "RMSSD"), " ms"), ""),
        ("pNN50", _fmt(_hrv("pnn50", "pnn50_pct", "pnn50%", "pNN50"), "%", nd=1), ""),
    ]
    ncol = 4
    nrow = math.ceil(len(cells) / ncol)
    x_w = 1.0 / ncol
    y_top = 0.83
    y_h = y_top / nrow
    for i, (label, val, hint) in enumerate(cells):
        r, c = divmod(i, ncol)
        cx = c * x_w + 0.02
        cy = y_top - r * y_h
        ax.text(cx, cy, label, transform=ax.transAxes, fontsize=6.8, color=MUTED, va="top")
        ax.text(cx, cy - 0.28 * y_h, val, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=INK, va="top")
        if hint:
            ax.text(cx, cy - 0.68 * y_h, hint, transform=ax.transAxes, fontsize=5.8,
                    color="#b7b7bd", va="top", style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# --- strips --------------------------------------------------------------------------
def _draw_strips(fig, gs, t, x, lead_names, fs, mains, window_s):
    n = x.shape[0]
    win = min(int(window_s * fs), n)
    seg = x[:win, :]
    tt = (t[:win] - t[0]) if t is not None and len(t) >= win else np.arange(win) / fs
    t0, t1 = float(tt[0]), float(tt[-1])
    nlead = len(lead_names)
    inner = gs.subgridspec(nlead, 1, hspace=0.12)
    axes = []
    for i, name in enumerate(lead_names):
        ax = fig.add_subplot(inner[i])
        col = _clean_stable(seg[:, i], fs, mains)
        finite = col[np.isfinite(col)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            half = max(0.75, 0.6 * (hi - lo), abs(np.median(finite)) + 0.5)
        else:
            half = 1.0
        center = float(np.median(finite)) if finite.size else 0.0
        _ecg_grid(ax, t0, t1, center, half)
        ax.plot(tt, col, color=TRACE, lw=0.7, zorder=3)
        ax.text(0.004, 0.90, name, transform=ax.transAxes, fontsize=9, fontweight="bold",
                color=ACCENT, va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        axes.append(ax)
    axes[-1].text(0.995, -0.02, "25 mm/s · 10 mm/mV", transform=axes[-1].transAxes,
                  fontsize=7, color=MUTED, ha="right", va="top")
    axes[0].set_title(f"ECG strips — first {win / fs:.0f} s (full recording at true scale on the following pages)"
                      if n > win else f"ECG strips — {win / fs:.0f} s (full recording)",
                      fontsize=9, color=INK, loc="left", pad=4)
    return axes


# --- full-disclosure pages (true paper scale) -------------------------------------------
PAGE_SECONDS = 10.0          # standard ECG paper: 10 s per page at 25 mm/s
MM_PER_S = 25.0
MM_PER_MV = 10.0
CAL_S = 0.4                  # lead-in with the 1 mV calibration pulse
A4_L = (11.69, 8.27)         # landscape, inches
A4_L_MM = (297.0, 210.0)


def _cal_pulse(ax, t_lead_in, y0):
    """Standard 1 mV x 0.2 s calibration pulse drawn in the lead-in before the trace."""
    a = t_lead_in + 0.08
    xs = [t_lead_in, a, a, a + 0.2, a + 0.2, t_lead_in + CAL_S]
    ys = [y0, y0, y0 + 1.0, y0 + 1.0, y0, y0]
    ax.plot(xs, ys, color=TRACE, lw=0.8, zorder=3)


def _page_layout(nlead, half, avail_mm, gap_mm=3.0, group_gap_mm=9.0):
    """How many 10-s row groups (all leads each) fit on one landscape page at true scale,
    and the mm/mV to use (reduced uniformly only when even one group would not fit)."""
    mm_per_mv = MM_PER_MV
    group_mm = nlead * 2 * half * mm_per_mv + (nlead - 1) * gap_mm
    if group_mm > avail_mm:
        mm_per_mv = (avail_mm - (nlead - 1) * gap_mm) / (nlead * 2 * half)
        return 1, mm_per_mv
    groups = int((avail_mm + group_gap_mm) // (group_mm + group_gap_mm))
    return max(1, groups), mm_per_mv


def _lane_scale(xf, fs, lead_names, t_start, t_end):
    """Per-lead centre (mV) and a common half-height (mV, >= 1.25) for a time range."""
    i0, i1 = int(round(t_start * fs)), min(xf.shape[0], int(round(t_end * fs)))
    seg = xf[i0:i1, :]
    centers, halves = [], []
    for i in range(len(lead_names)):
        col = seg[:, i]
        finite = col[np.isfinite(col)]
        c = float(np.median(finite)) if finite.size else 0.0
        if finite.size:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            need = max(abs(hi - c), abs(lo - c)) * 1.08
        else:
            need = 1.0
        centers.append(c)
        halves.append(max(1.25, need))
    return centers, max(halves)


EPISODE_COLORS = {"af": "#c62828", "vt": "#c62828", "vt_run": "#c62828", "pause": "#c62828", "svt": "#d9480f",
                  "svt_run": "#b26a00", "couplet": "#d9480f", "bigeminy": "#d9480f", "trigeminy": "#d9480f"}
EPISODE_LABELS = {"af": "AF?", "vt": "VT?", "vt_run": "V run", "pause": "pause", "svt": "SVT?", "svt_run": "S run",
                  "couplet": "couplet", "bigeminy": "bigeminy", "trigeminy": "trigeminy"}


def _episodes_for_report(analysis):
    """[(t0, t1, label, color)] spans to shade on the full-disclosure pages: pattern findings
    (bigeminy/trigeminy/couplets/runs/pauses) and AF episodes. Single ectopic beats are
    marked V/S instead."""
    out = []
    for f in list(_get(analysis, "arrhythmia.findings") or []):
        code = str(_get(f, "code", ""))
        if code not in EPISODE_COLORS:
            continue
        for a, b in list(_get(f, "episodes") or []):
            out.append((float(a), float(b), EPISODE_LABELS[code], EPISODE_COLORS[code]))
    af = _get(analysis, "af")
    if af is not None and _get(af, "is_af"):
        for ep in list(_get(af, "episodes") or []):
            out.append((float(ep[0]), float(ep[1]), "AF?", EPISODE_COLORS["af"]))
    return out


def _draw_full_page(fig, xf, lead_names, fs, groups, page_no, n_pages, meta, title,
                    labels_at=None, half=None, mm_per_mv=None, episodes=None):
    """One A4-landscape page of full disclosure: `groups` = [(t_start, t_end), ...] row groups
    (each = every lead over 10 s) at 25 mm/s and `mm_per_mv` (10 mm/mV unless reduced —
    and stated — to fit), with a 1 mV calibration pulse per lane."""
    W, H = A4_L_MM
    nlead = len(lead_names)
    t_first, t_last = groups[0][0], groups[-1][1]

    # ---- header
    ax = fig.add_axes([0.03, 0.935, 0.94, 0.05])
    ax.axis("off")
    ax.text(0.0, 0.7, title, fontsize=11, fontweight="bold", color=INK, va="center", transform=ax.transAxes)
    rec = _as_dt((meta or {}).get("recorded_at"))
    rec_s = rec.strftime("%Y-%m-%d %H:%M") if isinstance(rec, datetime) else (rec or "—")
    dev = (meta or {}).get("device") or "KardiaMobile 6L (AC-019)"
    ax.text(0.0, 0.1, f"{dev}    Recorded: {rec_s}    Full disclosure {t_first:.0f}–{t_last:.0f} s",
            fontsize=7.5, color=MUTED, va="center", transform=ax.transAxes)
    ax.text(1.0, 0.7, f"page {page_no} / {n_pages}", fontsize=8.5, color=INK, va="center", ha="right",
            transform=ax.transAxes)

    # ---- geometry (true scale)
    top_mm, bottom_mm, gap_mm, group_gap_mm = H - 20.0, 12.0, 3.0, 9.0
    if half is None:
        _c, half = _lane_scale(xf, fs, lead_names, t_first, t_last)
    if mm_per_mv is None:
        _g, mm_per_mv = _page_layout(nlead, half, top_mm - bottom_mm, gap_mm, group_gap_mm)
    lane_mm = 2 * half * mm_per_mv

    y_cursor = top_mm
    last_ax = None
    for (t_start, t_end) in groups:
        centers, _h = _lane_scale(xf, fs, lead_names, t_start, t_end)
        i0, i1 = int(round(t_start * fs)), min(xf.shape[0], int(round(t_end * fs)))
        seg = xf[i0:i1, :]
        tt = np.arange(i0, i1) / fs
        trace_w_mm = (PAGE_SECONDS + CAL_S) * MM_PER_S          # constant width -> aligned rows
        left_mm = (W - trace_w_mm) / 2.0
        span = t_start - CAL_S, t_start + PAGE_SECONDS
        for i, name in enumerate(lead_names):
            y_top = y_cursor
            y_bot = y_top - lane_mm
            y_cursor = y_bot - gap_mm
            ax = fig.add_axes([left_mm / W, y_bot / H, trace_w_mm / W, lane_mm / H])
            c = centers[i]
            _ecg_grid(ax, span[0], span[1], c, half)
            _cal_pulse(ax, t_start - CAL_S, c - 0.5)
            for (ea, eb, elab, ecol) in (episodes or []):
                if eb < t_start or ea > t_end:
                    continue
                a, b = max(ea, t_start), min(eb, t_end)
                if b - a < 0.05:
                    a, b = a - 0.05, b + 0.05
                ax.axvspan(a, b, color=ecol, alpha=0.10, lw=0, zorder=1)
                if i == 0:      # label at the lane's lower edge (the lead name owns the upper-left corner)
                    ax.text(a + 0.05, c - half * 0.97, elab, fontsize=6, color=ecol, fontweight="bold",
                            ha="left", va="bottom", zorder=4)
            ax.plot(tt, seg[:, i], color=TRACE, lw=0.7, zorder=3)
            ax.text(0.004 + CAL_S / (PAGE_SECONDS + CAL_S), 0.92, name, transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", color=ACCENT, va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
            if labels_at and i == 0:      # ectopic-beat labels above the top lane of the group
                for tb, lab in labels_at:
                    if t_start <= tb <= t_end:
                        ax.text(tb, c + half * 1.02, lab, fontsize=6.5, fontweight="bold", clip_on=False,
                                color="#d9480f" if lab == "V" else "#b26a00", ha="center", va="bottom", zorder=4)
            last_ax = ax
        # second ticks under the last lane of the group
        for sec in range(int(math.ceil(t_start)), int(math.floor(t_start + PAGE_SECONDS)) + 1):
            if sec > t_end + 1e-9:
                break
            last_ax.text(sec, last_ax.get_ylim()[0] - half * 0.06, f"{sec:d} s", fontsize=6, color=MUTED,
                         ha="center", va="top", clip_on=False)
        y_cursor -= (group_gap_mm - gap_mm)
    scale_txt = f"{MM_PER_S:.0f} mm/s · {mm_per_mv:.1f} mm/mV" + ("" if abs(mm_per_mv - MM_PER_MV) < 1e-6
                                                                    else " (gain reduced to fit)")
    fig.text(0.97, 0.03, scale_txt, fontsize=7, color=MUTED, ha="right", va="bottom")
    fig.text(0.03, 0.03, "RESEARCH SIGNAL — NOT A MEDICAL DEVICE. Not clinically validated; not for diagnosis.",
             fontsize=7, color=ACCENT, ha="left", va="bottom")


# --- median beat panel ---------------------------------------------------------------
def _draw_median_beat(fig, gs, analysis):
    mb = _get(analysis, "median_beat")
    ax = fig.add_subplot(gs)
    if not isinstance(mb, dict) or not mb:
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor=PANEL, edgecolor="#e5e5e8", lw=1.0))
        ax.text(0.5, 0.5, "median beat — n/a", transform=ax.transAxes, fontsize=8,
                color=MUTED, ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax
    fs_mb = _get(analysis, "median_beat_fs", 300.0) or 300.0
    ax.set_facecolor(PANEL)
    cmap = plt.cm.viridis
    keys = list(mb.keys())
    for j, k in enumerate(keys):
        arr = np.asarray(mb[k], dtype=np.float64)
        if arr.size == 0:
            continue
        tb = (np.arange(arr.size) - arr.size * 0.3) / fs_mb * 1000.0  # ms, R near 0
        ax.plot(tb, arr, lw=1.1, color=cmap(j / max(1, len(keys) - 1)), label=str(k))
    ax.axvline(0, color=MUTED, lw=0.6, ls="--", alpha=0.6)
    ax.set_xlabel("ms (R-aligned)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("mV", fontsize=7.5, color=MUTED)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    ax.set_title(f"Median beat  (n={_fmt(_get(analysis, 'n_beats'))})", fontsize=8.5,
                 color=INK, loc="left")
    ax.legend(fontsize=6, ncol=3, loc="upper right", frameon=False)
    for s in ax.spines.values():
        s.set_color("#e0e0e4")
    return ax


# --- quality / rhythm summary box ----------------------------------------------------
def _all_findings(analysis):
    """AF (if suggested) + beat-level findings as uniform dicts for printing."""
    out = []
    af = _get(analysis, "af")
    if af is not None and _get(af, "is_af"):
        eps = list(_get(af, "episodes") or [])
        when_af = ("episodes " + ", ".join(f"{a:.0f}–{b:.0f} s" for a, b, *_ in eps[:8])
                   + (f" (+{len(eps) - 8} more)" if len(eps) > 8 else "")) if eps else "whole recording"
        out.append({"severity": "urgent", "label": _get(af, "label") or "Possible atrial fibrillation",
                    "evidence": _get(af, "evidence") or "", "when": when_af,
                    "confidence": _get(af, "confidence")})
    for f in list(_get(analysis, "arrhythmia.findings") or []):
        out.append({"severity": str(_get(f, "severity", "info")), "label": str(_get(f, "label", "")),
                    "evidence": str(_get(f, "evidence", "") or ""),
                    "when": (f.when(12) if hasattr(f, "when") else ""), "confidence": _get(f, "confidence")})
    return out


def _draw_findings_page(fig, analysis, meta, title, page_no, n_pages):
    """A4-portrait page listing every rhythm finding with severity, confidence, evidence and
    times/episodes in full (used when the summary panel cannot fit them all)."""
    import textwrap
    fig.text(0.05, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.94, "Rhythm findings — automated screen (research aid, NOT a diagnosis)",
             fontsize=9, color=MUTED, va="top")
    fig.text(0.95, 0.965, f"page {page_no} / {n_pages}", fontsize=8.5, color=INK, va="top", ha="right")
    sev_col = {"urgent": "#c62828", "notable": "#b26a00", "info": MUTED}
    y = 0.90
    for f in _all_findings(analysis):
        if y < 0.08:
            fig.text(0.05, y, "… list truncated", fontsize=8, color=MUTED, va="top")
            break
        conf = f.get("confidence")
        conf_s = f"  ({float(conf) * 100:.0f} %)" if isinstance(conf, (int, float)) and math.isfinite(float(conf)) else ""
        fig.text(0.05, y, f["severity"].upper(), fontsize=8, fontweight="bold",
                 color=sev_col.get(f["severity"], MUTED), va="top")
        for ln in textwrap.wrap(f["label"] + conf_s, 90) or [""]:
            fig.text(0.16, y, ln, fontsize=9.5, fontweight="bold", color=INK, va="top"); y -= 0.020
        for ln in textwrap.wrap("When: " + (f["when"] or "whole recording"), 110):
            fig.text(0.16, y, ln, fontsize=8, color=ACCENT, va="top"); y -= 0.017
        for ln in textwrap.wrap(f["evidence"], 115):
            fig.text(0.16, y, ln, fontsize=8, color=MUTED, va="top"); y -= 0.017
        y -= 0.014
    fig.text(0.5, 0.03, "RESEARCH SIGNAL — NOT A MEDICAL DEVICE. Not clinically validated; not for diagnosis.",
             fontsize=9, fontweight="bold", color=ACCENT, ha="center", va="bottom")


def _draw_quality(fig, gs, analysis):
    ax = fig.add_subplot(gs)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=PANEL, edgecolor="#e5e5e8", lw=1.0, zorder=0))
    ax.text(0.05, 0.90, "Summary", transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=INK, va="top")
    sqi = _get(analysis, "quality_sqi")
    lead_used = _get(analysis, "lead_used")
    nbeats = _get(analysis, "n_beats")
    rows = [
        ("Signal quality", _fmt(float(sqi) * 100, "%") if isinstance(sqi, (int, float)) else "\u2014"),
        ("Analysis lead", str(lead_used) if lead_used else "\u2014"),
        ("Beats analysed", _fmt(nbeats)),
    ]
    findings = _all_findings(analysis)
    step = 0.075 if findings else 0.20
    y = 0.80 if findings else 0.66
    for label, val in rows:
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
        ax.text(0.95, y, val, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                color=INK, va="top", ha="right")
        y -= step
    # rhythm findings from the arrhythmia screen (ectopy, pauses, conduction, AF)
    if findings:
        sev_col = {"urgent": "#c62828", "notable": "#b26a00", "info": MUTED}
        y -= 0.01
        ax.text(0.05, y, "Rhythm findings (automated screen — not a diagnosis)",
                transform=ax.transAxes, fontsize=6.6, fontweight="bold", color=INK, va="top")
        y -= 0.07
        import textwrap
        y_floor = 0.05
        shown = 0
        for f in findings:
            sev = str(_get(f, "severity", "info"))
            lab = str(_get(f, "label", ""))
            when = _get(f, "when") or ""
            ev = str(_get(f, "evidence", "") or "")
            lab_lines = textwrap.wrap(lab, 62) or [""]
            sub_lines = textwrap.wrap(" — ".join(x for x in (when, ev) if x), 92)[:3]
            need = 0.046 * len(lab_lines) + 0.040 * len(sub_lines) + 0.014
            if y - need < y_floor:
                break
            ax.text(0.05, y, sev.upper(), transform=ax.transAxes, fontsize=5.6,
                    fontweight="bold", color=sev_col.get(sev, MUTED), va="top")
            for ln in lab_lines:
                ax.text(0.20, y, ln, transform=ax.transAxes, fontsize=6.3, color=INK, va="top")
                y -= 0.046
            for ln in sub_lines:
                ax.text(0.20, y, ln, transform=ax.transAxes, fontsize=5.0, color=MUTED, va="top")
                y -= 0.040
            y -= 0.014
            shown += 1
        if len(findings) > shown:
            ax.text(0.05, max(y, 0.02), f"… +{len(findings) - shown} more — all findings on page 2",
                    transform=ax.transAxes, fontsize=5.8, color=MUTED, va="top")
        ax._kardia_findings_overflow = len(findings) > shown
    flags = _get(analysis, "rhythm.flags")
    if flags and not findings:
        ftxt = ", ".join(flags) if isinstance(flags, (list, tuple)) else str(flags)
        ax.text(0.05, 0.06, f"Flags: {ftxt}", transform=ax.transAxes, fontsize=6.5,
                color=ACCENT, va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# --- footer --------------------------------------------------------------------------
def _draw_footer(fig, gs, analysis, meta):
    ax = fig.add_subplot(gs)
    ax.axis("off")
    mains = (meta or {}).get("mains", 60.0)
    warns = _get(analysis, "warnings")
    rationale = _get(analysis, "rhythm.rationale")
    line2 = f"0.5–40 Hz band-pass · {int(mains)} Hz notch (display)."
    if rationale:
        line2 = f"Rhythm rationale: {rationale}   |   " + line2
    if warns:
        wtxt = "; ".join(warns) if isinstance(warns, (list, tuple)) else str(warns)
        line2 += f"   Warnings: {wtxt}"
    ax.text(0.5, 0.62,
            "RESEARCH SIGNAL — NOT A MEDICAL DEVICE. Not clinically validated; not for diagnosis.",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=ACCENT,
            ha="center", va="center")
    ax.text(0.5, 0.18, line2, transform=ax.transAxes, fontsize=6.8, color=MUTED,
            ha="center", va="center", wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# --- public API ----------------------------------------------------------------------
def render_report(t, x, lead_names, fs, analysis=None, out_path="report.png", fmt="png",
                  meta=None, title="KardiaMobile 6L — Research ECG", window_s=10.0,
                  full_disclosure=True):
    """Render the ECG report. PNG: the one-page summary. PDF: the summary page followed by
    the WHOLE recording, 10 s per A4-landscape page at true paper scale (25 mm/s, 10 mm/mV,
    1 mV calibration pulse) — `full_disclosure=False` gives the single summary page.
    Returns the written path."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[0] < x.shape[1] and x.shape[0] <= 6:
        # accept (leads, samples) too
        x = x.T
    lead_names = list(lead_names)
    if x.shape[1] != len(lead_names):
        lead_names = list(lead_names[: x.shape[1]]) or [f"ch{i}" for i in range(x.shape[1])]
    if t is None:
        t = np.arange(x.shape[0]) / fs

    fmt = fmt.lower()
    out_path = str(Path(out_path).with_suffix("." + fmt))
    meta = dict(meta or {})
    meta.setdefault("duration_s", x.shape[0] / fs)
    mains = meta.get("mains", 60.0)

    duration = x.shape[0] / fs
    n_groups = int(math.ceil(duration / PAGE_SECONDS)) if (full_disclosure and fmt == "pdf") else 0
    xf = None
    if n_groups:
        # filter once; decide the common lane scale and how many 10-s rows fit per page
        xf = np.column_stack([_clean_stable(x[:, i], fs, mains) for i in range(x.shape[1])])
        _c, half = _lane_scale(xf, fs, lead_names, 0.0, duration)
        per_page, mm_per_mv = _page_layout(len(lead_names), half, A4_L_MM[1] - 20.0 - 12.0)
        groups = [(k * PAGE_SECONDS, min(duration, (k + 1) * PAGE_SECONDS)) for k in range(n_groups)]
        pages = [groups[i:i + per_page] for i in range(0, len(groups), per_page)]
    else:
        pages = []
    all_findings = _all_findings(analysis)
    est = sum(1 + (1 if (len(f["evidence"]) + len(f["when"])) > 92 else 0) for f in all_findings)
    findings_page = bool(all_findings) and (len(all_findings) > 3 or est > 4)
    n_pages = 1 + (1 if findings_page else 0) + len(pages)

    def summary_page():
        fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)  # A4 portrait
        gs = GridSpec(5, 1, figure=fig, height_ratios=[0.9, 1.6, 4.3, 2.3, 0.45],
                      hspace=0.16, left=0.045, right=0.965, top=0.972, bottom=0.022)
        _draw_header(fig, gs[0], title, meta, analysis)
        _draw_measurements(fig, gs[1], analysis)
        _draw_strips(fig, gs[2], t, x, lead_names, fs, mains, window_s)
        mid = gs[3].subgridspec(1, 2, width_ratios=[1.15, 1.45], wspace=0.10)
        _draw_median_beat(fig, mid[0], analysis)
        _draw_quality(fig, mid[1], analysis)
        _draw_footer(fig, gs[4], analysis, meta)
        if n_pages > 1:
            fig.text(0.965, 0.006, f"page 1 / {n_pages}", fontsize=7, color=MUTED, ha="right", va="bottom")
        return fig

    if fmt != "pdf" or not pages:
        fig = summary_page()
        fig.savefig(out_path, dpi=150 if fmt == "png" else 300, facecolor=BG)
        plt.close(fig)
        return out_path

    # ---- multi-page PDF: summary page + the whole recording, 10 s per page at true scale
    from matplotlib.backends.backend_pdf import PdfPages
    labels_at = _beat_labels_for_report(analysis, x, fs, lead_names, mains)
    episodes = _episodes_for_report(analysis)
    with PdfPages(out_path) as pdf:
        fig = summary_page()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)
        first_strip_page = 2
        if findings_page:
            fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
            _draw_findings_page(fig, analysis, meta, title, 2, n_pages)
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
            first_strip_page = 3
        for k, page_groups in enumerate(pages):
            fig = plt.figure(figsize=A4_L, facecolor=BG)
            _draw_full_page(fig, xf, lead_names, fs, page_groups, k + first_strip_page, n_pages, meta, title,
                            labels_at=labels_at, half=half, mm_per_mv=mm_per_mv, episodes=episodes)
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = title
        info["Subject"] = "Research ECG — not a medical device"
    return out_path


def _beat_labels_for_report(analysis, x, fs, lead_names, mains):
    """[(t_seconds, 'V'|'S'), ...] for ectopic beats, so the full-disclosure pages can mark
    them. Uses the arrhythmia screen when the analysis carries beat labels; R-peaks are
    re-detected on the primary lead the same way analysis does. Empty on any failure."""
    try:
        labels = _get(analysis, "arrhythmia.beat_labels")
        if labels is None:
            return []
        labels = [str(v) for v in np.asarray(labels).tolist()]
        from . import dsp
        names = list(lead_names)
        li = names.index("II") if "II" in names else (names.index("I") if "I" in names else 0)
        peaks = dsp.detect_r_peaks(dsp.clean(x[:, li], fs, mains), fs)
        if len(peaks) != len(labels):
            return []
        return [(float(p / fs), lab) for p, lab in zip(peaks, labels) if lab in ("V", "S")]
    except Exception:  # noqa: BLE001
        return []


def render_report_from_csv(csv_path, analysis=None, out_path=None, fmt="png", meta=None):
    """Load a open_kardia CSV recording and render a report."""
    from .sinks.csv_sink import read_csv

    fs, names, t, x = read_csv(csv_path)
    if out_path is None:
        out_path = str(Path(csv_path).with_suffix("")) + "_report"
    meta = dict(meta or {})
    meta.setdefault("recorded_at", None)
    return render_report(t, x, names, fs, analysis=analysis, out_path=out_path, fmt=fmt, meta=meta)
