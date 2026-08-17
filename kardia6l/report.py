"""Printable ECG report renderer (PNG / PDF) for kardia6l.

Pure matplotlib (Agg) — no GUI, no BLE. Renders a single-page clinical-style report
from a recording plus an optional analysis result (see kardia6l.analysis). The analysis
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
    ax.text(0.985, 0.74, str(rhythm), transform=ax.transAxes, fontsize=12.5,
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
    ax.text(0.02, 0.93, "Measurements", transform=ax.transAxes, fontsize=10,
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
        from kardia6l.analysis import range_text as _rng
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
    y_top = 0.80
    y_h = y_top / nrow
    for i, (label, val, hint) in enumerate(cells):
        r, c = divmod(i, ncol)
        cx = c * x_w + 0.02
        cy = y_top - r * y_h
        ax.text(cx, cy, label, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
        ax.text(cx, cy - 0.30 * y_h, val, transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", color=INK, va="top")
        if hint:
            ax.text(cx, cy - 0.74 * y_h, hint, transform=ax.transAxes, fontsize=6.2,
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
    axes[0].set_title(f"ECG strips — first {win / fs:.0f} s", fontsize=9, color=INK,
                      loc="left", pad=4)
    return axes


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
    y = 0.66
    for label, val in rows:
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
        ax.text(0.95, y, val, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                color=INK, va="top", ha="right")
        y -= 0.20
    flags = _get(analysis, "rhythm.flags")
    if flags:
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
                  meta=None, title="KardiaMobile 6L — Research ECG", window_s=10.0):
    """Render a one-page ECG report to PNG or PDF. Returns the written path."""
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

    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)  # A4 portrait
    gs = GridSpec(5, 1, figure=fig, height_ratios=[0.9, 1.15, 5.4, 1.4, 0.55],
                  hspace=0.16, left=0.045, right=0.965, top=0.972, bottom=0.022)
    _draw_header(fig, gs[0], title, meta, analysis)
    _draw_measurements(fig, gs[1], analysis)
    _draw_strips(fig, gs[2], t, x, lead_names, fs, mains, window_s)
    mid = gs[3].subgridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.10)
    _draw_median_beat(fig, mid[0], analysis)
    _draw_quality(fig, mid[1], analysis)
    _draw_footer(fig, gs[4], analysis, meta)

    dpi = 150 if fmt == "png" else 300
    fig.savefig(out_path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return out_path


def render_report_from_csv(csv_path, analysis=None, out_path=None, fmt="png", meta=None):
    """Load a kardia6l CSV recording and render a report."""
    from .sinks.csv_sink import read_csv

    fs, names, t, x = read_csv(csv_path)
    if out_path is None:
        out_path = str(Path(csv_path).with_suffix("")) + "_report"
    meta = dict(meta or {})
    meta.setdefault("recorded_at", None)
    return render_report(t, x, names, fs, analysis=analysis, out_path=out_path, fmt=fmt, meta=meta)
