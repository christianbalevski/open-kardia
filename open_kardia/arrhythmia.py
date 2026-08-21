"""Beat-level arrhythmia screening (ectopy, patterns, pauses, interval and rate flags).

RESEARCH SIGNAL — NOT A MEDICAL DEVICE. Nothing here is clinically validated or
suitable for diagnosis. Every finding is a heuristic on an automated single-lead
R-peak train and is phrased "possible …" on purpose. Atrial fibrillation is NOT
assessed here (see `open_kardia.af`); this module deliberately never claims AF.

Pure numpy/scipy. `detect_arrhythmias` never raises: on garbage input it returns an
empty `ArrhythmiaReport` with `warnings`.

Method summary
  1. Per-beat features from the R-peak train and the cleaned primary lead:
       * RR before/after, a prevailing "normal" RR reference (median of the last 8
         normal-to-normal RRs, tracked sequentially and frozen while ectopic beats go
         by; a centred ±8-beat median RR is the fallback), prematurity ratio
         (RR_pre / reference), compensatory-pause ratio ((RR_pre+RR_post)/(2·reference))
       * QRS width: width of the slope-energy envelope around R. We take the
         smoothed (~16 ms) RMS of the first derivative and measure the contiguous
         span (gaps ≤ 20 ms bridged) where it exceeds 8 % of its peak within
         ±60 ms of R, minus the smoothing width, times an empirical 1.3 to
         approximate QRS ms. Consistent across beats, so it is used mostly as a
         RATIO to the reference (normal-beat) width, plus a loose absolute threshold.
       * morphology correlation with a reference (normal) template on R ± 120 ms,
         maximised over ±20 ms alignment lags; R amplitude ratio; polarity.
     The template is first the supplied/computed median beat, then re-built from
     non-premature, well-correlated beats so bigeminy does not poison it.
  2. Soft scoring: premature × abnormal-morphology  → 'V'; abruptly premature ×
     normal morphology → 'S' (suppressed when the surrounding N-N rhythm is AF-like
     irregular); grossly abnormal wide morphology → 'V' even when not premature
     (late/interpolated PVCs, long runs); tiny-RR / low-amplitude detections → '?'.
  3. Rules over the label sequence: burden, couplets, runs, bigeminy/trigeminy,
     pauses, interval flags (PR/QRS/QTc from `analysis.Intervals`), rate screens on
     normal-to-normal RRs (brady/tachy/SVT/VT/irregular).
"""
from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field

import numpy as np

from . import dsp

NAN = float("nan")

# ------------------------------------------------------------------ thresholds (one place)
PREMATURE_RATIO = 0.80        # RR_pre / local median below this = clearly premature
PREMATURE_SOFT = 0.90         # ... above this = clearly not premature (linear in between)
CORR_ABNORMAL = 0.85          # template correlation below this = abnormal morphology
WIDTH_RATIO_ABNORMAL = 1.30   # per-beat QRS width / reference width
WIDTH_ABS_MS = 120.0          # absolute per-beat width above this = wide
WIDTH_SCALE = 1.3             # slope-envelope span -> approx QRS ms (empirical, MIT-BIH)
ARTIFACT_RR_RATIO = 0.30      # RR_pre below this fraction of local median = detector doublet
PAUSE_ABS_S = 2.0
PAUSE_RATIO = 1.8
FREQUENT_PVC_BURDEN = 0.10    # or ≥ 6 per minute (with ≥ 3 PVCs)
FREQUENT_PVC_PER_MIN = 6.0
SVT_HR = 150.0
VT_HR = 120.0
SEVERITY_RANK = {"urgent": 0, "notable": 1, "info": 2}


# ------------------------------------------------------------------ dataclasses
@dataclass
class Finding:
    code: str            # machine code, e.g. 'pvc', 'bigeminy', 'pause', 'long_qtc'
    label: str           # human label
    severity: str        # 'info' | 'notable' | 'urgent'
    confidence: float    # 0..1
    evidence: str        # one-line rationale with numbers
    count: int = 0
    beats: list = field(default_factory=list)   # indices into rpeaks involved
    times: list = field(default_factory=list)   # seconds of the involved beats (recording time base)
    episodes: list = field(default_factory=list)  # [(t_start, t_end), ...] seconds; [] = whole recording

    def when(self, max_items: int = 6) -> str:
        return when_text(self, max_items)


@dataclass
class ArrhythmiaReport:
    beat_labels: np.ndarray        # per R-peak: 'N' | 'S' | 'V' | '?'
    findings: list                 # list[Finding], most severe first
    metrics: dict                  # JSON-safe floats
    summary: str                   # '' if nothing found
    warnings: list = field(default_factory=list)


# how far apart (in beats) two involved beats may be and still belong to one episode
_EPISODE_GAP = {"bigeminy": 2, "trigeminy": 3, "couplet": 1, "vt_run": 1, "svt_run": 1, "pause": 1}


def _fmt_t(t: float) -> str:
    """Seconds -> '12.4 s' or '1:02.4' when over a minute."""
    if t >= 60:
        m = int(t // 60)
        return f"{m}:{t - 60 * m:04.1f}"
    return f"{t:.1f} s"


def when_text(f, max_items: int = 6) -> str:
    """Human-readable 'when' for a finding: episode spans, single-beat times, or
    'whole recording' for interval/rate flags."""
    eps = list(getattr(f, "episodes", []) or [])
    times = list(getattr(f, "times", []) or [])
    if eps and any((b - a) > 0.05 for a, b in eps):
        parts = [f"{_fmt_t(a)}–{_fmt_t(b)}" for a, b in eps[:max_items]]
        more = len(eps) - len(parts)
        return "episodes " + ", ".join(parts) + (f" (+{more} more)" if more > 0 else "")
    if times:
        parts = [_fmt_t(t) for t in times[:max_items]]
        more = len(times) - len(parts)
        return "at " + ", ".join(parts) + (f" (+{more} more)" if more > 0 else "")
    return "whole recording"


def annotate_times(findings: list, rpeaks: np.ndarray, fs: float, t0: float = 0.0) -> None:
    """Fill `times` / `episodes` on each finding from its beat indices (in place).
    Ectopic-beat findings (pvc/pac) get one zero-length episode per beat; pattern findings
    (bigeminy, trigeminy, couplets, runs) get spans over consecutive involved beats; a pause
    spans the long RR itself (previous beat -> the beat that ends the pause)."""
    rp = np.asarray(rpeaks, dtype=float)
    n = len(rp)
    for f in findings:
        idx = sorted({int(i) for i in (f.beats or []) if 0 <= int(i) < n})
        if not idx:
            f.times, f.episodes = [], []
            continue
        f.times = [float(round(t0 + rp[i] / fs, 3)) for i in idx]
        if f.code == "pause":
            f.episodes = [(float(round(t0 + rp[i - 1] / fs, 3)) if i > 0 else float(round(t0 + rp[i] / fs - 2.0, 3)),
                           float(round(t0 + rp[i] / fs, 3))) for i in idx]
            continue
        gap = _EPISODE_GAP.get(f.code)
        if gap is None:                       # pvc / pac / possible_missed_beats: individual beats
            f.episodes = [(t, t) for t in f.times]
            continue
        eps, start, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - prev > gap:
                eps.append((start, prev)); start = i
            prev = i
        eps.append((start, prev))
        # pad each span by half an RR on both sides so it visibly covers the beats
        f.episodes = []
        for a, b in eps:
            ta = rp[a] / fs - (0.5 * (rp[a] - rp[a - 1]) / fs if a > 0 else 0.3)
            tb = rp[b] / fs + (0.5 * (rp[b + 1] - rp[b]) / fs if b + 1 < n else 0.3)
            f.episodes.append((float(round(t0 + max(0.0, ta), 3)), float(round(t0 + tb, 3))))


def _empty(n: int, warns: list) -> ArrhythmiaReport:
    return ArrhythmiaReport(np.array(["?"] * n, dtype="<U1"), [], {"n_beats": float(n)}, "", warns)


# ------------------------------------------------------------------ feature helpers
def _local_median(v: np.ndarray, half: int = 8, mask: np.ndarray | None = None) -> np.ndarray:
    """Rolling median of `v` over ±half entries. If `mask` is given only masked-in
    entries contribute (falling back to all entries when fewer than 3 are available)."""
    n = len(v)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = v[lo:hi]
        if mask is not None:
            m = mask[lo:hi]
            if np.sum(m) >= 3:
                seg = seg[m]
        seg = seg[np.isfinite(seg)]
        out[i] = np.median(seg) if len(seg) else NAN
    return out


def _qrs_widths(x: np.ndarray, rpeaks: np.ndarray, fs: float) -> np.ndarray:
    """Per-beat approximate QRS width (ms) from the slope-energy envelope (module docstring)."""
    n = len(x)
    d = np.gradient(x) * fs
    sm = max(3, int(round(0.016 * fs)) | 1)              # ~16 ms, odd
    env = np.sqrt(np.convolve(d * d, np.ones(sm) / sm, mode="same"))
    w_out = int(0.15 * fs)                                # search ±150 ms
    w_core = int(0.06 * fs)                               # peak slope within ±60 ms
    gap = int(round(0.02 * fs))                           # bridge gaps ≤ 20 ms
    widths = np.full(len(rpeaks), NAN)
    for k, r in enumerate(rpeaks):
        lo, hi = max(0, r - w_out), min(n, r + w_out + 1)
        if hi - lo < 3:
            continue
        seg = env[lo:hi]
        c0, c1 = max(0, r - w_core - lo), min(len(seg), r + w_core - lo + 1)
        if c1 <= c0:
            continue
        pk_i = c0 + int(np.argmax(seg[c0:c1]))
        thr = 0.08 * seg[pk_i]
        if thr <= 0:
            continue
        above = seg > thr
        # extend left / right from the peak, bridging short gaps
        i = pk_i
        miss = 0
        while i > 0 and miss <= gap:
            i -= 1
            miss = 0 if above[i] else miss + 1
        left = i + miss
        j = pk_i
        miss = 0
        while j < len(seg) - 1 and miss <= gap:
            j += 1
            miss = 0 if above[j] else miss + 1
        right = j - miss
        widths[k] = max(0.0, (right - left + 1 - sm) / fs * 1000.0)
    # The envelope span under-reads the true QRS duration by ~25 % on MIT-BIH (normal beats
    # measure ~65–75 ms raw vs ~90 ms by delineation); scale to an approximate QRS ms.
    return WIDTH_SCALE * widths


def _segments(x: np.ndarray, rpeaks: np.ndarray, w: int, lag: int = 0) -> np.ndarray:
    """(n_beats, 2w+1) matrix of R±w windows shifted by `lag`; out-of-range → edge-padded."""
    n = len(x)
    idx = rpeaks[:, None] + lag + np.arange(-w, w + 1)[None, :]
    idx = np.clip(idx, 0, n - 1)
    return x[idx]


def _template_corr(x: np.ndarray, rpeaks: np.ndarray, tmpl: np.ndarray, w: int, max_lag: int):
    """Max (over ±max_lag) zero-lag Pearson correlation of each beat with `tmpl` (len 2w+1),
    plus the per-beat R amplitude (peak-to-peak of the best-aligned window)."""
    t = tmpl - tmpl.mean()
    tn = np.sqrt(np.sum(t * t)) + 1e-12
    best = np.full(len(rpeaks), -1.0)
    for lag in range(-max_lag, max_lag + 1):
        s = _segments(x, rpeaks, w, lag)
        s = s - s.mean(axis=1, keepdims=True)
        sn = np.sqrt(np.sum(s * s, axis=1)) + 1e-12
        c = (s @ t) / (sn * tn)
        best = np.maximum(best, c)
    seg0 = _segments(x, rpeaks, w, 0)
    amp = np.ptp(seg0, axis=1)
    return best, amp


def _build_template(x: np.ndarray, rpeaks: np.ndarray, sel: np.ndarray, w: int):
    if np.sum(sel) < 3:
        return None
    return np.median(_segments(x, rpeaks[sel], w, 0), axis=0)


# ------------------------------------------------------------------ pattern helpers
def _runs(labels: np.ndarray, sym: str) -> list[tuple[int, int]]:
    """(start, length) of maximal runs of `sym`."""
    out = []
    i, n = 0, len(labels)
    while i < n:
        if labels[i] == sym:
            j = i
            while j < n and labels[j] == sym:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def _nplets(labels: np.ndarray, sym: str, period: int) -> tuple[int, list[int]]:
    """Longest chain of ectopics separated by exactly (period-1) N beats — bigeminy has
    period 2 (N V N V), trigeminy period 3 (N N V N N V). Returns (max_cycles, beats)."""
    n = len(labels)
    best, best_beats = 0, []
    i = 0
    while i < n:
        if labels[i] == sym:
            cyc, beats, j = 1, [i], i
            while (j + period < n and labels[j + period] == sym
                   and all(labels[j + k] == "N" for k in range(1, period))):
                j += period
                cyc += 1
                beats.append(j)
            if cyc > best:
                best, best_beats = cyc, beats
            i = j + 1
        else:
            i += 1
    return best, best_beats


def _sev_sort(findings: list) -> list:
    return sorted(findings, key=lambda f: (SEVERITY_RANK.get(f.severity, 3), -f.confidence))


def _f(v, d=0):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def _pct(frac):
    v = 100.0 * frac
    return f"{v:.1f}%" if v < 1.0 else f"{v:.0f}%"


# ------------------------------------------------------------------ main
def detect_arrhythmias(clean_primary: np.ndarray, fs: float, rpeaks: np.ndarray,
                       median_beat: np.ndarray | None = None, r_idx: int = 0,
                       intervals=None, p_consistent: bool = False,
                       sqi: float = 1.0) -> ArrhythmiaReport:
    """Screen a cleaned single-lead ECG + R-peak train for beat-level arrhythmias.

    Never raises. Returns an `ArrhythmiaReport` (labels 'N'/'S'/'V'/'?', findings sorted
    most-severe first, JSON-safe metrics, one-line summary, warnings). Research use only.
    """
    warns: list[str] = []
    try:
        x = np.asarray(clean_primary, dtype=np.float64).ravel()
        rp = np.asarray(rpeaks).ravel()
        rp = rp[np.isfinite(rp.astype(float))].astype(int) if len(rp) else np.array([], dtype=int)
        fs = float(fs)
    except Exception:  # noqa: BLE001
        return _empty(0, ["unusable input"])
    nb = len(rp)
    if not np.isfinite(fs) or fs <= 0 or x.ndim != 1 or len(x) < int(2 * fs) or not np.all(np.isfinite(x)):
        return _empty(nb, ["signal unusable (too short, non-finite, or bad fs)"])
    rp = np.unique(rp[(rp >= 0) & (rp < len(x))])
    nb = len(rp)
    if nb < 4:
        return _empty(nb, [f"only {nb} beats; arrhythmia screening needs ≥ 4"])
    if not np.isfinite(sqi):
        sqi = 0.5
    sqi = float(np.clip(sqi, 0.0, 1.0))
    if sqi < 0.5:
        warns.append(f"low signal quality (SQI {sqi:.2f}); findings unreliable")
    duration_min = len(x) / fs / 60.0

    try:
        with _warnings.catch_warnings(), np.errstate(all="ignore"):
            _warnings.simplefilter("ignore", RuntimeWarning)
            rep = _detect(x, fs, rp, median_beat, r_idx, intervals, p_consistent, sqi,
                          duration_min, warns)
        try:
            annotate_times(rep.findings, rp, fs)
        except Exception:  # noqa: BLE001 — timing annotation is best-effort
            pass
        return rep
    except Exception as e:  # noqa: BLE001  — never raise
        return _empty(nb, warns + [f"arrhythmia screening failed: {type(e).__name__}: {e}"])


def _detect(x, fs, rp, median_beat, r_idx, intervals, p_consistent, sqi, duration_min, warns):
    nb = len(rp)
    # ---------------------------------------------------------------- RR features
    rr = np.diff(rp) / fs                                    # seconds, len nb-1
    rr_pre = np.concatenate([[NAN], rr])
    rr_post = np.concatenate([rr, [NAN]])
    loc_med = _local_median(rr_pre, 8)
    prem = rr_pre / loc_med                                  # prematurity ratio
    comp = (rr_pre + rr_post) / (2.0 * loc_med)              # ≈1 compensatory, <1 not

    # ---------------------------------------------------------------- morphology
    w = int(round(0.12 * fs))
    max_lag = max(1, int(round(0.02 * fs)))
    tmpl = None
    if median_beat is not None:
        mb = np.asarray(median_beat, dtype=float).ravel()
        if len(mb) > 2 * w + 1 and 0 <= r_idx - w and r_idx + w + 1 <= len(mb):
            tmpl = mb[r_idx - w:r_idx + w + 1]
    if tmpl is None:
        try:
            from .analysis import _median_beat
            mb, ri = _median_beat(x, rp, fs)
        except Exception:  # noqa: BLE001
            mb, ri = None, 0
        if mb is not None and ri - w >= 0 and ri + w + 1 <= len(mb):
            tmpl = mb[ri - w:ri + w + 1]
    if tmpl is None:
        tmpl = np.median(_segments(x, rp, w, 0), axis=0)
    corr, amp = _template_corr(x, rp, tmpl, w, max_lag)
    widths = _qrs_widths(x, rp, fs)

    # Reference (normal) set → rebuild the template and the reference width from it.
    # When ectopy dominates a strip (bigeminy, VVN patterns) the plain median beat IS
    # the ectopic morphology, so we seed from the beats that follow a long RR (the beat
    # after a compensatory pause is a normal one) when there are enough of them, else
    # from non-premature, well-correlated beats.
    ok_prem = np.isfinite(prem) & (prem > 0.9)
    post_pause = np.isfinite(prem) & (prem >= 1.2)
    if np.sum(post_pause) >= 3:                       # keep the narrower 75 % of them
        post_pause &= widths <= np.nanpercentile(widths[post_pause], 75)
    if np.sum(post_pause) >= 3:
        seed = post_pause
    elif np.any(ok_prem):
        seed = ok_prem & (corr >= np.nanmedian(corr[ok_prem]))
    else:
        seed = corr >= np.nanmedian(corr)
    t2 = _build_template(x, rp, seed, w)
    if t2 is not None:
        tmpl = t2
        corr, amp = _template_corr(x, rp, tmpl, w, max_lag)
    ref = ok_prem & (corr >= 0.9)
    if np.sum(ref) < 3:
        ref = seed if np.sum(seed) >= 3 else np.ones(nb, dtype=bool)
    ref_width = float(np.nanmedian(widths[ref])) if np.any(ref & np.isfinite(widths)) else float(np.nanmedian(widths))
    ref_amp = float(np.nanmedian(amp[ref])) if np.any(ref) else float(np.nanmedian(amp))
    if not np.isfinite(ref_width) or ref_width <= 0:
        ref_width = 90.0
        warns.append("QRS width estimate unavailable; width criterion disabled")
    amp_ratio = amp / (ref_amp + 1e-9)
    width_ratio = widths / ref_width
    # polarity of the beat around R relative to the template polarity
    seg0 = _segments(x, rp, int(round(0.04 * fs)), 0)
    pol = np.sign(seg0[:, seg0.shape[1] // 2] - np.median(seg0, axis=1))
    pol_t = np.sign(tmpl[w] - np.median(tmpl))
    pol_flip = (pol != 0) & (pol_t != 0) & (pol != pol_t)

    # ---------------------------------------------------------------- soft scores
    # Morphology-only scores are vectorised; the RR reference is tracked sequentially:
    # `ref_rr` = median of the last 8 normal-to-normal RRs, frozen while ectopic beats go
    # by, so every beat of a run stays "premature" relative to the rhythm before it.
    m_ab = np.clip((CORR_ABNORMAL + 0.07 - corr) / 0.14, 0, 1)            # 0.85 → 0.5, ≤0.78 → 1
    w_ab = np.clip((width_ratio - 1.15) / (2 * (WIDTH_RATIO_ABNORMAL - 1.15)), 0, 1)  # 1.30 → 0.5
    # absolute criterion (≈120 ms) — but never below the record's own baseline width, so a
    # bundle-branch-block baseline does not turn every premature beat into a "PVC"
    w_floor = max(WIDTH_ABS_MS - 10.0, 1.1 * ref_width)
    w_ab = np.fmax(w_ab, np.clip((widths - w_floor) / 40.0, 0, 1))        # floor+20 ms → 0.5
    w_ab = np.where(np.isfinite(w_ab), w_ab, 0.0)
    a_ab = np.clip((np.abs(np.log(np.fmax(amp_ratio, 1e-6))) - 0.4) / 0.6, 0, 1)   # 1.5x/0.67x → 0
    vent = 1.0 - (1.0 - m_ab) * (1.0 - w_ab) * (1.0 - 0.5 * a_ab)
    vent = np.where(pol_flip, np.fmax(vent, 0.7), vent)
    # Beats inside the filter edge transient (first/last 0.5 s) and the very first beat
    # (no RR_pre) get inflated width/amplitude estimates: there, only a clearly abnormal
    # morphology (corr ≤ CORR_ABNORMAL) or a polarity flip may count as ventricular.
    edge = (rp < 0.5 * fs) | (rp > len(x) - 0.5 * fs)
    edge[0] = True
    vent = np.where(edge & (corr > CORR_ABNORMAL) & ~pol_flip, 0.0, vent)
    w_ab = np.where(edge & (corr > CORR_ABNORMAL) & ~pol_flip, 0.0, w_ab)
    # Hard edge: a beat within 0.35 s of either end has its morphology window clipped and
    # sits in the onset/offset transient of a gated recording — its correlation is
    # meaningless (it can even be negative), so it can never be called ventricular.
    hard_edge = (rp < 0.35 * fs) | (rp > len(x) - 0.35 * fs)
    vent = np.where(hard_edge, 0.0, vent)
    m_ab = np.where(hard_edge, 0.0, m_ab)
    w_ab = np.where(hard_edge, 0.0, w_ab)
    # Width alone never carries a V decision (the envelope width is the least reliable
    # feature): a ventricular label needs corr ≤ 0.90, or a polarity flip, or an amplitude
    # ratio outside 0.5–1.8, or an absolute width ≥ 130 ms together with corr ≤ 0.95.
    v_ok = (corr <= 0.90) | pol_flip | (amp_ratio < 0.5) | (amp_ratio > 1.8) | \
           ((widths >= 130.0) & (corr <= 0.95))
    vent = np.where(v_ok, vent, np.fmin(vent, 0.3))

    labels = np.array(["N"] * nb, dtype="<U1")
    conf = np.zeros(nb)
    ref_arr = np.full(nb, NAN)          # prevailing normal RR used for each beat
    prem = np.full(nb, NAN)
    comp = np.full(nb, NAN)
    nn_hist: list[float] = []            # recent N-N RRs (s)
    init = rr[:8][np.isfinite(rr[:8])]
    ref0 = float(np.median(init)) if len(init) else NAN
    ref_rr = ref0
    since_update = 0                      # beats since the last N-N reference update
    pending_long = NAN                    # last isolated ~2x RR (see history update)

    def _ref_from_hist():
        # median of the last 8 N-N RRs, padded with the initial estimate while the
        # history is short so a single interval cannot swing the reference
        h = nn_hist[-8:]
        return float(np.median(h + [ref0] * (8 - len(h))))

    for i in range(nb):
        ref_arr[i] = ref_rr
        if i == 0 or not np.isfinite(ref_rr) or ref_rr <= 0:
            # no RR_pre: morphology only (a strip may well start on an ectopic beat)
            labels[i] = "V" if vent[i] >= 0.7 else "N"
            conf[i] = 0.5 * m_ab[i] if labels[i] == "V" else 0.0
            continue
        prem[i] = rr_pre[i] / ref_rr
        comp[i] = (rr_pre[i] + rr_post[i]) / (2.0 * ref_rr) if np.isfinite(rr_post[i]) else NAN
        p_prem = float(np.clip((PREMATURE_SOFT - prem[i]) / (PREMATURE_SOFT - PREMATURE_RATIO), 0, 1))
        # detector doublets: tiny RR and/or tiny amplitude relative to the reference
        art = (prem[i] < ARTIFACT_RR_RATIO) or (prem[i] < 0.45 and amp_ratio[i] < 0.35) or amp_ratio[i] < 0.2
        # compensatory pause nudge: full pause supports V, clearly incomplete supports S
        cb = float(np.clip((comp[i] - 0.92) / 0.10, -0.15, 0.15)) if np.isfinite(comp[i]) else 0.0
        vscore = float(np.clip(vent[i] + cb, 0, 1))
        # local N-N irregularity (AF-like): premature narrow beats are then not called PACs
        if len(nn_hist) >= 4:
            h = np.array(nn_hist[-8:])
            nn_cv = 1.4826 * float(np.median(np.abs(h - np.median(h)))) / float(np.median(h))
        else:
            nn_cv = 0.0
        # a PAC is ABRUPTLY premature (vs the previous RR) unless it continues a run;
        # a gradual sinus acceleration is not ectopy
        # (a slow "run" of near-normal RRs is a rate change, not ectopy: continuation needs
        # a tachycardic rate or clear prematurity)
        # (the beat after a ~2x gap — a missed detection or dropped beat — is not "abrupt")
        abrupt = (i >= 2 and rr_pre[i] < 0.9 * rr_pre[i - 1] and rr_pre[i - 1] < 1.7 * ref_rr) or \
                 (labels[i - 1] == "S" and (rr_pre[i] < 0.6 or prem[i] < 0.75))
        if art:
            labels[i] = "?"
        elif p_prem >= 0.5 and vscore >= 0.5:
            labels[i] = "V"; conf[i] = min(p_prem, vscore)
        elif p_prem >= 0.5 and nn_cv <= 0.12 and abrupt:
            labels[i] = "S"; conf[i] = min(p_prem, 1 - vscore)
        elif p_prem < 0.5 and m_ab[i] >= 0.9 and (w_ab[i] >= 0.5 or pol_flip[i]) and prem[i] < 1.6:
            # not premature but grossly abnormal AND wide/flipped: late / interpolated PVC
            labels[i] = "V"; conf[i] = 0.5 * m_ab[i]
        else:
            labels[i] = "N"
        # N-N history update. An isolated long RR (1.7–2.5x: missed R-peak or dropped beat)
        # must NOT enter the history; two consecutive long RRs of the same length are a
        # genuine step in rate (e.g. 2:1 block) and are accepted together.
        if labels[i] == "N" and labels[i - 1] == "N" and np.isfinite(ref_rr):
            ratio = rr_pre[i] / ref_rr
            if 0.5 < ratio < 1.7:
                nn_hist.append(float(rr_pre[i]))
                ref_rr = _ref_from_hist()
                since_update = 0
                pending_long = NAN
            elif 1.7 <= ratio < 2.5:
                if np.isfinite(pending_long) and abs(rr_pre[i] / pending_long - 1.0) < 0.1:
                    nn_hist.extend([float(pending_long), float(rr_pre[i])])
                    ref_rr = _ref_from_hist()
                    since_update = 0
                    pending_long = NAN
                else:
                    pending_long = float(rr_pre[i])
        since_update += 1
        if since_update >= 30:
            # 30 beats without a normal-to-normal interval: the base rhythm has changed
            # (or is a sustained tachycardia, which the rate screen handles) — re-anchor
            # the reference on the recent RRs so everything is not "premature" forever.
            nn_hist = [float(v) for v in rr_pre[i - 7:i + 1] if np.isfinite(v)]
            ref0 = ref_rr = float(np.median(nn_hist)) if nn_hist else ref_rr
            since_update = 0
    # a doublet's partner (the beat right after a '?') is not premature either
    for i in np.where(labels == "?")[0]:
        if i + 1 < nb and labels[i + 1] in ("S", "V") and rr_post[i] < 0.5 * ref_arr[i]:
            labels[i + 1] = "?"
    loc2 = np.where(np.isfinite(ref_arr), ref_arr, loc_med)

    # ---------------------------------------------------------------- findings
    findings: list[Finding] = []
    n_v = int(np.sum(labels == "V"))
    n_s = int(np.sum(labels == "S"))
    n_q = int(np.sum(labels == "?"))
    n_valid = max(1, nb - n_q)
    pvc_burden = n_v / n_valid
    pac_burden = n_s / n_valid
    v_runs = _runs(labels, "V")
    s_runs = _runs(labels, "S")
    couplets = [s for s, L in v_runs if L == 2]
    v_long = [(s, L) for s, L in v_runs if L >= 3]
    max_run = max([L for _, L in v_runs], default=0)
    big_v, big_v_beats = _nplets(labels, "V", 2)
    tri_v, tri_v_beats = _nplets(labels, "V", 3)
    big_s, big_s_beats = _nplets(labels, "S", 2)
    tri_s, tri_s_beats = _nplets(labels, "S", 3)
    q_scale = 0.5 + 0.5 * sqi          # confidence shrinks with poor signal quality

    def _rate_of_run(s, L):
        seg = rr[s:s + L - 1] if L > 1 else np.array([])
        seg = seg[np.isfinite(seg) & (seg > 0)]
        return float(60.0 / np.mean(seg)) if len(seg) else NAN

    # a single low-confidence ectopic beat is not worth a finding: count it in metrics only
    uncertain_v = int(n_v == 1 and float(np.mean(conf[labels == "V"])) * q_scale < 0.2)
    uncertain_s = int(n_s == 1 and float(np.mean(conf[labels == "S"])) * q_scale < 0.2)
    if n_v and not uncertain_v:
        pvc_per_min = n_v / duration_min if duration_min > 0 else NAN
        frequent = pvc_burden >= FREQUENT_PVC_BURDEN or (n_v >= 3 and pvc_per_min >= FREQUENT_PVC_PER_MIN)
        wide = np.nanmedian(width_ratio[labels == "V"]) if n_v else NAN
        ev = (f"{n_v} PVC-like beats ({_pct(pvc_burden)} of beats), median morphology corr "
              f"{_f(np.nanmedian(corr[labels=='V']),2)}, QRS width {_f(100*wide)}% of normal")
        if couplets:
            ev += f", {len(couplets)} couplet(s)"
        findings.append(Finding("pvc",
                                "Premature ventricular complexes" + (" (frequent)" if frequent else ""),
                                "notable" if frequent else "info",
                                float(np.clip(np.mean(conf[labels == "V"]) * q_scale, 0.05, 0.95)),
                                ev, n_v, [int(i) for i in np.where(labels == "V")[0]]))
    if n_s and not uncertain_s:
        pac_per_min = n_s / duration_min if duration_min > 0 else NAN
        frequent = pac_burden >= FREQUENT_PVC_BURDEN or (n_s >= 3 and pac_per_min >= FREQUENT_PVC_PER_MIN)
        ev = (f"{n_s} premature narrow beats ({_pct(pac_burden)} of beats), morphology corr "
              f"{_f(np.nanmedian(corr[labels=='S']),2)}, mostly non-compensatory pause")
        findings.append(Finding("pac",
                                "Premature atrial/supraventricular complexes" + (" (frequent)" if frequent else ""),
                                "notable" if frequent else "info",
                                float(np.clip(np.mean(conf[labels == "S"]) * q_scale, 0.05, 0.9)),
                                ev, n_s, [int(i) for i in np.where(labels == "S")[0]]))
    if couplets:
        findings.append(Finding("couplet", "Ventricular couplets", "notable", float(0.6 * q_scale),
                                f"{len(couplets)} pair(s) of consecutive PVC-like beats", len(couplets),
                                [int(i) for s in couplets for i in (s, s + 1)]))
    if v_long:
        s, L = max(v_long, key=lambda t: t[1])
        rate = _rate_of_run(s, L)
        fast = np.isfinite(rate) and rate >= 100
        findings.append(Finding("vt_run",
                                "Possible run of ventricular ectopy" + (" (VT-like, ≥3 beats ≥100 bpm)" if fast else " (slow, ≥3 beats)"),
                                "urgent" if fast else "notable", float((0.6 if fast else 0.5) * q_scale),
                                f"{len(v_long)} run(s) of ≥3 consecutive PVC-like beats, longest {L} beats at ~{_f(rate)} bpm — seek clinical review",
                                len(v_long), [int(i) for s0, L0 in v_long for i in range(s0, s0 + L0)]))
    svt_runs = [(s, L) for s, L in s_runs if L >= 3 and _rate_of_run(s, L) >= 100]
    if svt_runs:
        s, L = max(svt_runs, key=lambda t: t[1])
        findings.append(Finding("svt_run", "Possible run of supraventricular ectopy (≥3 beats ≥100 bpm)",
                                "notable", float(0.5 * q_scale),
                                f"{len(svt_runs)} run(s), longest {L} beats at ~{_f(_rate_of_run(s, L))} bpm",
                                len(svt_runs), [int(i) for s0, L0 in svt_runs for i in range(s0, s0 + L0)]))
    if big_v >= 3 or big_s >= 3:
        which, cyc, bts = ("ventricular", big_v, big_v_beats) if big_v >= big_s else ("supraventricular", big_s, big_s_beats)
        findings.append(Finding("bigeminy", f"Possible {which} bigeminy", "notable", float(0.6 * q_scale),
                                f"every other beat ectopic for {cyc} consecutive cycles", cyc, [int(i) for i in bts]))
    if tri_v >= 3 or tri_s >= 3:
        which, cyc, bts = ("ventricular", tri_v, tri_v_beats) if tri_v >= tri_s else ("supraventricular", tri_s, tri_s_beats)
        findings.append(Finding("trigeminy", f"Possible {which} trigeminy", "notable", float(0.5 * q_scale),
                                f"every third beat ectopic for {cyc} consecutive cycles", cyc, [int(i) for i in bts]))

    # ---------------------------------------------------------------- pauses
    # Absolute rule (> 2.0 s) always applies (away from detector artifacts). The relative
    # rule (> 1.8× the prevailing RR) is only trusted between two normal beats — a
    # post-ectopic compensatory pause is not a pause — and when the surrounding N-N
    # rhythm is regular (AF-like irregularity produces long RRs that are not "pauses").
    pause_idx = []
    gap_idx = []                       # RR ≈ 2x with normal beats either side: missed R-peaks?
    rr_nn_all = rr[(labels[:-1] == "N") & (labels[1:] == "N")]
    if len(rr_nn_all) >= 8:
        cv_robust = 1.4826 * float(np.median(np.abs(rr_nn_all - np.median(rr_nn_all)))) / float(np.median(rr_nn_all))
    else:
        cv_robust = 0.0
    for i in range(1, nb):
        if labels[i] == "?" or labels[i - 1] == "?":
            continue
        if rr_pre[i] > PAUSE_ABS_S:
            pause_idx.append(i)
            continue
        if labels[i] != "N" or labels[i - 1] != "N" or not np.isfinite(loc2[i]) or cv_robust > 0.12:
            continue
        if rr_pre[i] > PAUSE_RATIO * loc2[i]:
            lo, hi = max(0, i - 9), min(nb - 1, i + 8)
            nbr = [rr[j] for j in range(lo, hi) if j != i - 1 and labels[j] == "N" and labels[j + 1] == "N"
                   and rr[j] < 1.7 * loc2[i]]
            if len(nbr) >= 4:
                nbr = np.array(nbr)
                cv = 1.4826 * float(np.median(np.abs(nbr - np.median(nbr)))) / float(np.median(nbr))
                if cv > 0.15 or rr_pre[i] <= PAUSE_RATIO * float(np.median(nbr)):
                    continue
            ratio = rr_pre[i] / loc2[i]
            if 1.8 <= ratio <= 2.25 and corr[i] >= 0.9 and corr[i - 1] >= 0.9:
                gap_idx.append(i)
            else:
                pause_idx.append(i)
    # ≈2x gaps: repeated (≥ 3) or on a poor-quality trace they are far more likely missed
    # R-peaks than pauses → one 'info' finding; isolated ones on a good trace stay pauses.
    if gap_idx and (len(gap_idx) >= 3 or sqi < 0.6):
        findings.append(Finding("possible_missed_beats",
                                "Regular gaps ≈ 2× RR — likely missed R-peaks or dropped beats", "info",
                                float(0.4 * q_scale),
                                f"{len(gap_idx)} RR interval(s) at 1.8–2.25× the prevailing RR with normal beats "
                                f"on both sides (SQI {sqi:.2f}); possible missed R-peak detections or dropped beats",
                                len(gap_idx), [int(i) for i in gap_idx]))
    else:
        pause_idx = sorted(pause_idx + gap_idx)
    longest_rr = float(np.nanmax(rr)) if len(rr) else NAN
    if pause_idx:
        ratios = np.array([rr_pre[i] / loc2[i] for i in pause_idx])
        longest_p = max(rr_pre[i] for i in pause_idx)
        dropped = np.sum((ratios > 1.8) & (ratios < 2.25))
        ev = f"{len(pause_idx)} RR interval(s) > {PAUSE_ABS_S:.1f} s or > {PAUSE_RATIO}× local median; longest {longest_p:.2f} s"
        if dropped:
            ev += f"; {int(dropped)} ≈ 2× the local RR (possible dropped beat / 2nd-degree AV block, or a missed R-peak)"
        findings.append(Finding("pause", "Pause / possible dropped beat", "urgent" if longest_p >= 3.0 else "notable",
                                float(0.55 * q_scale), ev, len(pause_idx), [int(i) for i in pause_idx]))

    # ---------------------------------------------------------------- interval flags
    pr = qrs = qtc = NAN
    if intervals is not None:
        pr = float(getattr(intervals, "pr_ms", NAN) or NAN)
        qrs = float(getattr(intervals, "qrs_ms", NAN) or NAN)
        qtc = float(getattr(intervals, "qtc_bazett_ms", NAN) or NAN)
        if np.isfinite(pr) and pr > 200:
            findings.append(Finding("first_degree_av_block", "Possible 1st-degree AV block", "info", 0.6 * q_scale,
                                    f"PR {pr:.0f} ms (> 200 ms) on the median beat"))
        elif np.isfinite(pr) and pr < 120:
            findings.append(Finding("short_pr", "Short PR interval", "info", 0.5 * q_scale,
                                    f"PR {pr:.0f} ms (< 120 ms) on the median beat"))
        if np.isfinite(qrs) and qrs >= 120:
            findings.append(Finding("wide_qrs", "Wide QRS (bundle branch block / intraventricular conduction delay pattern)",
                                    "notable", 0.6 * q_scale, f"QRS {qrs:.0f} ms (≥ 120 ms) on the median beat"))
        if np.isfinite(qtc) and qtc > 460:
            findings.append(Finding("long_qtc", "Prolonged QTc (Bazett)", "urgent" if qtc > 500 else "notable",
                                    0.55 * q_scale, f"QTc {qtc:.0f} ms (> {'500' if qtc > 500 else '460'} ms)"))
        elif np.isfinite(qtc) and qtc < 340:
            findings.append(Finding("short_qtc", "Short QTc (Bazett)", "notable", 0.5 * q_scale,
                                    f"QTc {qtc:.0f} ms (< 340 ms)"))

    # ---------------------------------------------------------------- rate screens (N-N RRs)
    rr_nn = rr[(labels[:-1] == "N") & (labels[1:] == "N")]
    rr_nn = rr_nn[np.isfinite(rr_nn) & (rr_nn > 0.2)]
    hr_nn = float(60.0 / np.median(rr_nn)) if len(rr_nn) >= 3 else NAN
    cv_nn = float(np.std(rr_nn) / np.mean(rr_nn)) if len(rr_nn) >= 3 else NAN
    hr_ref = hr_nn
    if intervals is not None and np.isfinite(getattr(intervals, "hr_mean", NAN) or NAN) and not np.isfinite(hr_ref):
        hr_ref = float(intervals.hr_mean)
    narrow = (np.isfinite(qrs) and qrs < 120) or (not np.isfinite(qrs) and ref_width < 120)
    if np.isfinite(hr_ref) and np.isfinite(cv_nn):
        regular = cv_nn < 0.10
        if hr_ref >= VT_HR and regular and not narrow:
            findings.append(Finding("vt", "Possible ventricular tachycardia (regular wide-complex tachycardia) — seek clinical review",
                                    "urgent", 0.5 * q_scale, f"HR {hr_ref:.0f} bpm, RR CV {cv_nn*100:.0f}%, QRS {_f(qrs if np.isfinite(qrs) else ref_width)} ms"))
        elif hr_ref >= SVT_HR and regular and narrow:
            findings.append(Finding("svt", "Possible SVT (regular narrow-complex tachycardia) — seek clinical review",
                                    "urgent" if hr_ref >= 170 else "notable", 0.5 * q_scale,
                                    f"HR {hr_ref:.0f} bpm, RR CV {cv_nn*100:.0f}%, narrow QRS"))
        elif hr_ref > 100:
            findings.append(Finding("sinus_tachy", "Sinus tachycardia (rate > 100 bpm)", "info", 0.6 * q_scale,
                                    f"HR {hr_ref:.0f} bpm from normal-to-normal RRs"))
        elif hr_ref < 60:
            findings.append(Finding("sinus_brady", "Bradycardia" + (" (< 40 bpm)" if hr_ref < 40 else " (rate < 60 bpm)"),
                                    "notable" if hr_ref < 40 else "info", 0.6 * q_scale,
                                    f"HR {hr_ref:.0f} bpm from normal-to-normal RRs"))
        if cv_nn > 0.15 and cv_robust > 0.12 and len(pause_idx) < 0.02 * nb:
            # N-N RRs already exclude ectopic-adjacent intervals, so ectopy does not explain
            # this; a few pauses/outliers do not either (robust CV). AF is NOT claimed here.
            findings.append(Finding("irregular", "Irregular rhythm (not explained by ectopy)", "info", 0.4 * q_scale,
                                    f"normal-to-normal RR CV {cv_nn*100:.0f}% (> 15%); AF is assessed separately"))

    if sqi < 0.5:
        # poor trace: ectopy / run / tachycardia findings are never "urgent"
        for f in findings:
            if f.code in ("pvc", "pac", "couplet", "vt_run", "svt_run", "bigeminy", "trigeminy",
                          "vt", "svt") and f.severity == "urgent":
                f.severity = "notable"
                f.evidence += " (severity capped: low signal quality)"
    findings = _sev_sort(findings)

    # ---------------------------------------------------------------- metrics + summary
    def _jf(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return NAN
        return v if np.isfinite(v) else NAN

    metrics = {
        "n_beats": float(nb), "n_unclassified": float(n_q),
        "pvc_count": float(n_v), "pac_count": float(n_s),
        "uncertain_v_count": float(uncertain_v), "uncertain_s_count": float(uncertain_s),
        "pvc_burden": _jf(pvc_burden), "pac_burden": _jf(pac_burden),
        "pvc_per_min": _jf(n_v / duration_min if duration_min > 0 else NAN),
        "n_couplets": float(len(couplets)), "n_v_runs": float(len(v_long)), "max_run_len": float(max_run),
        "n_svt_runs": float(len(svt_runs)),
        "bigeminy_cycles": float(max(big_v, big_s)), "trigeminy_cycles": float(max(tri_v, tri_s)),
        "n_pauses": float(len(pause_idx)), "n_gap_2x": float(len(gap_idx)),
        "longest_rr_ms": _jf(longest_rr * 1000.0),
        "mean_rr_ms": _jf(np.nanmean(rr) * 1000.0), "hr_nn": _jf(hr_nn), "rr_cv_nn": _jf(cv_nn),
        "ref_qrs_width_ms": _jf(ref_width), "median_corr": _jf(np.nanmedian(corr)),
        "sqi": _jf(sqi),
    }
    parts = []
    for f in findings:
        if f.code == "pvc":
            parts.append(("Frequent " if "frequent" in f.label else "") + f"PVCs ({_pct(pvc_burden)} burden)")
        elif f.code == "pac":
            parts.append(("Frequent " if "frequent" in f.label else "") + f"PACs ({_pct(pac_burden)} burden)")
        elif f.code == "couplet":
            parts.append(f"{f.count} couplet(s)")
        elif f.code == "vt_run":
            parts.append(f"ventricular run of {max_run}")
        elif f.code == "svt_run":
            parts.append("SVT-like run")
        elif f.code in ("bigeminy", "trigeminy"):
            parts.append(f.label.replace("Possible ", ""))
        elif f.code == "pause":
            parts.append(f"{f.count} pause(s), longest {longest_rr:.1f} s")
        elif f.code == "possible_missed_beats":
            parts.append(f"{f.count} gap(s) ≈2× RR (missed beats?)")
        elif f.code == "first_degree_av_block":
            parts.append(f"PR {pr:.0f} ms (1st-degree AV block)")
        elif f.code == "short_pr":
            parts.append(f"PR {pr:.0f} ms (short)")
        elif f.code == "wide_qrs":
            parts.append(f"QRS {qrs:.0f} ms (wide)")
        elif f.code in ("long_qtc", "short_qtc"):
            parts.append(f"QTc {qtc:.0f} ms")
        elif f.code in ("svt", "vt"):
            parts.append(f"possible {f.code.upper()} at {hr_ref:.0f} bpm")
        elif f.code == "sinus_brady":
            parts.append(f"bradycardia {hr_ref:.0f} bpm")
        elif f.code == "sinus_tachy":
            parts.append(f"tachycardia {hr_ref:.0f} bpm")
        elif f.code == "irregular":
            parts.append("irregular RR")
    summary = "; ".join(parts)
    if summary:
        summary = summary[0].upper() + summary[1:] + " — possible findings, research only"
    rep = ArrhythmiaReport(labels, findings, metrics, summary, warns)
    # per-beat features (not part of the stable interface; handy for plotting/debugging)
    rep.beat_features = {"rr_pre_s": rr_pre, "ref_rr_s": ref_arr, "prematurity": prem,
                         "comp_ratio": comp, "corr": corr, "qrs_width_ms": widths,
                         "amp_ratio": amp_ratio, "confidence": conf}
    return rep
