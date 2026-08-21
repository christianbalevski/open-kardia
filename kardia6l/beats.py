"""Beat timing for the animated heart: R-peak times + per-reading wave fiducials.

The 3D heart (kardia6l/web/heart.js) is driven purely by *timing*: for every beat it needs
the R-peak time and, relative to R, the onsets/offsets of the P wave, the QRS complex and
the T wave. Those map onto the mechanical cycle (atrial systole follows P, ventricular
systole follows QRS, relaxation follows T) with the usual electromechanical delays.

This module produces that timing in one JSON-safe shape for BOTH the live stream and the
saved-reading playback, so the two look identical:

    {
      "fs": 300.0,
      "hr": 71.5,
      "beats": [t_R, ...],                 # seconds on the signal's own time base
      "labels": ["N","N","V", ...] | null, # per-beat type from kardia6l.arrhythmia (N/S/V/?)
      "fid": {"p_on": -0.20, "p_off": -0.10, "q_on": -0.04, "q_off": 0.05,
              "t_on": 0.16, "t_off": 0.38},  # seconds relative to R (median beat)
      "fid_source": "delineated" | "default"
    }

* R-peaks come from `dsp.detect_r_peaks` (Pan–Tompkins).
* The wave fiducials come from delineating the ensemble MEDIAN beat of the reading
  (`analysis._median_beat` + neurokit2 DWT via `analysis._delineate_median`) — the same
  numbers the analysis panel reports as PR / QRS / QT. If that fails (short/noisy signal,
  neurokit missing) we fall back to textbook intervals scaled to the heart rate.

RESEARCH SIGNAL — NOT A MEDICAL DEVICE.
"""
from __future__ import annotations

import math

import numpy as np

from . import dsp

# textbook adult resting values (seconds), used only as a fallback
_DEF_PR = 0.16       # P onset -> QRS onset
_DEF_P_DUR = 0.10
_DEF_QRS = 0.09
_DEF_Q_TO_R = 0.04   # QRS onset -> R peak
_DEF_T_DUR = 0.18
_DEF_QTC = 0.40      # Bazett-corrected QT


def default_fiducials(rr_s: float | None) -> dict:
    """Textbook fiducials (seconds relative to R) for a given RR interval."""
    rr = rr_s if (rr_s is not None and math.isfinite(rr_s) and 0.3 < rr_s < 2.5) else 0.85
    qt = _DEF_QTC * math.sqrt(rr)                 # inverse Bazett
    q_on = -_DEF_Q_TO_R
    return {
        "p_on": q_on - _DEF_PR,
        "p_off": q_on - _DEF_PR + _DEF_P_DUR,
        "q_on": q_on,
        "q_off": q_on + _DEF_QRS,
        "t_on": q_on + qt - _DEF_T_DUR,
        "t_off": q_on + qt,
    }


def _sane(fid: dict) -> bool:
    """Reject physiologically impossible delineations (keeps the animation from jumping)."""
    try:
        p_on, p_off, q_on, q_off, t_on, t_off = (fid[k] for k in
                                                ("p_on", "p_off", "q_on", "q_off", "t_on", "t_off"))
    except KeyError:
        return False
    vals = [p_on, p_off, q_on, q_off, t_on, t_off]
    if not all(isinstance(v, float) and math.isfinite(v) for v in vals):
        return False
    return (-0.40 < p_on < p_off < q_on < 0.0 < q_off < t_on < t_off < 0.75
            and 0.03 <= (p_off - p_on) <= 0.20
            and 0.04 <= (q_off - q_on) <= 0.20
            and 0.06 <= (t_off - t_on) <= 0.45
            and 0.20 <= (t_off - q_on) <= 0.70)


def delineate_fiducials(clean_primary: np.ndarray, rpeaks: np.ndarray, fs: float) -> dict | None:
    """Median-beat delineation -> fiducials relative to R (seconds), or None on failure."""
    if len(rpeaks) < 4:
        return None
    try:
        from .analysis import _delineate_median, _median_beat
        rr_med = int(round(np.median(np.diff(rpeaks))))
        beat, r = _median_beat(clean_primary, rpeaks, fs, rr_samp=rr_med)
        if beat is None:
            return None
        d = _delineate_median(beat, r, fs)
    except Exception:  # noqa: BLE001
        return None
    if not d:
        return None

    def rel(key):
        v = d.get(key, float("nan"))
        return (float(v) - r) / fs if v == v else float("nan")

    fid = {"p_on": rel("ECG_P_Onsets"), "p_off": rel("ECG_P_Offsets"),
           "q_on": rel("ECG_R_Onsets"), "q_off": rel("ECG_R_Offsets"),
           "t_on": rel("ECG_T_Onsets"), "t_off": rel("ECG_T_Offsets")}
    # a missing T onset is common; synthesize it from the offset
    if not math.isfinite(fid["t_on"]) and math.isfinite(fid["t_off"]):
        fid["t_on"] = fid["t_off"] - _DEF_T_DUR
    # the DWT delineator often clips the P offset on small/low-lead P waves; a P wave shorter
    # than 60 ms is not physiologic, so widen it toward the textbook duration
    if (math.isfinite(fid["p_on"]) and math.isfinite(fid["p_off"]) and math.isfinite(fid["q_on"])
            and fid["p_off"] - fid["p_on"] < 0.06):
        fid["p_off"] = min(fid["p_on"] + 0.09, fid["q_on"] - 0.02)
    return fid if _sane(fid) else None


def beat_labels(clean_primary: np.ndarray, fs: float, peaks: np.ndarray) -> list[str] | None:
    """Per-beat 'N' / 'S' / 'V' / '?' labels from kardia6l.arrhythmia (None if unavailable)."""
    if len(peaks) < 4:
        return None
    try:
        from .arrhythmia import detect_arrhythmias
        rep = detect_arrhythmias(clean_primary, fs, peaks)
        labels = [str(v) for v in np.asarray(rep.beat_labels).tolist()]
        return labels if len(labels) == len(peaks) else None
    except Exception:  # noqa: BLE001
        return None


def beat_timing(x_primary: np.ndarray, fs: float, t0: float = 0.0, mains: float = 60.0,
                delineate: bool = True, prev_fid: dict | None = None,
                label_beats: bool = True) -> dict:
    """R-peaks + fiducials for one lead. `t0` is the time of sample 0 on the caller's
    time base (so live ring slices and whole recordings share one shape).

    prev_fid: keep using an earlier good delineation instead of falling back to defaults
    when the current window can't be delineated (live use)."""
    x = np.asarray(x_primary, dtype=np.float64)
    out = {"fs": float(fs), "hr": float("nan"), "beats": [], "labels": None, "fid": None,
           "fid_source": "default"}
    if len(x) < int(2 * fs):
        out["fid"] = default_fiducials(None)
        return out
    try:
        clean = dsp.clean(x, fs, mains)
    except Exception:  # noqa: BLE001
        clean = x
    peaks = dsp.detect_r_peaks(clean, fs)
    hr = dsp.heart_rate_from_peaks(peaks, fs)
    rr = 60.0 / hr if (hr == hr and hr > 0) else None
    out["hr"] = float(hr) if hr == hr else float("nan")
    out["beats"] = [round(t0 + p / fs, 4) for p in peaks.tolist()]
    out["labels"] = beat_labels(clean, fs, peaks) if label_beats else None
    fid = delineate_fiducials(clean, peaks, fs) if delineate else None
    if fid is not None:
        out["fid"] = {k: round(v, 4) for k, v in fid.items()}
        out["fid_source"] = "delineated"
    elif prev_fid:
        out["fid"] = dict(prev_fid)
        out["fid_source"] = "delineated"
    else:
        out["fid"] = {k: round(v, 4) for k, v in default_fiducials(rr).items()}
    return out


def primary_lead_index(names: list[str]) -> int:
    """Lead II if present (best P waves), else I, else column 0 — same rule as analysis."""
    names = list(names)
    for want in ("II", "I"):
        if want in names:
            return names.index(want)
    return 0


def playback_bundle(x, fs: float, names: list[str], mains: float = 60.0,
                    decimals: int = 4) -> dict:
    """Everything the browser needs to replay a saved reading with the animated heart:
    a display-filtered signal for every lead plus the beat timing. JSON-safe."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    names = list(names)[: x.shape[1]]
    try:
        xf = dsp.display_filter(x, fs, mains) if n > int(3 * fs) else x
    except Exception:  # noqa: BLE001
        xf = x
    xf = np.nan_to_num(np.asarray(xf, dtype=np.float64))
    li = primary_lead_index(names)
    timing = beat_timing(x[:, li], fs, t0=0.0, mains=mains, delineate=True)
    return {
        "fs": float(fs),
        "leads": names,
        "t0": 0.0,
        "duration_s": n / fs,
        "data": [np.round(xf[:, j], decimals).tolist() for j in range(xf.shape[1])],
        "beats": timing["beats"],
        "labels": timing.get("labels"),
        "fid": timing["fid"],
        "fid_source": timing["fid_source"],
        "hr": timing["hr"] if timing["hr"] == timing["hr"] else None,
        "primary": names[li] if names else "I",
        "screen": live_screen(x[:, li], fs, mains=mains),
    }


def live_screen(x_primary: np.ndarray, fs: float, mains: float = 60.0, t0: float = 0.0) -> dict:
    """Rhythm screen for the live panel on the last ~30 s: arrhythmia findings + AF screen.
    JSON-safe; never raises. {'findings': [{code,label,severity,confidence,evidence}], 'summary',
    'af': {'label','score','evidence'} | None, 'n_beats'}"""
    out = {"findings": [], "summary": "", "af": None, "n_beats": 0}
    x = np.asarray(x_primary, dtype=np.float64)
    if len(x) < int(8 * fs):
        return out
    try:
        clean = dsp.clean(x, fs, mains)
        peaks = dsp.detect_r_peaks(clean, fs)
        out["n_beats"] = int(len(peaks))
        if len(peaks) < 6:
            return out
        try:
            q = dsp.signal_quality(x, fs)
            sqi = float(q.sqi)
        except Exception:  # noqa: BLE001
            sqi = 1.0
        try:
            from .arrhythmia import detect_arrhythmias
            rep = detect_arrhythmias(clean, fs, peaks, sqi=sqi)
            if t0:
                from .arrhythmia import annotate_times
                annotate_times(rep.findings, peaks, fs, t0=t0)     # absolute time base
            out["findings"] = [{"code": f.code, "label": f.label, "severity": f.severity,
                                "confidence": float(f.confidence), "evidence": f.evidence,
                                "times": [float(v) for v in f.times][:200],
                                "episodes": [(float(a), float(b)) for a, b in f.episodes][:200],
                                "when": f.when()}
                               for f in rep.findings]
            out["summary"] = rep.summary
        except Exception:  # noqa: BLE001
            pass
        try:
            from .af import af_episodes, af_score
            rr = np.diff(peaks) / fs * 1000.0
            af = af_score(rr, p_consistent=None, sqi=sqi)
            eps = af_episodes(rr, t0 + peaks / fs, p_consistent=None, sqi=sqi) if af.is_af else []
            out["af"] = {"label": af.label, "score": (float(af.score) if af.score == af.score else None),
                         "is_af": bool(af.is_af), "evidence": af.evidence,
                         "episodes": [(float(a), float(b)) for a, b, _c in eps]}
            if af.is_af:
                out["findings"].insert(0, {"code": "af", "label": "Possible atrial fibrillation", "severity": "urgent",
                                           "confidence": float(af.confidence), "evidence": af.evidence,
                                           "times": [], "episodes": [(float(a), float(b)) for a, b, _c in eps],
                                           "when": ("episodes " + ", ".join(f"{a:.1f}–{b:.1f} s" for a, b, _c in eps[:6])) if eps else "whole window"})
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass
    return out
