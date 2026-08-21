"""Post-recording ECG rhythm evaluation and interval/measurement extraction.

RESEARCH SIGNAL — NOT A MEDICAL DEVICE. Nothing here is clinically validated or
suitable for diagnosis. Measurements are automated estimates on a consumer single-
/six-lead recording and can be wrong, especially on noisy data.

Everything is pure numpy/scipy/neurokit2 (no GUI, no BLE). `analyze_recording` never
raises on bad input: it returns an `AnalysisResult` with nan fields and `warnings`.

Method summary:
  * primary lead = II (best P waves) if present, else I, else lead 0
  * R-peaks via dsp.detect_r_peaks (Pan-Tompkins); RR/HR from the peak train
  * an ensemble MEDIAN BEAT is built per lead (high SNR), then delineated by tiling it
    and running neurokit2's discrete-wavelet delineator — much steadier than per-beat
    delineation on raw noisy data
  * intervals (P, PR, QRS, QT, QTc-Bazett/Fridericia, ST level, T duration) from the
    median-beat fiducials; RR mean/SD and HR min/mean/max from the peak train
  * frontal-plane QRS axis from net QRS area in I and aVF (6-lead only)
  * a conservative heuristic rhythm label from HR + RR irregularity + P-wave consistency
  * HRV time-domain (SDNN, RMSSD, pNN50) via neurokit2 with a manual fallback
"""
from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import dsp
from .leads import LEAD_NAMES_6

NAN = float("nan")


# --------------------------------------------------------------------------- dataclasses
# Adult resting reference ranges (research aid, not diagnostic thresholds). Each entry:
#   key -> (low, high, unit, note).  low/high None means "one-sided" or not range-defined.
REFERENCE_RANGES: dict[str, tuple] = {
    "hr_mean":           (60.0, 100.0, "bpm", "resting adult"),
    "rr_mean_ms":        (600.0, 1000.0, "ms", "for HR 60–100"),
    "p_dur_ms":          (80.0, 120.0, "ms", "P wave < 120 ms"),
    "pr_ms":             (120.0, 200.0, "ms", "PR interval"),
    "qrs_ms":            (70.0, 110.0, "ms", "< 120 ms"),
    "qt_ms":             (None, None, "ms", "rate-dependent — use QTc"),
    "qtc_bazett_ms":     (350.0, 450.0, "ms", "M < 450, F < 460"),
    "qtc_fridericia_ms": (350.0, 450.0, "ms", "M < 450, F < 460"),
    "st_level_mv":       (-0.1, 0.1, "mV", "≤ 0.1 mV deviation"),
    "t_dur_ms":          (120.0, 200.0, "ms", "approximate"),
    "qrs_axis_deg":      (-30.0, 90.0, "°", "normal frontal axis"),
}


def reference_range(key: str):
    """(low, high, unit, note) for a measurement key, or None if not defined."""
    return REFERENCE_RANGES.get(key)


def range_text(key: str) -> str:
    """Human-readable normal range, e.g. '120–200 ms' or '< 120 ms' or '−30–90°'."""
    r = REFERENCE_RANGES.get(key)
    if not r:
        return ""
    lo, hi, unit, _ = r
    u = unit if unit in ("°",) else (" " + unit if unit else "")
    if lo is None and hi is None:
        return ""
    if lo is None:
        return f"< {hi:g}{u}"
    if hi is None:
        return f"> {lo:g}{u}"
    dash = "–"
    lo_s = f"{lo:g}".replace("-", "−")
    hi_s = f"{hi:g}".replace("-", "−")
    return f"{lo_s}{dash}{hi_s}{u}"


def classify(key: str, value) -> str:
    """'low' | 'normal' | 'high' | '' (unknown/undefined) for a measurement value."""
    r = REFERENCE_RANGES.get(key)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not r or not np.isfinite(v):
        return ""
    lo, hi, _, _ = r
    if lo is not None and v < lo:
        return "low"
    if hi is not None and v > hi:
        return "high"
    return "normal"


@dataclass
class Intervals:
    rr_mean_ms: float = NAN
    rr_sd_ms: float = NAN
    hr_mean: float = NAN
    hr_min: float = NAN
    hr_max: float = NAN
    p_dur_ms: float = NAN
    pr_ms: float = NAN
    qrs_ms: float = NAN
    qt_ms: float = NAN
    qtc_bazett_ms: float = NAN
    qtc_fridericia_ms: float = NAN
    st_level_mv: float = NAN
    t_dur_ms: float = NAN
    measured_on: str = ""          # which lead the intervals were measured on

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class RhythmAssessment:
    label: str = "Undetermined"
    confidence: float = 0.0
    rationale: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    fs: float
    duration_s: float
    lead_used: str
    n_beats: int
    intervals: Intervals
    rhythm: RhythmAssessment
    hrv: dict
    qrs_axis_deg: float
    axis_label: str
    median_beat: dict            # {lead: np.ndarray}
    median_beat_fs: float
    fiducials: dict              # sample indices on the median beat (nan-safe)
    quality_sqi: float
    warnings: list[str] = field(default_factory=list)
    arrhythmia: Any = None       # open_kardia.arrhythmia.ArrhythmiaReport (ectopy, pauses, conduction) or None
    af: Any = None               # open_kardia.af.AFResult (RR-based atrial-fibrillation screen) or None

    def rows(self) -> list[dict]:
        """Labeled measurement rows with value, unit, normal range, and status
        ('low'/'normal'/'high'/'') — the single source the GUI panel and PDF report render.
        Each row: {label, key, value, unit, digits, range, status}."""
        iv = self.intervals
        spec = [
            ("Heart rate", "hr_mean", iv.hr_mean, "bpm", 0),
            ("RR interval", "rr_mean_ms", iv.rr_mean_ms, "ms", 0),
            ("P duration", "p_dur_ms", iv.p_dur_ms, "ms", 0),
            ("PR interval", "pr_ms", iv.pr_ms, "ms", 0),
            ("QRS duration", "qrs_ms", iv.qrs_ms, "ms", 0),
            ("QT interval", "qt_ms", iv.qt_ms, "ms", 0),
            ("QTc (Bazett)", "qtc_bazett_ms", iv.qtc_bazett_ms, "ms", 0),
            ("QTc (Fridericia)", "qtc_fridericia_ms", iv.qtc_fridericia_ms, "ms", 0),
            ("ST level", "st_level_mv", iv.st_level_mv, "mV", 2),
            ("T duration", "t_dur_ms", iv.t_dur_ms, "ms", 0),
            ("QRS axis", "qrs_axis_deg", self.qrs_axis_deg, "°", 0),
        ]
        out = []
        for label, key, val, unit, digits in spec:
            out.append({"label": label, "key": key, "value": val, "unit": unit,
                        "digits": digits, "range": range_text(key), "status": classify(key, val)})
        return out

    def summary(self) -> str:
        iv = self.intervals

        def f(v, u="ms", d=0):
            return "--" if not np.isfinite(v) else f"{v:.{d}f} {u}"

        def fr(v, key, u="ms", d=0):
            """value with its normal range, e.g. 'PR 153 ms (normal 120–200 ms)'."""
            s = f(v, u, d)
            rt = range_text(key)
            st = classify(key, v)
            tag = f" [{st.upper()}]" if st in ("low", "high") else ""
            return f"{s} (normal {rt}){tag}" if rt else s

        lines = [
            f"Rhythm: {self.rhythm.label} ({self.rhythm.confidence*100:.0f}% conf) — {self.rhythm.rationale}",
            f"Beats: {self.n_beats}   Duration: {self.duration_s:.1f}s   Lead: {self.lead_used}   SQI: {self.quality_sqi:.2f}",
            f"HR: {fr(iv.hr_mean,'hr_mean','bpm')} (min {f(iv.hr_min,'bpm')}, max {f(iv.hr_max,'bpm')})",
            f"RR: {fr(iv.rr_mean_ms,'rr_mean_ms')} ± {f(iv.rr_sd_ms)}",
            f"P dur: {fr(iv.p_dur_ms,'p_dur_ms')}",
            f"PR: {fr(iv.pr_ms,'pr_ms')}",
            f"QRS: {fr(iv.qrs_ms,'qrs_ms')}",
            f"QT: {f(iv.qt_ms)} (rate-dependent — use QTc)",
            f"QTc(Bazett): {fr(iv.qtc_bazett_ms,'qtc_bazett_ms')}",
            f"QTc(Fridericia): {fr(iv.qtc_fridericia_ms,'qtc_fridericia_ms')}",
            f"ST level: {fr(iv.st_level_mv,'st_level_mv','mV',2)}",
            f"T dur: {fr(iv.t_dur_ms,'t_dur_ms')}",
            f"QRS axis: {f(self.qrs_axis_deg,'deg')} ({self.axis_label}, normal {range_text('qrs_axis_deg')})",
            f"HRV: SDNN {f(self.hrv.get('SDNN',NAN))}  RMSSD {f(self.hrv.get('RMSSD',NAN))}  "
            f"pNN50 {f(self.hrv.get('pNN50',NAN),'%',1)}",
        ]
        findings = list(getattr(self.arrhythmia, "findings", []) or [])
        if findings:
            lines.append("Rhythm findings (automated screen, not a diagnosis):")
            for fnd in findings:
                when = fnd.when() if hasattr(fnd, "when") else ""
                lines.append(f"  [{str(fnd.severity).upper()}] {fnd.label} — {fnd.evidence}"
                             + (f" ({when})" if when else ""))
        if self.af is not None and getattr(self.af, "evidence", ""):
            eps = getattr(self.af, "episodes", []) or []
            ep_txt = ("; episodes " + ", ".join(f"{a:.1f}–{b:.1f} s" for a, b, _s in eps[:6])) if eps else ""
            lines.append(f"AF screen: {self.af.label} (score {getattr(self.af, 'score', NAN):.2f}) — {self.af.evidence}{ep_txt}")
        if self.rhythm.flags:
            lines.append("Flags: " + ", ".join(self.rhythm.flags))
        if self.warnings:
            lines.append("Warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers
def _nk():
    try:
        import neurokit2 as nk  # noqa
        return nk
    except Exception:  # noqa: BLE001
        return None


def _median_beat(clean_sig: np.ndarray, rpeaks: np.ndarray, fs: float,
                 pre_s: float = 0.4, rr_samp: int | None = None):
    """Ensemble-median beat aligned on R. Returns (beat, r_index_in_beat) or (None, 0)."""
    n = len(clean_sig)
    pre = int(pre_s * fs)
    if rr_samp is None:
        rr_samp = int(round(np.median(np.diff(rpeaks)))) if len(rpeaks) > 1 else int(0.8 * fs)
    post = max(int(0.5 * fs), rr_samp - pre)
    beats = [clean_sig[r - pre:r + post] for r in rpeaks if r - pre >= 0 and r + post < n]
    if len(beats) < 3:
        return None, 0
    L = min(len(b) for b in beats)
    beats = np.vstack([b[:L] for b in beats])
    return np.median(beats, axis=0), pre


def _delineate_median(beat: np.ndarray, r_idx: int, fs: float, reps: int = 12):
    """Delineate an ensemble median beat by tiling it into a clean strip and running
    neurokit2's DWT delineator. Returns a dict of median offsets (samples, within one
    beat) for P/Q/R/S/T onsets/offsets, or {} on failure."""
    nk = _nk()
    if nk is None or beat is None:
        return {}
    L = len(beat)
    strip = np.tile(beat, reps)
    rpeaks = np.array([r_idx + i * L for i in range(reps)])
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            _, w = nk.ecg_delineate(strip, rpeaks=rpeaks, sampling_rate=int(fs), method="dwt")
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, float] = {}
    for key in ("ECG_P_Onsets", "ECG_P_Offsets", "ECG_P_Peaks", "ECG_Q_Peaks",
                "ECG_R_Onsets", "ECG_R_Offsets", "ECG_S_Peaks",
                "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets"):
        v = np.array(w.get(key, []), dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            out[key] = NAN
            continue
        # collapse to within-beat offset (0..L), take the median across reps
        off = v - (v // L) * L
        # unwrap offsets that land just before R (e.g. P slightly negative wrap)
        out[key] = float(np.median(off))
    return out


def _safe_ms(a: float, b: float, fs: float, lo: float = 0.0, hi: float = 1000.0) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return NAN
    d = (b - a) / fs * 1000.0
    return d if lo < d < hi else NAN


def _qrs_axis(median_beats: dict, r_idx: dict, fs: float) -> tuple[float, str]:
    """Frontal-plane QRS axis from net QRS area in I and aVF (samples ±50 ms around R)."""
    if "I" not in median_beats or "aVF" not in median_beats:
        return NAN, "Undetermined"
    w = int(0.05 * fs)
    def net(lead):
        b = median_beats[lead]; r = r_idx.get(lead, len(b) // 2)
        lo, hi = max(0, r - w), min(len(b), r + w)
        seg = b[lo:hi]
        base = np.median(b[:max(1, int(0.05 * fs))])
        return float(np.sum(seg - base))
    ni, nf = net("I"), net("aVF")
    if abs(ni) < 1e-9 and abs(nf) < 1e-9:
        return NAN, "Undetermined"
    axis = float(np.degrees(np.arctan2(nf, ni)))
    if -30 <= axis <= 90:
        lab = "Normal"
    elif -90 <= axis < -30:
        lab = "Left axis deviation"
    elif 90 < axis <= 180:
        lab = "Right axis deviation"
    else:
        lab = "Extreme axis deviation"
    return axis, lab


def _hrv(rpeaks: np.ndarray, fs: float) -> dict:
    rr = np.diff(rpeaks) / fs * 1000.0
    rr = rr[(rr > 250) & (rr < 3000)]
    out = {"SDNN": NAN, "RMSSD": NAN, "pNN50": NAN}
    if len(rr) < 3:
        return out
    nk = _nk()
    if nk is not None:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                h = nk.hrv_time({"ECG_R_Peaks": np.asarray(rpeaks)}, sampling_rate=int(fs))
            for k in ("SDNN", "RMSSD", "pNN50"):
                col = f"HRV_{k}"
                if col in h:
                    out[k] = float(h[col].iloc[0])
            if np.isfinite(out["SDNN"]):
                return out
        except Exception:  # noqa: BLE001
            pass
    # manual fallback
    out["SDNN"] = float(np.std(rr, ddof=1))
    diff = np.diff(rr)
    out["RMSSD"] = float(np.sqrt(np.mean(diff**2))) if len(diff) else NAN
    out["pNN50"] = float(100.0 * np.mean(np.abs(diff) > 50)) if len(diff) else NAN
    return out


def _robust_rr(rr_ms: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop obvious missed-beat / doubled-beat artifacts (RR outside 0.5..1.75x median),
    which on consumer single-lead traces are almost always detection errors rather than
    real pauses. Returns (cleaned_rr, n_removed). AF-style beat-to-beat irregularity is
    preserved (it lives inside that band)."""
    rr_ms = np.asarray(rr_ms, dtype=float)
    rr_ms = rr_ms[(rr_ms > 250) & (rr_ms < 3000)]
    if len(rr_ms) < 3:
        return rr_ms, 0
    med = np.median(rr_ms)
    keep = (rr_ms > 0.5 * med) & (rr_ms < 1.75 * med)
    return rr_ms[keep], int(np.sum(~keep))


def _classify_rhythm(hr_mean: float, rr_ms: np.ndarray, sqi: float,
                     p_consistent: bool, rmssd: float) -> RhythmAssessment:
    flags: list[str] = []
    rr_ms, n_removed = _robust_rr(rr_ms)
    if sqi < 0.4 or len(rr_ms) < 4:
        return RhythmAssessment("Too noisy to classify", 0.2,
                                f"low signal quality (SQI {sqi:.2f}) or too few beats", flags)
    if n_removed:
        flags.append(f"{n_removed} beat(s) excluded as likely detection artifacts")
    rr_cv = float(np.std(rr_ms) / np.mean(rr_ms)) if np.mean(rr_ms) > 0 else NAN
    # irregularly-irregular detector: high beat-to-beat variability
    succ = np.abs(np.diff(rr_ms))
    irr_frac = float(np.mean(succ > 0.15 * np.mean(rr_ms))) if len(succ) else 0.0
    if sqi < 0.6:
        flags.append("low signal quality")
    if not p_consistent:
        flags.append("P waves not consistently detected")

    # Possible AF: irregularly irregular + poor/absent P waves + adequate quality
    if rr_cv > 0.12 and irr_frac > 0.5 and not p_consistent and sqi >= 0.5:
        return RhythmAssessment("Possible atrial fibrillation",
                                min(0.6, 0.3 + irr_frac / 2),
                                f"irregularly irregular RR (CV {rr_cv*100:.0f}%), P waves absent",
                                flags + ["irregularly irregular — NOT a diagnosis"])
    # Generic irregular
    if rr_cv > 0.15 and irr_frac > 0.4:
        # sinus arrhythmia if P waves present and variation is smoother
        if p_consistent and irr_frac < 0.6:
            return RhythmAssessment("Sinus arrhythmia", 0.55,
                                    f"HR {hr_mean:.0f}, RR CV {rr_cv*100:.0f}% with consistent P waves", flags)
        return RhythmAssessment("Irregular rhythm", 0.45,
                                f"variable RR (CV {rr_cv*100:.0f}%)", flags + ["irregular RR"])

    # Regular rhythms by rate
    reg = f"regular RR (CV {rr_cv*100:.0f}%)"
    if not np.isfinite(hr_mean):
        return RhythmAssessment("Undetermined", 0.2, "HR not measurable", flags)
    if hr_mean < 60:
        return RhythmAssessment("Sinus bradycardia", 0.7, f"HR {hr_mean:.0f} bpm, {reg}", flags)
    if hr_mean > 100:
        return RhythmAssessment("Sinus tachycardia", 0.7, f"HR {hr_mean:.0f} bpm, {reg}", flags)
    conf = 0.8 if p_consistent else 0.6
    return RhythmAssessment("Normal sinus rhythm", conf,
                            f"HR {hr_mean:.0f} bpm, {reg}" + ("" if p_consistent else ", P waves uncertain"),
                            flags)


def _screen_arrhythmias(clean_primary, fs, rpeaks, median_beat, r_idx, iv, p_consistent, sqi, warns):
    """Run the beat-level arrhythmia screen and the RR-based AF screen. Both are optional
    modules; failures degrade to None + a warning, never an exception."""
    arr = af = None
    try:
        from .arrhythmia import detect_arrhythmias
        arr = detect_arrhythmias(clean_primary, fs, rpeaks, median_beat=median_beat, r_idx=r_idx,
                                 intervals=iv, p_consistent=p_consistent, sqi=sqi)
    except Exception as e:  # noqa: BLE001
        warns.append(f"arrhythmia screen unavailable: {e}")
    try:
        from .af import af_episodes, af_score
        rr_all = np.diff(np.asarray(rpeaks, dtype=float)) / fs * 1000.0
        af = af_score(rr_all, p_consistent=p_consistent, sqi=sqi)
        try:
            af.episodes = af_episodes(rr_all, np.asarray(rpeaks, dtype=float) / fs,
                                      p_consistent=p_consistent, sqi=sqi)
        except Exception:  # noqa: BLE001
            af.episodes = []
        # Paroxysmal AF inside a long recording: the whole-recording score is diluted by the
        # sinus stretches, but a sustained high-scoring episode is still AF-suggestive.
        strong = [e for e in af.episodes if e[2] >= 0.7 and (e[1] - e[0]) >= 15.0]
        if strong and not af.is_af and af.label != "Undetermined":
            af.is_af = True
            af.label = "Possible atrial fibrillation (episodes)"
            span = "; ".join(f"{a:.0f}–{b:.0f} s (score {sc:.2f})" for a, b, sc in strong[:4])
            af.evidence = (f"AF-like RR irregularity in {len(strong)} sustained episode(s): {span}"
                           f"; whole-recording score {af.score:.2f}")
            af.confidence = float(min(0.7, max(0.4, np.mean([e[2] for e in strong]) - 0.2)))
            af.warnings = list(af.warnings) + ["episode-based call — whole-recording score below threshold"]
    except Exception as e:  # noqa: BLE001
        warns.append(f"AF screen unavailable: {e}")
    return arr, af


def _merge_rhythm(base: RhythmAssessment, arr, af, hr_mean: float) -> RhythmAssessment:
    """Fold the arrhythmia / AF screens into the headline rhythm label.

    Precedence: AF (if the RR-based screen says so) > urgent findings (VT / SVT / long pause)
    > base label decorated with notable ectopy (e.g. "Sinus rhythm with frequent PVCs
    (bigeminy)"). Every finding label also goes into `flags` so nothing is lost."""
    label, conf, rationale = base.label, base.confidence, base.rationale
    flags = list(base.flags)
    findings = list(getattr(arr, "findings", []) or [])
    for fnd in findings:
        flags.append(str(fnd.label))
    codes = {getattr(f, "code", ""): f for f in findings}
    ectopy_notable = [f for f in findings if f.code in ("pvc", "pac", "bigeminy", "trigeminy", "couplet")
                      and f.severity in ("notable", "urgent") and float(getattr(f, "confidence", 0.0)) >= 0.5]

    def strong(f):   # only let a screen finding take over the headline when it is confident
        return f is not None and f.severity == "urgent" and float(getattr(f, "confidence", 0.0)) >= 0.5

    if af is not None and getattr(af, "is_af", False):
        label = getattr(af, "label", "") or "Possible atrial fibrillation"
        conf = float(getattr(af, "confidence", 0.5) or 0.5)
        rationale = getattr(af, "evidence", "") or rationale
        flags.append("irregularly irregular — NOT a diagnosis")
    elif strong(codes.get("vt")) or strong(codes.get("vt_run")):
        f = codes.get("vt") if strong(codes.get("vt")) else codes.get("vt_run")
        label = "Possible ventricular tachycardia"; conf = f.confidence; rationale = f.evidence
    elif strong(codes.get("svt")):
        f = codes["svt"]; label = "Possible supraventricular tachycardia"; conf = f.confidence; rationale = f.evidence
    elif ectopy_notable and label in ("Normal sinus rhythm", "Sinus bradycardia", "Sinus tachycardia",
                                     "Irregular rhythm", "Sinus arrhythmia"):
        # describe the dominant ectopy pattern
        pat = next((f for f in ectopy_notable if f.code in ("bigeminy", "trigeminy")), None)
        ect = next((f for f in ectopy_notable if f.code in ("pvc", "pac")), None) or ectopy_notable[0]
        what = "PVCs" if ect.code in ("pvc", "bigeminy", "couplet") and "ventric" in ect.label.lower() else (
               "PVCs" if ect.code == "pvc" else "PACs" if ect.code == "pac" else "ectopic beats")
        base_lbl = "Sinus rhythm" if label in ("Normal sinus rhythm", "Irregular rhythm") else label
        label = f"{base_lbl} with frequent {what}" + (f" ({pat.code})" if pat else "")
        conf = min(conf, max(0.4, ect.confidence))
        rationale = f"{rationale}; {ect.evidence}"
    elif label == "Irregular rhythm" and af is not None and not getattr(af, "is_af", False) \
            and any(f.code in ("pvc", "pac") for f in findings):
        label = "Sinus rhythm with ectopic beats"
        rationale = f"{rationale}; irregularity explained by premature beats"
    # AF screen result as a flag when it disagrees with the headline
    if af is not None and getattr(af, "label", "") == "Undetermined" and label.startswith("Possible atrial"):
        flags.append("AF screen undetermined (short/noisy)")
    # de-duplicate flags, keep order
    seen = set(); flags = [x for x in flags if not (x in seen or seen.add(x))]
    return RhythmAssessment(label, float(conf), rationale, flags)


# --------------------------------------------------------------------------- main
def analyze_recording(x, fs: float, lead_names: list[str], mains: float = 60.0) -> AnalysisResult:
    """Analyze a (n_samples, n_leads) mV recording. Never raises; degrades to nan + warnings."""
    warns: list[str] = []
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, nlead = x.shape
    lead_names = list(lead_names)[:nlead] or [f"ch{i}" for i in range(nlead)]
    duration = n / fs

    empty_iv = Intervals()
    empty = AnalysisResult(fs=fs, duration_s=duration, lead_used="", n_beats=0,
                           intervals=empty_iv, rhythm=RhythmAssessment(),
                           hrv={"SDNN": NAN, "RMSSD": NAN, "pNN50": NAN},
                           qrs_axis_deg=NAN, axis_label="Undetermined",
                           median_beat={}, median_beat_fs=fs, fiducials={},
                           quality_sqi=0.0, warnings=warns)

    if n < int(2 * fs):
        warns.append("recording too short to analyze (<2 s)")
        return empty

    # primary lead
    if "II" in lead_names:
        lead_used = "II"
    elif "I" in lead_names:
        lead_used = "I"
    else:
        lead_used = lead_names[0]
    li = lead_names.index(lead_used)
    primary = x[:, li]

    q = dsp.signal_quality(primary, fs)
    sqi = q.sqi
    if q.flat:
        warns.append("primary lead is flat")
        empty.quality_sqi = sqi
        empty.lead_used = lead_used
        empty.rhythm = RhythmAssessment("Too noisy to classify", 0.0, "flat signal", ["flat signal"])
        return empty

    clean_primary = dsp.clean(primary, fs, mains)
    rpeaks = dsp.detect_r_peaks(clean_primary, fs)
    n_beats = int(len(rpeaks))
    if n_beats < 3:
        warns.append(f"only {n_beats} beats detected; measurements unreliable")
        empty.quality_sqi = sqi
        empty.lead_used = lead_used
        empty.n_beats = n_beats
        empty.rhythm = _classify_rhythm(dsp.heart_rate_from_peaks(rpeaks, fs),
                                        np.diff(rpeaks) / fs * 1000.0 if n_beats > 1 else np.array([]),
                                        sqi, False, NAN)
        return empty

    rr_ms = np.diff(rpeaks) / fs * 1000.0
    rr_ok, _ = _robust_rr(rr_ms)               # exclude missed-beat/doublet artifacts
    rr_mean = float(np.mean(rr_ok)) if len(rr_ok) else NAN
    rr_sd = float(np.std(rr_ok, ddof=1)) if len(rr_ok) > 1 else NAN
    hr_series = 60000.0 / rr_ok if len(rr_ok) else np.array([])
    hr_mean = float(np.median(hr_series)) if len(hr_series) else NAN
    hr_min = float(np.min(hr_series)) if len(hr_series) else NAN
    hr_max = float(np.max(hr_series)) if len(hr_series) else NAN
    rr_med_samp = int(round(np.median(np.diff(rpeaks))))

    # median beats for every lead
    median_beats: dict[str, np.ndarray] = {}
    r_idx: dict[str, int] = {}
    for j, name in enumerate(lead_names):
        cb = dsp.clean(x[:, j], fs, mains)
        mb, r = _median_beat(cb, rpeaks, fs, rr_samp=rr_med_samp)
        if mb is not None:
            median_beats[name] = mb
            r_idx[name] = r

    # delineate the primary median beat -> intervals
    iv = Intervals(rr_mean_ms=rr_mean, rr_sd_ms=rr_sd, hr_mean=hr_mean,
                   hr_min=hr_min, hr_max=hr_max, measured_on=lead_used)
    fiducials: dict[str, float] = {}
    p_consistent = False
    if lead_used in median_beats:
        beat = median_beats[lead_used]
        r = r_idx[lead_used]
        d = _delineate_median(beat, r, fs)
        if d:
            Pon, Poff = d.get("ECG_P_Onsets", NAN), d.get("ECG_P_Offsets", NAN)
            Ron, Roff = d.get("ECG_R_Onsets", NAN), d.get("ECG_R_Offsets", NAN)
            Ton, Toff = d.get("ECG_T_Onsets", NAN), d.get("ECG_T_Offsets", NAN)
            iv.p_dur_ms = _safe_ms(Pon, Poff, fs, 40, 200)
            iv.pr_ms = _safe_ms(Pon, Ron, fs, 80, 320)
            iv.qrs_ms = _safe_ms(Ron, Roff, fs, 40, 200)
            iv.qt_ms = _safe_ms(Ron, Toff, fs, 200, 700)
            iv.t_dur_ms = _safe_ms(Ton, Toff, fs, 80, 400)
            p_consistent = np.isfinite(Pon) and np.isfinite(iv.pr_ms)
            rr_s = rr_mean / 1000.0 if np.isfinite(rr_mean) else NAN
            if np.isfinite(iv.qt_ms) and np.isfinite(rr_s) and rr_s > 0:
                iv.qtc_bazett_ms = float(iv.qt_ms / np.sqrt(rr_s))
                iv.qtc_fridericia_ms = float(iv.qt_ms / (rr_s ** (1.0 / 3.0)))
            # ST level at J (=R offset) + 60 ms, relative to PR-segment isoelectric
            if np.isfinite(Roff):
                j_pt = int(round(Roff + 0.06 * fs))
                iso = NAN
                if np.isfinite(Pon) and np.isfinite(Ron):
                    iso = float(np.median(beat[int(Poff if np.isfinite(Poff) else Pon):int(Ron) + 1])) \
                        if int(Ron) > int(Pon) else NAN
                if not np.isfinite(iso):
                    iso = float(np.median(beat[:max(1, int(0.05 * fs))]))
                if 0 <= j_pt < len(beat):
                    iv.st_level_mv = float(beat[j_pt] - iso)
            fiducials = {k: (float(v) if np.isfinite(v) else NAN) for k, v in d.items()}
            fiducials["ECG_R_Peaks"] = float(r)
        else:
            warns.append("delineation failed; interval measurements unavailable")
    else:
        warns.append("could not build a median beat on the primary lead")

    # axis
    axis, axis_label = _qrs_axis(median_beats, r_idx, fs)

    # HRV
    hrv = _hrv(rpeaks, fs)

    # rhythm (base heuristic), then the arrhythmia + AF screens folded into it
    rhythm = _classify_rhythm(hr_mean, rr_ok, sqi, p_consistent, hrv.get("RMSSD", NAN))
    arr, af = _screen_arrhythmias(clean_primary, fs, rpeaks, median_beats.get(lead_used),
                                  r_idx.get(lead_used, 0), iv, p_consistent, sqi, warns)
    rhythm = _merge_rhythm(rhythm, arr, af, hr_mean)
    if not q.readable:
        rhythm.flags = list(dict.fromkeys(rhythm.flags + ["low signal quality"]))

    return AnalysisResult(
        fs=fs, duration_s=duration, lead_used=lead_used, n_beats=n_beats,
        intervals=iv, rhythm=rhythm, hrv=hrv, qrs_axis_deg=axis, axis_label=axis_label,
        median_beat=median_beats, median_beat_fs=fs, fiducials=fiducials,
        quality_sqi=sqi, warnings=warns, arrhythmia=arr, af=af,
    )
