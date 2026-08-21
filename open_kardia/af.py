"""Atrial-fibrillation (AF) screening from the RR-interval series.

RESEARCH SIGNAL — NOT A MEDICAL DEVICE. This is an automated, heuristic estimate on a
consumer single-/six-lead recording. It is not clinically validated and must not be used
for diagnosis; "Possible atrial fibrillation" means "the RR series looks irregularly
irregular", nothing more.

Pure numpy; never raises — bad input degrades to an 'Undetermined' AFResult with warnings.

Method (all on RR intervals in ms; optionally P-wave evidence from the caller):
  * COSEn — coefficient of sample entropy (Lake & Moorman 2011, Am J Physiol Heart Circ
    Physiol 300:H319): SampEn(m=1, r=30 ms) + ln(2r) − ln(mean RR).  Adding ln(2r) turns
    the conditional-probability estimate into a density (so r cancels), subtracting
    ln(mean RR) removes rate dependence; the sum is unit-free (only r/mean RR matters), so
    it is the same whether RR is in ms or s.  Lake & Moorman quote an AF cut-off of about
    −1.4 (Carrara et al. 2015 use −1.4 .. −1.0).  We compute it on the classic 12-beat
    window, sliding one beat at a time over the recording, and take the MEDIAN.  When a
    window has no template matches at all (B = 0) SampEn is capped at ln(#pairs); when
    only A = 0 we use A = 0.5 (Lake's "minimum count" idea) instead of infinity.
  * Normalized RMSSD = RMSSD / mean RR (Dash et al. 2009, Ann Biomed Eng 37:1701 use
    nRMSSD > ~0.1 for AF), RR coefficient of variation, Poincaré SD1/SD2, fraction of
    |ΔRR| > 50 ms (pNN50) and > 8 % of mean RR (rate-normalized "pNN8%").
  * Turning-point ratio TPR (Dash 2009): fraction of interior RRs that are local extrema.
    An i.i.d. sequence gives 2/3 (SD √((16n−29)/90)/(n−2)); AF sits in that band, smooth
    sinus arrhythmia sits below (~0.3–0.5), strict alternation (bigeminy) at ~1.0.
  * Lag-1 autocorrelation of RR: ~0 in AF, strongly positive for respiratory sinus
    arrhythmia, strongly negative for bigeminy.
  * Ectopy robustness, two layers:
      1. an ectopy filter (Dash-style): every short→long RR pair (RR_i < 0.88·median and
         RR_i+1 > 1.08·median — a premature beat plus its pause) and every isolated
         RR > 1.6·median (missed beat / pause) is removed; the RESIDUAL series is scored
         again (resid_cosen, resid_nrmssd, resid_dev_frac = fraction of residual RRs
         > 10 % from their median).  Sinus + PVCs (any frequency, couplets included)
         leaves a regular residual; AF stays irregular after the filter.
      2. a "regularly irregular" guard: bigeminy/trigeminy give a bimodal RR histogram
         (two tight clusters) plus a periodic pattern (positive autocorrelation at lag 2
         or 3, TPR ≈ 1).  If that fires the AF score is capped at 0.3 and the evidence
         string says so.
  * RRs < 300 ms or > 2500 ms (missed / doubled detections) are dropped but counted.
  * Score: a hand-tuned logistic combination of the standardized features (see af_score),
    shifted by P-wave evidence (consistent P waves argue against AF; absent P waves for
    it).  0.5 is the decision threshold.  Weights were tuned on MIT-BIH Arrhythmia
    records with validate/mitdb_af.py (30 s windows, end-to-end through dsp.detect_r_peaks):
    AF records 201/202/203/210/219/221/222 + controls 100/101/103/106/119/208/233 gave
    Se 92 %, Sp 88 % pooled (Sp 91 % without atrial-flutter windows, 94 % also without
    record 222 whose non-AF stretches are frequent APCs / SVTA runs); MIT-BIH AFDB
    (first 30 min of 23 records) Se 96 %, Sp 99 %.  Remaining errors are borderline AF
    (COSEn −1.0…−1.4, slow/organized AF) and very PVC-rich sinus rhythm (208).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NAN = float("nan")

MIN_RR = 12              # fewer RRs than this -> Undetermined
COSEN_WINDOW = 12        # Lake & Moorman's window length (beats)
COSEN_R_MS = 30.0        # SampEn tolerance in ms
COSEN_THRESHOLD = -1.4   # literature cut-off; the score uses it as the zero point
RR_LO_MS, RR_HI_MS = 300.0, 2500.0   # outside -> detection artifact, dropped
AF_THRESHOLD = 0.5


@dataclass
class AFResult:
    score: float          # 0..1 probability-like AF score (0.5 = decision threshold)
    is_af: bool           # score >= threshold AND enough beats AND quality ok
    confidence: float     # 0..1 how sure (more beats, cleaner RR, absent P -> higher)
    label: str            # 'Possible atrial fibrillation' | 'AF not suggested' | 'Undetermined'
    evidence: str         # one line with the numbers
    metrics: dict         # JSON-safe feature dict (see rr_features)
    warnings: list = field(default_factory=list)
    episodes: list = field(default_factory=list)   # [(t_start, t_end, mean_score), ...] from af_episodes()


# --------------------------------------------------------------------------- helpers
def _f(x) -> float:
    """float() that never raises and maps None/inf to nan (JSON-safe via allow_nan)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


def _sample_entropy_m1(x: np.ndarray, r: float) -> float:
    """SampEn(m=1) of a short series: −ln(A/B) with B = #pairs matching at length 1 and
    A = #pairs also matching at length 2 (i < j, no self-matches). Capped/floored so it
    is always finite (see module docstring)."""
    n = len(x)
    if n < 3:
        return NAN
    d = np.abs(x[:, None] - x[None, :]) <= r
    iu = np.triu_indices(n - 1, k=1)          # templates of length 1 exist for i < n-1
    m1 = d[:-1, :-1][iu]
    m2 = m1 & d[1:, 1:][iu]
    B, A = int(np.sum(m1)), int(np.sum(m2))
    if B == 0:
        return float(np.log(len(iu[0])))      # nothing matches even at m=1: maximal
    return float(-np.log(max(A, 0.5) / B))


def _cosen(rr: np.ndarray, window: int = COSEN_WINDOW, r: float = COSEN_R_MS) -> tuple[float, float]:
    """(median COSEn, median SampEn) over sliding `window`-beat windows."""
    n = len(rr)
    if n < 3:
        return NAN, NAN
    if n <= window:
        starts = [0]
        window = n
    else:
        starts = range(0, n - window + 1)
    se, ce = [], []
    for s in starts:
        w = rr[s:s + window]
        e = _sample_entropy_m1(w, r)
        if np.isfinite(e):
            se.append(e)
            ce.append(e + np.log(2 * r) - np.log(np.mean(w)))
    if not se:
        return NAN, NAN
    return float(np.median(ce)), float(np.median(se))


def _turning_point_ratio(rr: np.ndarray) -> float:
    if len(rr) < 3:
        return NAN
    d = np.diff(rr)
    tp = (d[:-1] * d[1:]) < 0                 # strict local max/min
    return float(np.mean(tp))


def _acf(rr: np.ndarray, lag: int) -> float:
    if len(rr) <= lag + 2:
        return NAN
    x = rr - np.mean(rr)
    den = float(np.sum(x * x))
    if den <= 0:
        return NAN
    return float(np.sum(x[:-lag] * x[lag:]) / den)


def _bimodality(rr: np.ndarray) -> tuple[float, float, float]:
    """Two-cluster split of the RR histogram (1-D 2-means started at the median).
    Returns (separation / mean RR, worst within-cluster CV, minority fraction).
    Bigeminy: separation ~0.4, within-CV ~0.02, minority ~0.5; AF: within-CV large."""
    if len(rr) < 6:
        return NAN, NAN, NAN
    thr = float(np.median(rr))
    for _ in range(10):
        lo, hi = rr[rr <= thr], rr[rr > thr]
        if len(lo) == 0 or len(hi) == 0:
            break
        new = 0.5 * (np.mean(lo) + np.mean(hi))
        if abs(new - thr) < 1e-6:
            break
        thr = new
    lo, hi = rr[rr <= thr], rr[rr > thr]
    if len(lo) < 2 or len(hi) < 2:
        return 0.0, NAN, 0.0
    sep = float((np.mean(hi) - np.mean(lo)) / np.mean(rr))
    wcv = float(max(np.std(lo) / np.mean(lo), np.std(hi) / np.mean(hi)))
    frac = float(min(len(lo), len(hi)) / len(rr))
    return sep, wcv, frac


def _ectopy_filter(rr: np.ndarray, lo: float = 0.88, hi: float = 1.08,
                   long: float = 1.6) -> tuple[np.ndarray, int, int]:
    """Remove premature-beat pairs (short RR followed by long RR) and isolated long RRs.
    Returns (residual_rr, n_pairs, n_long)."""
    n = len(rr)
    if n < 3:
        return rr, 0, 0
    med = float(np.median(rr))
    pair = (rr[:-1] < lo * med) & (rr[1:] > hi * med)
    idx = np.flatnonzero(pair)
    drop = np.zeros(n, dtype=bool)
    drop[idx] = True
    drop[idx + 1] = True
    # extend backwards over runs of short RRs ending in a pause (couplets / salvos)
    for i in range(n - 2, -1, -1):
        if drop[i + 1] and not drop[i] and rr[i] < lo * med:
            drop[i] = True
    long_mask = (rr > long * med) & ~drop
    return rr[~(drop | long_mask)], int(len(idx)), int(np.sum(long_mask))


# --------------------------------------------------------------------------- features
def rr_features(rr_ms) -> dict:
    """All RR-series metrics used by the AF detector, as a JSON-safe dict (floats/ints/
    bools; nan where not computable). Drops non-finite and out-of-range RRs first."""
    out = {"n_rr_raw": 0, "n_rr": 0, "n_rr_dropped": 0}
    try:
        rr = np.asarray(rr_ms, dtype=float).ravel()
    except (TypeError, ValueError):
        rr = np.array([], dtype=float)
    rr = rr[np.isfinite(rr)]
    out["n_rr_raw"] = int(len(rr))
    keep = (rr >= RR_LO_MS) & (rr <= RR_HI_MS)
    out["n_rr_dropped"] = int(np.sum(~keep))
    rr = rr[keep]
    n = int(len(rr))
    out["n_rr"] = n
    keys = ("mean_rr_ms", "median_rr_ms", "hr_bpm", "rr_sd_ms", "rr_cv", "rmssd_ms", "nrmssd",
            "pnn50", "pnn8pct", "tpr", "tpr_z", "acf1", "acf2", "acf3", "poincare_sd1_ms",
            "poincare_sd2_ms", "poincare_sd1_sd2", "sampen", "cosen", "bimodal_sep",
            "bimodal_within_cv", "bimodal_minority_frac", "resid_cosen", "resid_nrmssd",
            "resid_dev_frac")
    for k in keys:
        out[k] = NAN
    out["regularly_irregular"] = False
    out["ectopic_pairs"] = 0
    out["long_rr_dropped"] = 0
    out["resid_n"] = 0
    if n < 3:
        return out
    mean = float(np.mean(rr))
    d = np.diff(rr)
    out["mean_rr_ms"] = mean
    out["median_rr_ms"] = float(np.median(rr))
    out["hr_bpm"] = 60000.0 / mean
    out["rr_sd_ms"] = float(np.std(rr, ddof=1))
    out["rr_cv"] = out["rr_sd_ms"] / mean
    rmssd = float(np.sqrt(np.mean(d ** 2)))
    out["rmssd_ms"] = rmssd
    out["nrmssd"] = rmssd / mean
    out["pnn50"] = float(np.mean(np.abs(d) > 50.0))
    out["pnn8pct"] = float(np.mean(np.abs(d) > 0.08 * mean))
    tpr = _turning_point_ratio(rr)
    out["tpr"] = tpr
    # z-score of TPR against the i.i.d. expectation 2(n-2)/3, var (16n-29)/90
    if n >= 4 and np.isfinite(tpr):
        exp_tp, var_tp = 2.0 * (n - 2) / 3.0, (16.0 * n - 29.0) / 90.0
        out["tpr_z"] = float((tpr * (n - 2) - exp_tp) / math.sqrt(var_tp))
    out["acf1"], out["acf2"], out["acf3"] = _acf(rr, 1), _acf(rr, 2), _acf(rr, 3)
    sd1 = float(np.std(d, ddof=0) / math.sqrt(2.0))
    sd2 = float(math.sqrt(max(2.0 * np.var(rr) - 0.5 * np.var(d), 0.0)))
    out["poincare_sd1_ms"], out["poincare_sd2_ms"] = sd1, sd2
    out["poincare_sd1_sd2"] = sd1 / sd2 if sd2 > 0 else NAN
    out["cosen"], out["sampen"] = _cosen(rr)
    sep, wcv, frac = _bimodality(rr)
    out["bimodal_sep"], out["bimodal_within_cv"], out["bimodal_minority_frac"] = sep, wcv, frac
    # "Regularly irregular" (bigeminy/trigeminy-like): two tight, well-separated RR
    # clusters with a sizeable minority AND a periodic pattern (lag-2/3 autocorrelation
    # or a near-1 turning-point ratio).  AF has a broad, unimodal-ish histogram and ~0 ACF.
    acf23 = max([a for a in (out["acf2"], out["acf3"]) if np.isfinite(a)], default=NAN)
    periodic = (np.isfinite(acf23) and acf23 > 0.35) or (np.isfinite(tpr) and tpr > 0.85)
    out["regularly_irregular"] = bool(np.isfinite(sep) and sep > 0.12 and wcv < 0.07
                                      and frac > 0.2 and periodic)
    # ectopy-filtered residual series (see module docstring)
    res, n_pairs, n_long = _ectopy_filter(rr)
    out["ectopic_pairs"], out["long_rr_dropped"], out["resid_n"] = n_pairs, n_long, int(len(res))
    if len(res) >= 8:
        rmed = float(np.median(res))
        rd = np.diff(res)
        out["resid_cosen"] = _cosen(res)[0]
        out["resid_nrmssd"] = float(np.sqrt(np.mean(rd ** 2)) / np.mean(res))
        out["resid_dev_frac"] = float(np.mean(np.abs(res - rmed) > 0.10 * rmed))
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


# --------------------------------------------------------------------------- score
def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z)))))


def af_score(rr_ms, p_consistent: bool | None = None, sqi: float = 1.0) -> AFResult:
    """Score an RR series (ms) for AF-like irregular irregularity.

    Logit  z = −1.5 + 1.0·(COSEn + 1.4)/0.35                       (primary: Lake & Moorman)
                    + 0.3·(resid COSEn + 1.4)/0.35 + 0.3·(resid dev frac − 0.25)/0.10
                    + 0.2·(resid nRMSSD − 0.12)/0.05               (ectopy-filtered series)
                    − 1.0·max(0, ACF1 − 0.3)/0.2                   (smooth sinus arrhythmia)
                    − 1.5·max(0, −ACF1 − 0.3)/0.2                  (short-long alternation = ectopy)
                    − 0.8·max(0, |TPR − 2/3| − 0.15)/0.1           (not random-like)
    each term clipped to ±3; missing terms contribute 0.  Then z −= 1.5 if p_consistent
    is True, z += 0.6 if False; score = sigmoid(z).  A "regularly irregular" pattern
    (see rr_features) caps the score at 0.3.  COSEn = −1.4 with a regular residual gives
    z ≈ −3 (score ≈ 0.05); COSEn ≈ −1.2 with an AF-like residual crosses 0.5.  Weights
    were hand-tuned on MIT-BIH (validate/mitdb_af.py; see module docstring) so that plain
    sinus, sinus arrhythmia and PVC bigeminy/trigeminy fall below 0.5 and AF above.
    nRMSSD, CV, pNN50, Poincaré SD1/SD2 are reported in `metrics` but carry no weight:
    they are dominated by ectopy and added nothing on MIT-BIH once COSEn was in.

    `sqi` < 0.5 → 'Undetermined' unless the score is overwhelming (≥ 0.85); < 0.3 always
    Undetermined.  Never raises.
    """
    warns: list[str] = []
    m = rr_features(rr_ms)
    sqi = _f(sqi)
    if not np.isfinite(sqi):
        sqi = 1.0
    if m["n_rr_dropped"]:
        warns.append(f"{m['n_rr_dropped']} RR interval(s) outside {RR_LO_MS:.0f}–{RR_HI_MS:.0f} ms dropped "
                     "(likely missed/double detections)")
    n = m["n_rr"]

    def _undetermined(reason: str, score: float = NAN) -> AFResult:
        warns.append(reason)
        m["af_evidence_score"] = _f(score)
        return AFResult(_f(score), False, 0.0, "Undetermined", reason, m, warns)

    if n < MIN_RR:
        return _undetermined(f"too few RR intervals ({n} < {MIN_RR}) for AF scoring")
    if n < 20:
        warns.append(f"only {n} RR intervals; AF score is unreliable below ~20 beats")

    def term(v, center, scale, w):
        return w * float(np.clip((v - center) / scale, -3.0, 3.0)) if np.isfinite(v) else 0.0

    z = -1.5
    z += term(m["cosen"], COSEN_THRESHOLD, 0.35, 1.0)
    # residual (ectopy-filtered) terms; if the filter ate almost everything (bigeminy)
    # they are nan and contribute nothing — the regularly-irregular guard handles that
    z += term(m["resid_cosen"], COSEN_THRESHOLD, 0.35, 0.3)
    z += term(m["resid_dev_frac"], 0.25, 0.10, 0.3)
    z += term(m["resid_nrmssd"], 0.12, 0.05, 0.2)
    if np.isfinite(m["acf1"]):
        z -= 1.0 * min(3.0, max(0.0, m["acf1"] - 0.3) / 0.2)     # smooth modulation
        z -= 1.5 * min(3.0, max(0.0, -m["acf1"] - 0.3) / 0.2)    # short-long alternation
    if np.isfinite(m["tpr"]):                        # not "random-like" turning points
        z -= 0.8 * min(3.0, max(0.0, abs(m["tpr"] - 2.0 / 3.0) - 0.15) / 0.1)
    z_rr = z
    if p_consistent is True:
        z -= 1.5
    elif p_consistent is False:
        z += 0.6
    score = _sigmoid(z)
    m["af_evidence_score"] = _f(_sigmoid(z_rr))      # RR-only score, before P evidence

    pattern = bool(m.get("regularly_irregular"))
    if pattern:
        score = min(score, 0.3)
        warns.append("RR pattern is regularly irregular (bimodal, periodic) — ectopy pattern rather than AF")

    ev = (f"COSEn {m['cosen']:.2f}, nRMSSD {m['nrmssd']:.2f}, TPR {m['tpr']:.2f}, "
          f"RR CV {100 * m['rr_cv']:.0f}%")
    if pattern:
        ev = "RR regularly irregular (bimodal / bigeminy-trigeminy-like pattern): " + ev
    elif score >= AF_THRESHOLD:
        ev = "RR irregularly irregular: " + ev
    else:
        ev = "RR not irregularly irregular: " + ev
    if p_consistent is True:
        ev += "; consistent P waves"
    elif p_consistent is False:
        ev += "; P waves not detected"

    # confidence: distance from threshold, shrunk for few beats / low SQI / conflicting P
    conf = min(1.0, abs(score - AF_THRESHOLD) * 2.0)
    conf *= min(1.0, n / 30.0)
    conf *= float(np.clip(sqi, 0.0, 1.0))
    if p_consistent is not None:
        p_says_af = not p_consistent
        conf *= 1.1 if p_says_af == (score >= AF_THRESHOLD) else 0.7
    conf = float(np.clip(conf, 0.0, 1.0))

    if sqi < 0.3 or (sqi < 0.5 and score < 0.85):
        warns.append(f"low signal quality (SQI {sqi:.2f}); AF call withheld")
        return AFResult(_f(score), False, conf * 0.5, "Undetermined", ev, m, warns)
    if sqi < 0.5:
        warns.append(f"low signal quality (SQI {sqi:.2f}) but RR irregularity is strong")

    is_af = bool(score >= AF_THRESHOLD)
    label = "Possible atrial fibrillation" if is_af else "AF not suggested"
    if is_af:
        warns.append("irregularly irregular RR — NOT a diagnosis")
    return AFResult(_f(score), is_af, conf, label, ev, m, warns)


def af_episodes(rr_ms, r_times, window_beats: int = 16, step_beats: int = 2,
                threshold: float = AF_THRESHOLD, p_consistent=None, sqi: float = 1.0) -> list:
    """Localize AF-like stretches: score sliding windows of `window_beats` RR intervals with
    `af_score` and merge consecutive windows at/above `threshold` into episodes.
    `r_times`: seconds of the R-peaks (len(rr_ms) + 1). Returns [(t_start, t_end, mean_score)].
    Never raises. Short windows are noisier than the whole-recording score, so treat these as
    the *where*, not as a second detector."""
    try:
        rr = np.asarray(rr_ms, dtype=float).ravel()
        tr = np.asarray(r_times, dtype=float).ravel()
        if len(rr) < window_beats or len(tr) != len(rr) + 1:
            return []
        wins = []
        for i in range(0, len(rr) - window_beats + 1, step_beats):
            res = af_score(rr[i:i + window_beats], p_consistent=p_consistent, sqi=sqi)
            sc = res.score if res.score == res.score else 0.0
            wins.append((float(tr[i]), float(tr[i + window_beats]), float(sc)))
        # hysteresis: an episode starts at >= threshold and continues while >= 0.7*threshold
        # (a single organized stretch inside AF should not split the episode in two)
        exit_thr = 0.7 * threshold
        eps = []
        cur = None
        for a, b, sc in wins:
            if (cur is None and sc >= threshold) or (cur is not None and sc >= exit_thr):
                if cur is None:
                    cur = [a, b, [sc]]
                else:
                    cur[1] = b; cur[2].append(sc)
            elif cur is not None:
                eps.append(cur)
                cur = None
        if cur is not None:
            eps.append(cur)
        # a window flags AF when most of it is AF, so its edges over-reach by up to ~1/2 window;
        # trim a quarter window at each end (bounded so the span never inverts)
        wlen = float(np.median([b - a for a, b, _ in wins])) if wins else 0.0
        # merge episodes separated by less than one window (a brief organized-looking stretch)
        merged = []
        for a, b, scores in eps:
            if merged and a - merged[-1][1] < wlen:
                merged[-1][1] = b; merged[-1][2] = merged[-1][2] + scores
            else:
                merged.append([a, b, list(scores)])
        out = []
        for a, b, scores in merged:
            trim = min(0.25 * wlen, 0.45 * (b - a))
            out.append((round(a + trim, 3), round(b - trim, 3), round(float(np.mean(scores)), 3)))
        return out
    except Exception:  # noqa: BLE001
        return []
