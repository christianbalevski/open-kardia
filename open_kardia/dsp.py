"""ECG signal processing: filters, R-peak detection, heart rate, signal quality.

All functions are pure numpy/scipy; nothing here touches Bluetooth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


# --------------------------------------------------------------------------- filters
def bandpass(x, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass (default 0.5–40 Hz, the usual monitor band)."""
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x)


def notch(x, fs: float, f0: float = 60.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch at mains frequency (60 Hz US; use 50 for EU)."""
    x = np.asarray(x, dtype=np.float64)
    if f0 >= fs / 2.0:
        return x
    b, a = signal.iirnotch(f0, q, fs)
    return signal.filtfilt(b, a, x)


def remove_baseline(x, fs: float, cutoff: float = 0.5, order: int = 2) -> np.ndarray:
    """High-pass to remove baseline wander."""
    x = np.asarray(x, dtype=np.float64)
    sos = signal.butter(order, cutoff / (fs / 2.0), btype="high", output="sos")
    return signal.sosfiltfilt(sos, x)


def clean(x, fs: float, mains: float = 60.0) -> np.ndarray:
    """Standard cleaning chain: bandpass 0.5–40 Hz then mains notch."""
    return notch(bandpass(x, fs), fs, mains)


# ------------------------------------------------------ edge-stable display filters
# `clean`/`bandpass`/`notch` use zero-phase filtfilt/sosfiltfilt. On a SHORT slice
# (e.g. the ~6 s live-plot window) filtfilt's default edge handling leaves a visible
# transient ("shaky" ringing) at both ends of the shown window. The two functions below
# remove that for display:
#   * clean_stable  -- pads the ends by reflection before filtering, then trims, so the
#                      returned window is transient-free at its own edges.
#   * display_filter-- the GUI helper: filter the WHOLE ring buffer, then return only the
#                      last `window_s` seconds, so what's shown is an interior slice of a
#                      longer filtered signal and its edges are inherently clean.
# These are for visualization only; analysis code should keep using `clean`.
def clean_stable(x, fs: float, mains: float = 60.0, pad_s: float = 1.0) -> np.ndarray:
    """Zero-phase 0.5–40 Hz bandpass + mains notch with a LONGER settling pad than the
    filters' defaults, so the edge transient is pushed further out of the returned window.

    Same passband and length as `clean`, but each filtfilt/sosfiltfilt stage uses scipy's
    odd end-extension at `pad_s` seconds (clamped to < len) instead of the default short
    padlen. On its own this only softens the edges of a bare slice; for a truly clean
    displayed window use `display_filter`, which filters real neighbouring samples.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return x.copy()
    if x.ndim == 2:
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            out[:, c] = clean_stable(x[:, c], fs, mains, pad_s)
        return out
    pad = int(round(pad_s * fs))
    pad = max(0, min(pad, n - 1))          # scipy requires padlen < n
    nyq = fs / 2.0
    hi = min(40.0, 0.95 * nyq)
    sos = signal.butter(4, [0.5 / nyq, hi / nyq], btype="band", output="sos")
    # 'constant' end-extension (hold the endpoint value) over a long pad: for a baseline-
    # wandering ECG slice this is a better neighbour guess for the low-frequency content
    # than the default odd-reflection, which invents an opposite-going trend the 0.5 Hz
    # high-pass edge then rings on. Empirically ~2x less edge transient than plain clean.
    pl = pad if pad > 0 else None
    y = signal.sosfiltfilt(sos, x, padtype="constant", padlen=pl)
    if mains < nyq:
        b, a = signal.iirnotch(mains, 30.0, fs)
        y = signal.filtfilt(b, a, y, padtype="constant", padlen=pl)
    return y


def display_filter(buffer, fs: float, mains: float = 60.0, window_s: float | None = None,
                   pad_s: float = 1.0) -> np.ndarray:
    """GUI display helper. Filter the ENTIRE `buffer` (longer than what's shown, e.g. the
    30 s ring) with `clean_stable`, then, if `window_s` is given, return only the last
    `window_s` seconds — so the visible window is an interior slice of a longer filtered
    signal and its edges have no filter transient.

    Accepts 1D `(n,)` or 2D `(n, ch)` and returns the same rank. If `window_s` is None the
    whole filtered buffer is returned.
    """
    buf = np.asarray(buffer, dtype=np.float64)
    filtered = clean_stable(buf, fs, mains, pad_s)
    if window_s is None:
        return filtered
    w = int(round(window_s * fs))
    if w <= 0 or w >= filtered.shape[0]:
        return filtered
    return filtered[-w:] if filtered.ndim == 1 else filtered[-w:, :]


# --------------------------------------------------------------------------- R peaks
def detect_r_peaks(x, fs: float, refractory_s: float = 0.25) -> np.ndarray:
    """Pan–Tompkins-style R-peak detector.

    Returns sample indices of detected R peaks in `x` (raw or cleaned).
    Steps: bandpass 5–15 Hz -> derivative -> square -> moving-window integrate ->
    distance-limited candidate peaks -> adaptive threshold with search-back ->
    refine each detection to the local max of the bandpassed signal.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < int(2 * fs):
        return np.array([], dtype=int)

    nyq = fs / 2.0
    sos = signal.butter(3, [5.0 / nyq, min(15.0, 0.9 * nyq) / nyq], btype="band", output="sos")
    bp = signal.sosfiltfilt(sos, x)
    deriv = np.gradient(bp)
    sq = deriv**2
    win = max(1, int(0.150 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")

    refractory = max(1, int(refractory_s * fs))
    cand, _ = signal.find_peaks(integ, distance=refractory)
    if len(cand) == 0:
        return np.array([], dtype=int)
    h = integ[cand]

    # Initialise from the first 2 s (or all if shorter).
    init = h[cand < 2 * fs] if np.any(cand < 2 * fs) else h
    spki = float(np.max(init))
    npki = float(np.median(init))
    thr = npki + 0.25 * (spki - npki)

    accepted: list[int] = []
    rr_hist: list[float] = []
    for k, (c, v) in enumerate(zip(cand, h)):
        if v > thr:
            # search-back: did we skip a beat between the last accepted and here?
            if accepted and len(rr_hist) >= 2:
                rr_avg = float(np.mean(rr_hist[-8:]))
                gap = c - accepted[-1]
                if gap > 1.66 * rr_avg:
                    mask = (cand > accepted[-1] + refractory) & (cand < c - refractory)
                    if np.any(mask):
                        j = np.argmax(h[mask])
                        cb, vb = cand[mask][j], h[mask][j]
                        if vb > 0.5 * thr:
                            accepted.append(int(cb))
                            rr_hist.append(cb - accepted[-2] if len(accepted) > 1 else gap)
            if accepted:
                rr_hist.append(float(c - accepted[-1]))
            accepted.append(int(c))
            spki = 0.125 * v + 0.875 * spki
        else:
            npki = 0.125 * v + 0.875 * npki
        thr = npki + 0.25 * (spki - npki)

    # Refine to local maximum of the bandpassed ECG within ±100 ms.
    w = max(1, int(0.1 * fs))
    out = []
    for c in accepted:
        lo, hi = max(0, c - w), min(n, c + w + 1)
        out.append(lo + int(np.argmax(bp[lo:hi])))
    return np.array(sorted(set(out)), dtype=int)


# --------------------------------------------------------------------------- HR
def rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    return np.diff(np.asarray(peaks, dtype=np.float64)) / fs


def heart_rate_from_peaks(peaks: np.ndarray, fs: float, window_s: float | None = None,
                          n_samples: int | None = None) -> float:
    """HR (bpm) = 60 / median RR. If window_s given, only RRs ending in the last
    window_s seconds (relative to n_samples) are used — mirrors the IFU's
    'median of last 5 s' style estimate."""
    peaks = np.asarray(peaks)
    if len(peaks) < 2:
        return float("nan")
    if window_s is not None and n_samples is not None:
        cutoff = n_samples - window_s * fs
        peaks = peaks[peaks >= cutoff]
        if len(peaks) < 2:
            return float("nan")
    rr = rr_intervals(peaks, fs)
    rr = rr[(rr > 0.25) & (rr < 3.0)]
    if len(rr) == 0:
        return float("nan")
    return float(60.0 / np.median(rr))


def heart_rate(x, fs: float, window_s: float | None = None) -> float:
    peaks = detect_r_peaks(x, fs)
    return heart_rate_from_peaks(peaks, fs, window_s, len(x))


# --------------------------------------------------------------------------- quality
@dataclass
class QualityReport:
    sqi: float          # 0..1, higher is better
    readable: bool
    hr_bpm: float
    n_peaks: int
    rr_cv: float        # coefficient of variation of RR
    noise_ratio: float  # HF power / total power
    flat: bool
    saturated: bool


def signal_quality(x, fs: float, hf_cutoff: float = 40.0, sat_frac: float = 0.98,
                   full_scale: float | None = None) -> QualityReport:
    """Heuristic signal-quality index approximating the app's 'Unreadable' gate.

    Components:
      * flatline (very low variance) -> unreadable
      * saturation (samples pinned near full scale) -> unreadable
      * out-of-band power ratio (>40 Hz vs total)
      * R-peak regularity (RR CV) and physiologic HR
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < int(2 * fs):
        return QualityReport(0.0, False, float("nan"), 0, float("nan"), float("nan"), True, False)

    flat = np.std(x) < 1e-6 or np.ptp(x) < 1e-5
    saturated = False
    if full_scale is not None:
        saturated = np.mean(np.abs(x) > sat_frac * full_scale) > 0.05

    f, p = signal.welch(x - np.mean(x), fs=fs, nperseg=min(n, int(4 * fs)))
    total = np.sum(p) + 1e-30
    hf = np.sum(p[f > hf_cutoff])
    noise_ratio = float(hf / total)

    peaks = detect_r_peaks(x, fs)
    hr = heart_rate_from_peaks(peaks, fs)
    rr = rr_intervals(peaks, fs) if len(peaks) > 1 else np.array([])
    rr_cv = float(np.std(rr) / np.mean(rr)) if len(rr) > 1 else float("nan")

    score = 1.0
    if flat or saturated:
        score = 0.0
    else:
        score *= max(0.0, 1.0 - 3.0 * noise_ratio)
        if not (30 <= hr <= 220):
            score *= 0.2
        if np.isfinite(rr_cv):
            score *= max(0.2, 1.0 - rr_cv)
        expected = n / fs / 60.0 * (hr if np.isfinite(hr) else 70)
        if len(peaks) < 0.5 * expected:
            score *= 0.3
    return QualityReport(float(np.clip(score, 0, 1)), score >= 0.5, hr, int(len(peaks)),
                         rr_cv, noise_ratio, bool(flat), bool(saturated))


# --------------------------------------------------------------------------- synthetic
def synthetic_ecg(duration_s: float, fs: float, hr_bpm: float = 60.0, noise_uv: float = 0.0,
                  seed: int | None = 0, amplitude_mv: float = 1.0) -> np.ndarray:
    """Simple synthetic ECG (Gaussian P/Q/R/S/T bumps) in millivolts. Used by tests
    and by the validation harness as a stand-in when no hardware is available."""
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    rr = 60.0 / hr_bpm
    phase = (t % rr) / rr  # 0..1 within each beat
    # Waves as (center, width, amplitude) in beat-phase units
    waves = [
        (0.15, 0.025, 0.12),   # P
        (0.28, 0.008, -0.10),  # Q
        (0.30, 0.010, 1.00),   # R
        (0.32, 0.008, -0.20),  # S
        (0.55, 0.045, 0.30),   # T
    ]
    x = np.zeros(n)
    for c, w, a in waves:
        x += a * np.exp(-((phase - c) ** 2) / (2 * w**2))
    x *= amplitude_mv
    if noise_uv > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0, noise_uv / 1000.0, n)
    return x
