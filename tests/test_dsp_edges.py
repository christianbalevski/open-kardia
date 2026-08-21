"""Tests for the edge-stable display filters (`clean_stable`, `display_filter`).

These verify that padding-based filtering removes the ringing transient that plain
`clean` (filtfilt on a short slice) leaves at the ends of the displayed window.
"""
import numpy as np
import pytest

from open_kardia import dsp


def _edge_ratio(y: np.ndarray, fs: float, edge_s: float = 0.150) -> float:
    """Max |value| in the first/last `edge_s` relative to the interior RMS.
    Larger -> more edge transient/ringing."""
    e = max(1, int(edge_s * fs))
    interior = y[e:-e]
    rms = np.sqrt(np.mean(interior**2)) + 1e-12
    edge_peak = max(np.max(np.abs(y[:e])), np.max(np.abs(y[-e:])))
    return float(edge_peak / rms)


def _long_and_slice(fs=300.0):
    """A long ECG record (with baseline wander) and a 6 s middle slice. The ideal filtered
    result for the slice is the interior of the long record's filtered output — any
    deviation at the slice edges is exactly the windowed-filter transient we want gone."""
    full = dsp.synthetic_ecg(20.0, fs, hr_bpm=72.0, noise_uv=15.0, seed=3)
    t = np.arange(len(full)) / fs
    full = full + 0.4 * np.sin(2 * np.pi * 0.15 * t)     # slow baseline wander
    a, b = int(7 * fs), int(13 * fs)
    ideal = dsp.clean(full, fs)[a:b]                     # edges are interior -> transient-free
    return full[a:b], ideal, fs


def _edge_deviation(y, ideal, fs, edge_s=0.150):
    """Max deviation of `y` from the ideal (long-filter) result within the edge regions."""
    e = max(1, int(edge_s * fs))
    d = np.abs(y - ideal)
    return float(max(np.max(d[:e]), np.max(d[-e:])))


def test_clean_stable_reduces_edge_transient():
    x, ideal, fs = _long_and_slice()
    y_plain = dsp.clean(x, fs)
    y_stable = dsp.clean_stable(x, fs)
    dev_plain = _edge_deviation(y_plain, ideal, fs)
    dev_stable = _edge_deviation(y_stable, ideal, fs)
    # clean_stable's edges must track the ideal filter far better than plain clean's
    assert dev_stable < dev_plain, (dev_stable, dev_plain)
    assert dev_stable < 0.7 * dev_plain, (
        f"expected >=30% edge-transient reduction, got plain={dev_plain:.4f} stable={dev_stable:.4f}")


def _slice_with_edge_offset(fs=300.0):
    x, _ideal, fs = _long_and_slice(fs)
    return x, fs


def test_clean_stable_preserves_length_and_interior():
    x, fs = _slice_with_edge_offset()
    y_plain = dsp.clean(x, fs)
    y_stable = dsp.clean_stable(x, fs)
    assert y_stable.shape == x.shape
    # interior (middle 80%) should agree closely with plain clean
    n = len(x)
    lo, hi = int(0.1 * n), int(0.9 * n)
    corr = np.corrcoef(y_plain[lo:hi], y_stable[lo:hi])[0, 1]
    assert corr > 0.98, corr


def test_clean_stable_short_input_no_crash():
    fs = 300.0
    x = dsp.synthetic_ecg(0.5, fs, hr_bpm=60.0)  # shorter than pad_s
    y = dsp.clean_stable(x, fs, pad_s=1.0)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    # empty input
    assert dsp.clean_stable(np.array([]), fs).shape == (0,)


def test_display_filter_returns_last_window_1d():
    fs = 300.0
    buf = dsp.synthetic_ecg(30.0, fs, hr_bpm=66.0, noise_uv=20.0, seed=1)
    out = dsp.display_filter(buf, fs, window_s=6.0)
    assert out.shape == (int(6 * fs),)
    # interior slice of a long filtered signal -> edges must be clean
    assert _edge_ratio(out, fs) < 6.0
    # and much cleaner at the window edge than naively filtering just that window
    naive = dsp.clean(buf[-int(6 * fs):], fs)
    assert _edge_ratio(out, fs) < _edge_ratio(naive, fs)


def test_display_filter_2d_shape_and_rank():
    fs = 300.0
    cols = [dsp.synthetic_ecg(30.0, fs, hr_bpm=66.0, noise_uv=20.0, seed=s) for s in range(3)]
    buf = np.column_stack(cols)
    out = dsp.display_filter(buf, fs, window_s=6.0)
    assert out.shape == (int(6 * fs), 3)
    for c in range(3):
        assert _edge_ratio(out[:, c], fs) < 6.0


def test_display_filter_window_none_returns_full():
    fs = 300.0
    buf = dsp.synthetic_ecg(10.0, fs, hr_bpm=60.0)
    out = dsp.display_filter(buf, fs, window_s=None)
    assert out.shape == buf.shape


@pytest.mark.parametrize("window_s", [4.0, 6.0, 8.0])
def test_display_filter_various_windows(window_s):
    fs = 300.0
    buf = dsp.synthetic_ecg(30.0, fs, hr_bpm=70.0, noise_uv=15.0)
    out = dsp.display_filter(buf, fs, window_s=window_s)
    assert out.shape == (int(window_s * fs),)
