import numpy as np
import pytest

from open_kardia import dsp


@pytest.mark.parametrize("fs", [250.0, 300.0, 500.0])
@pytest.mark.parametrize("hr", [48.0, 72.0, 110.0, 150.0])
def test_hr_on_synthetic(fs, hr):
    x = dsp.synthetic_ecg(30.0, fs, hr_bpm=hr, noise_uv=30.0)
    est = dsp.heart_rate(x, fs)
    assert est == pytest.approx(hr, abs=1.0)


def test_hr_last_window():
    fs = 300.0
    x = dsp.synthetic_ecg(30.0, fs, hr_bpm=65.0, noise_uv=20.0)
    est = dsp.heart_rate(x, fs, window_s=5.0)
    assert est == pytest.approx(65.0, abs=1.5)


def test_peak_count():
    fs = 300.0
    x = dsp.synthetic_ecg(30.0, fs, hr_bpm=60.0)
    peaks = dsp.detect_r_peaks(x, fs)
    assert 29 <= len(peaks) <= 31


def test_filters_preserve_qrs():
    fs = 300.0
    x = dsp.synthetic_ecg(10.0, fs, hr_bpm=60.0, noise_uv=50.0)
    y = dsp.clean(x, fs)
    assert len(y) == len(x)
    # QRS amplitude roughly preserved (within 30 %)
    assert np.max(y) == pytest.approx(np.max(x), rel=0.3)


def test_notch_removes_mains():
    fs = 300.0
    t = np.arange(0, 5, 1 / fs)
    x = dsp.synthetic_ecg(5.0, fs) + 0.5 * np.sin(2 * np.pi * 60 * t)
    y = dsp.notch(x, fs, 60.0)
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / fs)
    idx = np.argmin(np.abs(f - 60))
    spec_x = np.abs(np.fft.rfft(x))
    assert spec[idx] < 0.05 * spec_x[idx]


def test_quality_good_vs_flat_vs_noise():
    fs = 300.0
    good = dsp.synthetic_ecg(15.0, fs, hr_bpm=70.0, noise_uv=20.0)
    q = dsp.signal_quality(good, fs)
    assert q.readable and q.sqi > 0.5
    flat = np.zeros(int(15 * fs))
    assert not dsp.signal_quality(flat, fs).readable
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, int(15 * fs))
    assert dsp.signal_quality(noise, fs).sqi < dsp.signal_quality(good, fs).sqi
