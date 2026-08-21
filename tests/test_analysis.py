"""Tests for open_kardia.analysis — post-recording rhythm/interval extraction.

Uses synthetic ECG for deterministic checks and the real capture at
captures/rec_cli.csv (if present) for physiologic-range assertions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from open_kardia import dsp
from open_kardia.analysis import AnalysisResult, Intervals, analyze_recording
from open_kardia.leads import LEAD_NAMES_6, derive_leads

REC = Path(__file__).resolve().parents[1] / "captures" / "rec_cli.csv"


def _synth_6lead(duration=30.0, fs=300.0, hr=60.0, noise_uv=15.0):
    lead_i = dsp.synthetic_ecg(duration, fs, hr_bpm=hr, noise_uv=noise_uv, seed=1)
    lead_ii = dsp.synthetic_ecg(duration, fs, hr_bpm=hr, noise_uv=noise_uv, seed=2, amplitude_mv=1.25)
    return derive_leads(lead_i, lead_ii).as_matrix(), fs, list(LEAD_NAMES_6)


# --------------------------------------------------------------------------- basic contract
def test_returns_result_and_never_raises_on_flat():
    x = np.zeros((3000, 6))
    r = analyze_recording(x, 300.0, list(LEAD_NAMES_6))
    assert isinstance(r, AnalysisResult)
    assert r.quality_sqi == 0.0
    assert not np.isfinite(r.intervals.qrs_ms)
    assert r.warnings


def test_never_raises_on_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (3000, 1))
    r = analyze_recording(x, 300.0, ["I"])
    assert isinstance(r, AnalysisResult)
    # noise should not be called a confident normal rhythm
    assert r.rhythm.label in {"Too noisy to classify", "Irregular rhythm", "Possible atrial fibrillation",
                              "Undetermined", "Sinus tachycardia", "Sinus arrhythmia"}


def test_too_short():
    r = analyze_recording(np.zeros((100, 1)), 300.0, ["I"])
    assert r.n_beats == 0
    assert any("short" in w for w in r.warnings)


def test_single_lead_no_axis():
    x = dsp.synthetic_ecg(30.0, 300.0, hr_bpm=65, noise_uv=10)[:, None]
    r = analyze_recording(x, 300.0, ["I"])
    assert r.lead_used == "I"
    assert not np.isfinite(r.qrs_axis_deg)     # axis needs I and aVF
    assert r.axis_label == "Undetermined"
    assert r.n_beats > 25


# --------------------------------------------------------------------------- synthetic physiology
def test_synthetic_normal_sinus():
    x, fs, names = _synth_6lead(hr=60.0)
    r = analyze_recording(x, fs, names)
    assert r.lead_used == "II"
    assert r.rhythm.label == "Normal sinus rhythm"
    assert abs(r.intervals.hr_mean - 60) < 3
    assert np.isfinite(r.qrs_axis_deg)         # 6-lead -> axis available
    assert np.isfinite(r.intervals.qrs_ms)


def test_bradycardia_label():
    x, fs, names = _synth_6lead(hr=48.0)
    r = analyze_recording(x, fs, names)
    assert r.rhythm.label == "Sinus bradycardia"
    assert r.intervals.hr_mean < 60


def test_tachycardia_label():
    x, fs, names = _synth_6lead(hr=115.0)
    r = analyze_recording(x, fs, names)
    assert r.rhythm.label == "Sinus tachycardia"
    assert r.intervals.hr_mean > 100


def test_hr_matches_dsp():
    x, fs, names = _synth_6lead(hr=72.0)
    r = analyze_recording(x, fs, names)
    hr_dsp = dsp.heart_rate(x[:, names.index("II")], fs)
    assert abs(r.intervals.hr_mean - hr_dsp) < 2.0


def test_median_beat_and_fiducials_present():
    x, fs, names = _synth_6lead(hr=60.0)
    r = analyze_recording(x, fs, names)
    assert set(r.median_beat.keys()) == set(names)
    for b in r.median_beat.values():
        assert b.ndim == 1 and len(b) > int(0.5 * fs)
    assert "ECG_R_Peaks" in r.fiducials


def test_hrv_fields():
    x, fs, names = _synth_6lead(hr=60.0)
    r = analyze_recording(x, fs, names)
    for k in ("SDNN", "RMSSD", "pNN50"):
        assert k in r.hrv


def test_summary_string():
    x, fs, names = _synth_6lead()
    r = analyze_recording(x, fs, names)
    s = r.summary()
    assert "Rhythm:" in s and "QRS" in s and "axis" in s.lower()


# --------------------------------------------------------------------------- real capture
@pytest.mark.skipif(not REC.exists(), reason="captures/rec_cli.csv not present")
def test_real_capture_physiologic():
    from open_kardia.sinks.csv_sink import read_csv
    fs, names, t, x = read_csv(REC)
    r = analyze_recording(x, fs, names)
    iv = r.intervals
    # HR agreement with dsp
    hr_dsp = dsp.heart_rate(x[:, names.index("II")], fs)
    assert abs(iv.hr_mean - hr_dsp) < 3.0
    # physiologic interval ranges on a real normal ECG
    assert 60 <= iv.qrs_ms <= 120, f"QRS {iv.qrs_ms}"
    assert 120 <= iv.pr_ms <= 220, f"PR {iv.pr_ms}"
    assert 300 <= iv.qtc_bazett_ms <= 480, f"QTcB {iv.qtc_bazett_ms}"
    assert 300 <= iv.qt_ms <= 500, f"QT {iv.qt_ms}"
    assert np.isfinite(r.qrs_axis_deg)
    assert r.axis_label in {"Normal", "Left axis deviation", "Right axis deviation", "Extreme axis deviation"}
    assert r.rhythm.label == "Normal sinus rhythm"
    assert r.quality_sqi > 0.5


# --- normal-range reference API -------------------------------------------------------
def test_reference_ranges_and_classify():
    from open_kardia import analysis as A
    # every interval key in rows() that has a defined range classifies correctly
    assert A.classify("pr_ms", 160) == "normal"
    assert A.classify("pr_ms", 90) == "low"
    assert A.classify("pr_ms", 240) == "high"
    assert A.classify("qrs_ms", 140) == "high"
    assert A.classify("qtc_bazett_ms", 500) == "high"
    assert A.classify("st_level_mv", 0.0) == "normal"
    assert A.classify("qrs_axis_deg", -60) == "low"    # left-axis
    # nan / undefined -> ''
    assert A.classify("pr_ms", float("nan")) == ""
    assert A.classify("qt_ms", 400) == "normal"        # qt has no bounds -> normal by default
    # range_text formatting
    assert A.range_text("pr_ms") == "120–200 ms"
    assert A.range_text("qrs_axis_deg") == "−30–90°"
    assert A.range_text("qt_ms") == ""                 # no numeric bounds


def test_rows_carry_ranges_and_status():
    import numpy as np
    from open_kardia import dsp
    from open_kardia.analysis import analyze_recording
    fs = 300.0
    x = np.column_stack([dsp.synthetic_ecg(20, fs, 62, noise_uv=15),
                         dsp.synthetic_ecg(20, fs, 62, noise_uv=15, seed=2, amplitude_mv=1.3)])
    res = analyze_recording(x, fs, ["I", "II"])
    rows = res.rows()
    keys = {r["key"] for r in rows}
    for k in ("hr_mean", "pr_ms", "qrs_ms", "qt_ms", "qtc_bazett_ms", "qtc_fridericia_ms",
              "p_dur_ms", "st_level_mv", "t_dur_ms", "qrs_axis_deg"):
        assert k in keys
    for r in rows:
        assert set(r) >= {"label", "key", "value", "unit", "range", "status"}
        assert r["status"] in ("low", "normal", "high", "")
    # summary now quotes ranges
    s = res.summary()
    assert "normal 120–200 ms" in s and "normal 70–110 ms" in s
