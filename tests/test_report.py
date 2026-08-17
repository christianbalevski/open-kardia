"""Tests for kardia6l.report — the printable ECG report renderer."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from kardia6l import dsp, report
from kardia6l.leads import LEAD_NAMES_6, derive_leads


def _synth_6lead(secs=12.0, fs=300.0, hr=66.0):
    I = dsp.synthetic_ecg(secs, fs, hr_bpm=hr, noise_uv=20.0, seed=1)
    II = dsp.synthetic_ecg(secs, fs, hr_bpm=hr, noise_uv=20.0, seed=2, amplitude_mv=1.3)
    six = derive_leads(I, II)
    x = six.as_matrix()
    t = np.arange(x.shape[0]) / fs
    return t, x, list(LEAD_NAMES_6), fs


def _stub_analysis():
    """A minimal duck-typed analysis object with a couple of fields populated and
    several intentionally missing/NaN, to exercise the defensive accessors."""
    intervals = SimpleNamespace(
        pr_ms=160.0, qrs_ms=94.0, qt_ms=380.0, qtc_bazett_ms=410.0,
        qtc_fridericia_ms=float("nan"), st_level_mv=0.02, p_dur_ms=None,
        rr_mean_ms=900.0, rr_sd_ms=35.0, hr_mean=66.0, hr_min=60.0, hr_max=72.0)
    rhythm = SimpleNamespace(label="Normal Sinus Rhythm", confidence=0.92,
                             rationale="regular RR, P before each QRS", flags=[])
    mb = {"I": dsp.synthetic_ecg(0.8, 300.0, 66.0)[:240],
          "II": dsp.synthetic_ecg(0.8, 300.0, 66.0, amplitude_mv=1.3)[:240]}
    return SimpleNamespace(
        intervals=intervals, rhythm=rhythm, hrv={"sdnn_ms": 42.0, "rmssd_ms": 30.0, "pnn50": 8.5},
        qrs_axis_deg=45.0, axis_label="Normal", median_beat=mb, median_beat_fs=300.0,
        n_beats=13, duration_s=12.0, lead_used="II", quality_sqi=0.93, warnings=[])


@pytest.mark.parametrize("fmt", ["png", "pdf"])
def test_render_full(tmp_path, fmt):
    t, x, names, fs = _synth_6lead()
    out = report.render_report(t, x, names, fs, analysis=_stub_analysis(),
                               out_path=str(tmp_path / "rep"), fmt=fmt,
                               meta={"recorded_at": "2026-08-16 17:57", "patient_name": "Test"})
    assert os.path.exists(out) and out.endswith(fmt)
    assert os.path.getsize(out) > 20_000


def test_render_no_analysis(tmp_path):
    t, x, names, fs = _synth_6lead()
    out = report.render_report(t, x, names, fs, analysis=None,
                               out_path=str(tmp_path / "rep_bare"), fmt="png")
    assert os.path.exists(out) and os.path.getsize(out) > 20_000


def test_render_partial_analysis(tmp_path):
    """A nearly-empty analysis object must still render (defensive _get)."""
    t, x, names, fs = _synth_6lead()
    partial = SimpleNamespace(intervals=SimpleNamespace(hr_mean=70.0), rhythm=None)
    out = report.render_report(t, x, names, fs, analysis=partial,
                               out_path=str(tmp_path / "rep_partial"), fmt="png")
    assert os.path.exists(out) and os.path.getsize(out) > 20_000


def test_render_single_lead(tmp_path):
    fs = 300.0
    I = dsp.synthetic_ecg(10.0, fs, hr_bpm=72.0, noise_uv=15.0)
    t = np.arange(I.size) / fs
    out = report.render_report(t, I[:, None], ["I"], fs, analysis=None,
                               out_path=str(tmp_path / "rep_1lead"), fmt="png")
    assert os.path.exists(out) and os.path.getsize(out) > 15_000


def test_get_helper():
    ns = SimpleNamespace(intervals=SimpleNamespace(pr_ms=160.0), rhythm=None)
    assert report._get(ns, "intervals.pr_ms") == 160.0
    assert report._get(ns, "intervals.qrs_ms", "d") == "d"
    assert report._get(ns, "rhythm.label", "—") == "—"
    assert report._get(None, "a.b.c", 5) == 5
    assert report._get({"a": {"b": 2}}, "a.b") == 2


def test_fmt_helper():
    assert report._fmt(float("nan")) == "—"
    assert report._fmt(None) == "—"
    assert report._fmt(160.4, " ms") == "160 ms"
    assert report._fmt(0.023, " mV", nd=2) == "0.02 mV"


@pytest.mark.skipif(not os.path.exists("captures/rec_cli.csv"),
                    reason="no real capture present")
def test_render_from_real_csv(tmp_path):
    out = report.render_report_from_csv("captures/rec_cli.csv", analysis=None,
                                        out_path=str(tmp_path / "real"), fmt="png")
    assert os.path.exists(out) and os.path.getsize(out) > 20_000
