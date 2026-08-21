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


def _count_pdf_pages(path):
    import re
    data = open(path, "rb").read()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


def test_pdf_is_multipage_full_disclosure(tmp_path):
    """PDF = summary page + the whole recording at 10 s per row group (true scale)."""
    from kardia6l import dsp
    fs = 300.0
    x1 = dsp.synthetic_ecg(35.0, fs, hr_bpm=70.0)                # 35 s -> 4 row groups
    x6 = np.column_stack([x1 * (0.8 + 0.1 * i) for i in range(6)])
    t = np.arange(len(x1)) / fs
    out = report.render_report(t, x6, list("ABCDEF"), fs, analysis=None,
                               out_path=str(tmp_path / "full"), fmt="pdf")
    # 6 leads -> one 10-s group per landscape page -> 1 + 4 pages
    assert _count_pdf_pages(out) == 5
    # single lead -> several 10-s rows per page -> fewer pages, but still > 1
    out1 = report.render_report(t, x1[:, None], ["I"], fs, analysis=None,
                                out_path=str(tmp_path / "full1"), fmt="pdf")
    n1 = _count_pdf_pages(out1)
    assert 2 <= n1 <= 3
    # opt-out gives the single summary page; PNG is always one image
    out2 = report.render_report(t, x6, list("ABCDEF"), fs, analysis=None,
                                out_path=str(tmp_path / "one"), fmt="pdf", full_disclosure=False)
    assert _count_pdf_pages(out2) == 1
    # a short recording (<= 10 s) needs no continuation pages
    outs = report.render_report(t[:2400], x6[:2400], list("ABCDEF"), fs, analysis=None,
                                out_path=str(tmp_path / "short"), fmt="pdf")
    assert _count_pdf_pages(outs) == 2   # summary + one 8-s group


def test_pdf_pages_carry_event_and_episode_labels(tmp_path):
    """The rhythm findings, their times/episodes, the AF episode shading label and the V/S
    beat markers must all be printed in the PDF (summary panel + full-disclosure pages)."""
    fitz = pytest.importorskip("fitz")     # pymupdf: text extraction from the rendered PDF
    from types import SimpleNamespace as NS
    from kardia6l import dsp
    from kardia6l.arrhythmia import Finding, annotate_times
    fs = 300.0
    x = dsp.synthetic_ecg(40.0, fs, hr_bpm=72.0)
    peaks = dsp.detect_r_peaks(dsp.clean(x, fs), fs)
    labels = np.array(["N"] * len(peaks), dtype=object)
    labels[5] = "V"; labels[7] = "V"; labels[9] = "V"; labels[12] = "S"
    f_pvc = Finding("pvc", "Premature ventricular complexes", "notable", 0.8, "3 PVC-like beats", 3, [5, 7, 9])
    f_big = Finding("bigeminy", "Possible ventricular bigeminy", "notable", 0.6, "3 cycles", 3, [5, 7, 9])
    annotate_times([f_pvc, f_big], peaks, fs)
    arr = NS(findings=[f_pvc, f_big], beat_labels=labels, summary="PVCs", metrics={}, warnings=[])
    af = NS(is_af=True, label="Possible atrial fibrillation (episodes)", score=0.6, confidence=0.6,
            evidence="AF-like RR", episodes=[(20.0, 32.0, 0.8)], metrics={}, warnings=[])
    analysis = NS(rhythm=NS(label="Possible atrial fibrillation (episodes)", confidence=0.6, rationale="", flags=[]),
                  intervals=None, hrv={}, qrs_axis_deg=None, axis_label="—", median_beat={}, median_beat_fs=fs,
                  quality_sqi=0.9, n_beats=len(peaks), lead_used="II", duration_s=40.0, warnings=[],
                  arrhythmia=arr, af=af)
    out = report.render_report(np.arange(len(x)) / fs, x[:, None], ["II"], fs, analysis=analysis,
                               out_path=str(tmp_path / "ev"), fmt="pdf")
    doc = fitz.open(out)
    assert doc.page_count == 2                       # summary + one page of 4 x 10-s rows (3 findings fit on page 1)
    p1, p2 = doc[0].get_text(), doc[1].get_text()
    assert "Rhythm findings" in p1 and "Possible atrial fibrillation (episodes)" in p1
    assert "episodes 20–32 s" in p1                  # AF 'when'
    assert "at " in p1 and "bigeminy" in p1.lower()  # PVC times + pattern finding
    assert "AF?" in p2 and "bigeminy" in p2          # shaded episode labels on the strips
    assert "\nV\n" in p2 and "\nS\n" in p2           # beat markers


def test_pdf_adds_findings_page_when_many(tmp_path):
    fitz = pytest.importorskip("fitz")
    from types import SimpleNamespace as NS
    from kardia6l import dsp
    from kardia6l.arrhythmia import Finding
    fs = 300.0
    x = dsp.synthetic_ecg(12.0, fs, hr_bpm=72.0)
    fnds = [Finding(c, f"Finding {c}", "info", 0.5, "evidence " * 6, 0, []) for c in ("a", "b", "c", "d", "e")]
    arr = NS(findings=fnds, beat_labels=None, summary="", metrics={}, warnings=[])
    analysis = NS(rhythm=NS(label="X", confidence=0.5, rationale="", flags=[]), intervals=None, hrv={},
                  qrs_axis_deg=None, axis_label="—", median_beat={}, median_beat_fs=fs, quality_sqi=0.9,
                  n_beats=10, lead_used="II", duration_s=12.0, warnings=[], arrhythmia=arr, af=None)
    out = report.render_report(np.arange(len(x)) / fs, x[:, None], ["II"], fs, analysis=analysis,
                               out_path=str(tmp_path / "many"), fmt="pdf")
    doc = fitz.open(out)
    assert doc.page_count == 3                       # summary + findings page + 1 strip page
    p2 = doc[1].get_text()
    assert "Rhythm findings" in p2 and all(f"Finding {c}" in p2 for c in "abcde") and "When: whole recording" in p2
    assert "all findings on page 2" in doc[0].get_text()
