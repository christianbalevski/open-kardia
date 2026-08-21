"""Tests for kardia6l.arrhythmia — beat-level arrhythmia screening on synthetic ECGs.

No network, no hardware: beats are synthesised (Gaussian P/QRS/T bumps in time) with
controlled RR sequences and morphologies, then run through dsp.clean/detect_r_peaks
exactly like the analysis pipeline would.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from kardia6l import dsp
from kardia6l.analysis import Intervals
from kardia6l.arrhythmia import ArrhythmiaReport, Finding, detect_arrhythmias

FS = 300.0


def _ecg(rr: list, kinds: list, fs: float = FS, noise_uv: float = 8.0, seed: int = 0):
    """ECG (mV) with R peaks at 1 s + cumsum(rr); kinds[i] in {'N','V'} sets the morphology
    of beat i ('V' = wide, inverted, no P, discordant T). Returns (x, r_times_s)."""
    t_r = np.concatenate([[1.0], 1.0 + np.cumsum(rr)])
    n = int((t_r[-1] + 1.0) * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for t0, k in zip(t_r, kinds):
        if k == "V":
            waves = [(0.0, 0.030, -1.2), (0.30, 0.045, 0.45)]
        else:
            waves = [(-0.16, 0.020, 0.12), (-0.02, 0.007, -0.10), (0.0, 0.009, 1.0),
                     (0.02, 0.007, -0.20), (0.25, 0.040, 0.30)]
        for c, w, a in waves:
            x += a * np.exp(-((t - t0 - c) ** 2) / (2 * w**2))
    x += np.random.default_rng(seed).normal(0, noise_uv / 1000.0, n)
    return x, t_r


def _run(rr, kinds, **kw):
    x, _ = _ecg(rr, kinds)
    c = dsp.clean(x, FS)
    rp = dsp.detect_r_peaks(c, FS)
    return detect_arrhythmias(c, FS, rp, **kw), rp


def _codes(rep):
    return {f.code for f in rep.findings}


# --------------------------------------------------------------------------- (a) PVCs
def test_pvcs_every_fifth_beat():
    rr, kinds = [], ["N"]
    for _ in range(8):
        rr += [1.0, 1.0, 1.0, 0.6, 1.4]        # premature wide beat + compensatory pause
        kinds += ["N", "N", "N", "V", "N"]
    rep, rp = _run(rr, kinds)
    lab = "".join(rep.beat_labels)
    assert len(rep.beat_labels) == len(rp)
    n_v = lab.count("V")
    assert n_v >= 7, lab                        # 8 injected
    assert lab.count("S") == 0
    assert "pvc" in _codes(rep)
    f = next(f for f in rep.findings if f.code == "pvc")
    assert f.severity == "notable" and "frequent" in f.label   # 20 % burden
    assert f.count == n_v and len(f.beats) == n_v
    assert 0.0 < f.confidence <= 1.0
    assert "PVC" in rep.summary and rep.metrics["pvc_burden"] > 0.15
    # every injected PVC is the 5th beat of its cycle (index 4, 9, 14, ...)
    for i, ch in enumerate(lab):
        if ch == "V":
            assert i % 5 == 4, (i, lab)


def test_ventricular_bigeminy():
    rr, kinds = [], ["N"]
    for _ in range(12):
        rr += [0.6, 1.4]
        kinds += ["V", "N"]
    rep, _ = _run(rr, kinds)
    lab = "".join(rep.beat_labels)
    assert lab.count("V") >= 10, lab
    assert "bigeminy" in _codes(rep) and "pvc" in _codes(rep)
    big = next(f for f in rep.findings if f.code == "bigeminy")
    assert big.count >= 3 and "ventricular" in big.label
    assert rep.metrics["bigeminy_cycles"] >= 3


def test_couplets_and_run():
    rr = [0.9] * 6 + [0.55, 0.5, 1.5] + [0.9] * 8 + [0.55, 0.5, 1.5] + [0.9] * 6
    kinds = ["N"] * 7 + ["V", "V", "N"] + ["N"] * 8 + ["V", "V", "N"] + ["N"] * 6
    rep, _ = _run(rr, kinds)
    assert "couplet" in _codes(rep)
    assert rep.metrics["n_couplets"] >= 1
    rr = [0.8] * 10 + [0.45] * 6 + [1.0] + [0.8] * 10
    kinds = ["N"] * 11 + ["V"] * 6 + ["N"] * 11
    rep, _ = _run(rr, kinds)
    assert "vt_run" in _codes(rep)
    run = next(f for f in rep.findings if f.code == "vt_run")
    assert run.severity == "urgent" and rep.metrics["max_run_len"] >= 3
    assert rep.findings[0].severity == "urgent"          # most severe first


# --------------------------------------------------------------------------- (b) PACs
def test_pacs_premature_narrow():
    rr, kinds = [], ["N"]
    for _ in range(8):
        rr += [0.9, 0.9, 0.9, 0.65, 1.0]        # premature, normal morphology, non-compensatory
        kinds += ["N"] * 5
    rep, _ = _run(rr, kinds)
    lab = "".join(rep.beat_labels)
    assert lab.count("S") >= 7, lab
    assert lab.count("V") == 0, lab
    assert "pac" in _codes(rep) and "pvc" not in _codes(rep)
    assert rep.metrics["pac_count"] == lab.count("S")


def test_svt_run():
    rr = [0.8] * 10 + [0.38] * 8 + [0.9] + [0.8] * 10
    rep, _ = _run(rr, ["N"] * (len(rr) + 1))
    assert "svt_run" in _codes(rep)
    assert "".join(rep.beat_labels).count("S") >= 6


# --------------------------------------------------------------------------- (c) pause
def test_pause_detected():
    rr = [0.9] * 12 + [2.5] + [0.9] * 12
    rep, _ = _run(rr, ["N"] * (len(rr) + 1))
    assert "pause" in _codes(rep)
    p = next(f for f in rep.findings if f.code == "pause")
    assert p.count == 1 and p.severity == "notable"
    assert abs(rep.metrics["longest_rr_ms"] - 2500) < 60
    assert "V" not in rep.beat_labels and "S" not in rep.beat_labels


# --------------------------------------------------------------------------- (d) clean sinus
def test_clean_sinus_no_findings():
    x = dsp.synthetic_ecg(30.0, FS, hr_bpm=70, noise_uv=10)
    c = dsp.clean(x, FS)
    rp = dsp.detect_r_peaks(c, FS)
    rep = detect_arrhythmias(c, FS, rp)
    assert isinstance(rep, ArrhythmiaReport)
    assert rep.findings == [] and rep.summary == ""
    assert set(rep.beat_labels) == {"N"}
    assert rep.metrics["pvc_count"] == 0 and rep.metrics["pac_count"] == 0
    assert 65 < rep.metrics["hr_nn"] < 75


def test_rate_screens():
    rep, _ = _run([0.375] * 60, ["N"] * 61)              # regular narrow 160 bpm
    assert "svt" in _codes(rep)
    rep, _ = _run([1.58] * 15, ["N"] * 16)               # 38 bpm
    b = next(f for f in rep.findings if f.code == "sinus_brady")
    assert b.severity == "notable"
    rep, _ = _run([0.4] * 50, ["V"] * 51)                # regular wide 150 bpm
    assert "vt" in _codes(rep)


# --------------------------------------------------------------------------- (e) intervals
def test_interval_flags():
    x = dsp.synthetic_ecg(30.0, FS, hr_bpm=70, noise_uv=10)
    c = dsp.clean(x, FS)
    rp = dsp.detect_r_peaks(c, FS)
    iv = Intervals(pr_ms=220.0, qrs_ms=130.0, qtc_bazett_ms=480.0, hr_mean=70.0)
    rep = detect_arrhythmias(c, FS, rp, intervals=iv)
    codes = _codes(rep)
    assert {"first_degree_av_block", "wide_qrs", "long_qtc"} <= codes
    sev = {f.code: f.severity for f in rep.findings}
    assert sev["first_degree_av_block"] == "info"
    assert sev["wide_qrs"] == "notable" and sev["long_qtc"] == "notable"
    assert "PR 220" in rep.summary
    # urgent QTc and short PR
    rep = detect_arrhythmias(c, FS, rp, intervals=Intervals(pr_ms=100.0, qtc_bazett_ms=520.0))
    sev = {f.code: f.severity for f in rep.findings}
    assert sev["long_qtc"] == "urgent" and sev["short_pr"] == "info"
    assert rep.findings[0].code == "long_qtc"
    # nan intervals -> nothing
    rep = detect_arrhythmias(c, FS, rp, intervals=Intervals())
    assert rep.findings == []


# --------------------------------------------------------------------------- (f) garbage
def test_garbage_input_never_raises():
    cases = [
        (np.zeros(10), FS, np.array([])),
        (np.zeros(3000), FS, np.array([])),
        (np.zeros(3000), FS, np.array([100, 500, 900, 1300, 1700])),
        (np.full(3000, np.nan), FS, np.array([100, 500, 900])),
        (np.random.default_rng(0).normal(0, 1, 3000), FS, np.array([5, 10])),
        (np.random.default_rng(0).normal(0, 1, 3000), FS, np.array([100, 200, 300, 400, 2900, 3500])),
        (np.zeros(3000), 0.0, np.array([1, 2, 3, 4, 5])),
        ("abc", FS, None),
        (np.zeros((100, 2)), FS, np.array([1, 2, 3, 4])),
    ]
    for x, fs, rp in cases:
        rep = detect_arrhythmias(x, fs, rp)
        assert isinstance(rep, ArrhythmiaReport)
        assert isinstance(rep.beat_labels, np.ndarray)
        assert isinstance(rep.findings, list) and isinstance(rep.metrics, dict)
    rep = detect_arrhythmias(np.zeros(10), FS, np.array([]))
    assert rep.findings == [] and rep.summary == "" and rep.warnings


# --------------------------------------------------------------------------- (g) JSON
def test_json_serializable():
    rr, kinds = [], ["N"]
    for _ in range(6):
        rr += [1.0, 1.0, 0.6, 1.4]
        kinds += ["N", "N", "V", "N"]
    rep, _ = _run(rr, kinds, intervals=Intervals(pr_ms=210.0))
    assert rep.findings
    payload = {
        "beat_labels": rep.beat_labels.tolist(),
        "metrics": rep.metrics,
        "findings": [dataclasses.asdict(f) for f in rep.findings],
        "summary": rep.summary,
        "warnings": rep.warnings,
    }
    s = json.dumps(payload, allow_nan=False)      # raises on nan/inf
    back = json.loads(s)
    assert back["metrics"]["pvc_count"] == rep.metrics["pvc_count"]
    for v in rep.metrics.values():
        assert isinstance(v, float) and np.isfinite(v)
    for f in rep.findings:
        assert isinstance(f, Finding)
        assert f.severity in ("info", "notable", "urgent")
        assert 0.0 <= f.confidence <= 1.0
        assert all(isinstance(b, int) for b in f.beats)


def test_median_beat_argument_and_sqi():
    from kardia6l.analysis import _median_beat
    rr, kinds = [], ["N"]
    for _ in range(6):
        rr += [1.0, 1.0, 0.6, 1.4]
        kinds += ["N", "N", "V", "N"]
    x, _ = _ecg(rr, kinds)
    c = dsp.clean(x, FS)
    rp = dsp.detect_r_peaks(c, FS)
    mb, r_idx = _median_beat(c, rp, FS)
    rep = detect_arrhythmias(c, FS, rp, median_beat=mb, r_idx=r_idx, sqi=0.2)
    assert "".join(rep.beat_labels).count("V") >= 5
    assert any("low signal quality" in w for w in rep.warnings)
    rep_hi = detect_arrhythmias(c, FS, rp, median_beat=mb, r_idx=r_idx, sqi=1.0)
    lo = next(f for f in rep.findings if f.code == "pvc").confidence
    hi = next(f for f in rep_hi.findings if f.code == "pvc").confidence
    assert lo < hi


def test_first_beat_in_edge_transient_not_pvc():
    """Regression: a normal first beat inside the filter's edge transient (no RR_pre,
    inflated width estimate) must not be labelled 'V' nor produce a pvc finding."""
    fs = FS
    rr = [0.86] * 30
    t_r = np.concatenate([[0.3], 0.3 + np.cumsum(rr)])          # first R at 0.3 s
    n = int((t_r[-1] + 0.35) * fs)                                # last beat near the end too
    t = np.arange(n) / fs
    x = np.zeros(n)
    for t0 in t_r:
        for c, w, a in [(-0.16, 0.020, 0.12), (-0.02, 0.007, -0.10), (0.0, 0.009, 1.0),
                        (0.02, 0.007, -0.20), (0.25, 0.040, 0.30)]:
            x += a * np.exp(-((t - t0 - c) ** 2) / (2 * w**2))
    x += 0.4 + np.random.default_rng(1).normal(0, 0.008, n)      # DC offset -> edge transient
    c = dsp.clean(x, fs)
    rp = dsp.detect_r_peaks(c, fs)
    assert rp[0] < 0.4 * fs
    rep = detect_arrhythmias(c, fs, rp)
    assert rep.beat_labels[0] == "N" and rep.beat_labels[-1] == "N"
    assert set(rep.beat_labels) == {"N"}
    assert "pvc" not in _codes(rep) and "pac" not in _codes(rep)
    assert rep.metrics["uncertain_v_count"] == 0.0


def test_single_low_confidence_ectopic_goes_to_metrics():
    from kardia6l import arrhythmia as A
    rep = A.ArrhythmiaReport(np.array(["N"]), [], {}, "")
    assert rep.summary == ""
    # a real single PVC with good confidence still yields a finding
    rr = [0.9] * 8 + [0.55, 1.25] + [0.9] * 8
    kinds = ["N"] * 9 + ["V"] + ["N"] * 8
    rep, _ = _run(rr, kinds)
    assert rep.metrics["pvc_count"] == 1.0
    assert "pvc" in _codes(rep) and rep.metrics["uncertain_v_count"] == 0.0
    assert "uncertain_v_count" in rep.metrics and "uncertain_s_count" in rep.metrics


def test_missed_detections_are_not_ectopy_or_pauses():
    """Regression: clean synthetic sinus tachycardia where the R-peak detector drops beats
    (RR ≈ 2x) must not produce vt_run/pvc/pac/pause findings."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_analysis import _synth_6lead
    from kardia6l.analysis import analyze_recording
    x, fs, names = _synth_6lead(hr=115.0)
    r = analyze_recording(x, fs, names)
    rep = r.arrhythmia
    assert rep is not None
    codes = _codes(rep)
    assert not codes & {"vt_run", "pvc", "pac", "pause", "couplet", "svt_run", "vt", "svt"}, codes
    assert codes <= {"sinus_tachy", "possible_missed_beats", "short_qtc", "long_qtc", "short_pr",
                     "first_degree_av_block", "wide_qrs", "irregular"}
    assert set(rep.beat_labels) <= {"N", "?"}
    if "possible_missed_beats" in codes:
        f = next(f for f in rep.findings if f.code == "possible_missed_beats")
        assert f.severity == "info" and f.count >= 3
    # a synthetic run of missed beats built by hand: every 4th beat removed from a 100 bpm train
    rr = []
    for k in range(10):
        rr += [0.6, 0.6, 1.2]
    rep, _ = _run(rr, ["N"] * (len(rr) + 1))
    assert not _codes(rep) & {"pvc", "pac", "pause", "vt_run"}
    assert "possible_missed_beats" in _codes(rep)
    assert set(rep.beat_labels) == {"N"}


def test_premature_normal_morphology_is_S_not_V():
    rr = [0.9] * 8 + [0.6, 1.0] + [0.9] * 8
    rep, _ = _run(rr, ["N"] * (len(rr) + 1))
    lab = "".join(rep.beat_labels)
    assert lab.count("S") == 1 and lab.count("V") == 0, lab
    assert lab.index("S") == 9
    assert "pac" in _codes(rep) and "pvc" not in _codes(rep)


def test_low_sqi_caps_severity():
    rr = [0.8] * 10 + [0.45] * 6 + [1.0] + [0.8] * 10
    kinds = ["N"] * 11 + ["V"] * 6 + ["N"] * 11
    rep, _ = _run(rr, kinds, sqi=0.3)
    assert "vt_run" in _codes(rep)
    assert all(f.severity != "urgent" for f in rep.findings)
    assert any("low signal quality" in w for w in rep.warnings)


def test_findings_carry_times_and_episodes():
    """Every finding gets `times`/`episodes` on the recording's time base and a 'when' text."""
    import numpy as np
    from kardia6l import dsp
    from kardia6l.arrhythmia import Finding, annotate_times, when_text
    fs = 300.0
    rp = np.array([300, 600, 900, 1200, 1500, 1800, 2100, 2400])
    f_pvc = Finding("pvc", "PVCs", "notable", 0.8, "", 2, [2, 5])
    f_big = Finding("bigeminy", "Bigeminy", "notable", 0.6, "", 3, [1, 3, 5])
    f_pause = Finding("pause", "Pause", "notable", 0.5, "", 1, [4])
    f_flag = Finding("long_qtc", "Long QTc", "notable", 0.5, "", 0, [])
    annotate_times([f_pvc, f_big, f_pause, f_flag], rp, fs, t0=10.0)
    assert f_pvc.times == [13.0, 16.0] and f_pvc.episodes == [(13.0, 13.0), (16.0, 16.0)]
    assert when_text(f_pvc).startswith("at 13.0 s, 16.0 s")
    assert len(f_big.episodes) == 1 and f_big.episodes[0][0] < 12.0 < 16.0 < f_big.episodes[0][1]
    assert when_text(f_big).startswith("episodes ")
    assert f_pause.episodes == [(14.0, 15.0)]          # previous beat -> beat ending the pause
    assert f_flag.times == [] and when_text(f_flag) == "whole recording"
    assert all(isinstance(v, float) for v in f_pvc.times)


def test_af_episodes_localize_af_stretch():
    from kardia6l.af import af_episodes
    rng = np.random.default_rng(3)
    rr = np.concatenate([np.full(30, 850.0) + rng.normal(0, 12, 30),          # sinus 0..25 s
                         rng.uniform(450, 1150, 45),                            # AF-like ~35 s
                         np.full(30, 850.0) + rng.normal(0, 12, 30)])          # sinus
    t = np.concatenate([[0.0], np.cumsum(rr) / 1000.0])
    eps = af_episodes(rr, t, p_consistent=None)
    assert eps, "expected at least one AF episode"
    a, b, sc = eps[0]
    af_start, af_end = t[30], t[75]
    assert abs(a - af_start) < 8.0 and abs(b - af_end) < 8.0 and sc >= 0.5, (a, b, af_start, af_end)
    assert af_episodes(rr[:5], t[:6]) == []


def test_beat_in_onset_transient_is_never_ventricular():
    """A gated recording often starts 0.1–0.3 s before a beat; the clipped, ringing edge
    must not be labelled a PVC (seen in the field: 'PVC at 0.1 s, corr -0.07')."""
    from kardia6l import dsp
    from kardia6l.arrhythmia import detect_arrhythmias
    fs = 300.0
    x = dsp.synthetic_ecg(20.0, fs, hr_bpm=70.0, noise_uv=15.0, seed=5)
    # find the sample of an R peak and cut the record 0.1 s before it, adding a DC step
    peaks = dsp.detect_r_peaks(x, fs)
    cut = int(peaks[3] - 0.1 * fs)
    y = x[cut:] + 0.8                     # offset -> band-pass ringing at the onset
    c = dsp.clean(y, fs)
    pk = dsp.detect_r_peaks(c, fs)
    assert pk[0] < 0.35 * fs
    rep = detect_arrhythmias(c, fs, pk)
    assert rep.beat_labels[0] != "V"
    assert not any(f.code == "pvc" for f in rep.findings)
