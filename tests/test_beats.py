"""Beat timing for the animated heart (open_kardia.beats)."""
import json
import math

import numpy as np
import pytest

from open_kardia import beats, dsp

FS = 300.0


def _ecg(hr=72.0, secs=20.0):
    return dsp.synthetic_ecg(secs, FS, hr_bpm=hr, noise_uv=15.0, seed=1)


def test_default_fiducials_are_ordered_and_rate_scaled():
    slow = beats.default_fiducials(1.2)     # 50 bpm
    fast = beats.default_fiducials(0.5)     # 120 bpm
    for f in (slow, fast, beats.default_fiducials(None), beats.default_fiducials(float("nan"))):
        assert f["p_on"] < f["p_off"] < f["q_on"] < 0.0 < f["q_off"] < f["t_on"] < f["t_off"]
    # QT shortens with rate (inverse Bazett)
    assert (fast["t_off"] - fast["q_on"]) < (slow["t_off"] - slow["q_on"])


def test_beat_timing_finds_beats_and_hr_on_synthetic():
    x = _ecg(72.0)
    t = beats.beat_timing(x, FS, t0=100.0)          # arbitrary time base
    assert abs(t["hr"] - 72.0) < 2.0
    b = np.array(t["beats"])
    assert len(b) >= 20
    assert b[0] >= 100.0                             # honours t0
    rr = np.diff(b)
    assert np.allclose(rr, 60.0 / 72.0, atol=0.02)
    f = t["fid"]
    assert f["p_on"] < f["p_off"] < f["q_on"] < 0.0 < f["q_off"] < f["t_on"] < f["t_off"]
    assert t["fid_source"] in ("delineated", "default")


def test_beat_timing_short_signal_falls_back_to_defaults():
    t = beats.beat_timing(np.zeros(100), FS)
    assert t["beats"] == [] and t["fid_source"] == "default"
    assert math.isnan(t["hr"])


def test_beat_timing_prev_fid_is_reused_when_not_delineating():
    x = _ecg(60.0, 10.0)
    prev = {"p_on": -0.2, "p_off": -0.1, "q_on": -0.04, "q_off": 0.05, "t_on": 0.2, "t_off": 0.38}
    t = beats.beat_timing(x, FS, delineate=False, prev_fid=prev)
    assert t["fid"] == prev and t["fid_source"] == "delineated"


def test_sane_rejects_impossible_delineation():
    ok = {"p_on": -0.2, "p_off": -0.1, "q_on": -0.04, "q_off": 0.05, "t_on": 0.2, "t_off": 0.38}
    assert beats._sane(ok)
    bad = dict(ok, t_off=0.1)                       # T ends before it starts
    assert not beats._sane(bad)
    assert not beats._sane(dict(ok, p_on=float("nan")))
    assert not beats._sane({"p_on": -0.2})


def test_primary_lead_index():
    assert beats.primary_lead_index(["I", "II", "III"]) == 1
    assert beats.primary_lead_index(["I"]) == 0
    assert beats.primary_lead_index(["aVR", "aVL"]) == 0


def test_playback_bundle_is_json_safe_and_consistent():
    x1 = _ecg(70.0, 15.0)
    x = np.column_stack([x1, x1 * 1.3])
    b = beats.playback_bundle(x, FS, ["I", "II"])
    s = json.dumps(b)                                # no NaN/inf, no numpy types
    assert "NaN" not in s
    assert b["leads"] == ["I", "II"] and b["primary"] == "II"
    assert len(b["data"]) == 2 and len(b["data"][0]) == len(x1)
    assert abs(b["duration_s"] - 15.0) < 1e-9
    assert len(b["beats"]) >= 15 and all(0 <= t <= 15 for t in b["beats"])
    assert abs(b["hr"] - 70.0) < 2.0


def test_playback_bundle_single_lead_1d():
    b = beats.playback_bundle(_ecg(80.0, 8.0), FS, ["I"])
    assert b["leads"] == ["I"] and b["primary"] == "I" and len(b["data"]) == 1
