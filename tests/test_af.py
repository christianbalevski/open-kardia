"""Tests for kardia6l.af (RR-based AF screening). No network, synthetic RR series only."""
import json
import math

import numpy as np
import pytest

from kardia6l import af


def _sinus(n=40, rr=850.0, jitter=15.0, seed=0):
    return rr + np.random.default_rng(seed).normal(0, jitter, n)


def _sinus_arrhythmia(n=40, rr=850.0, depth=0.10, breath_hz=0.2, seed=1):
    t = np.cumsum(np.full(n, rr / 1000.0))
    return rr * (1 + depth * np.sin(2 * np.pi * breath_hz * t)) + np.random.default_rng(seed).normal(0, 5, n)


def _af(n=40, seed=2, lo=500.0, hi=1100.0):
    return np.random.default_rng(seed).uniform(lo, hi, n)


def _af_lognormal(n=40, seed=3):
    return np.exp(np.random.default_rng(seed).normal(np.log(700.0), 0.2, n))


def test_regular_sinus_not_af():
    r = af.af_score(_sinus())
    assert not r.is_af
    assert r.label == "AF not suggested"
    assert r.score < 0.2
    assert r.confidence > 0.5


def test_sinus_arrhythmia_not_af():
    for hz in (0.15, 0.2, 0.25, 0.3):
        r = af.af_score(_sinus_arrhythmia(breath_hz=hz))
        assert not r.is_af, (hz, r.score, r.evidence)
        assert r.score < 0.5


@pytest.mark.parametrize("rr", [_af(seed=2), _af(seed=7), _af_lognormal(seed=3), _af_lognormal(seed=9)])
def test_af_like_is_af(rr):
    r = af.af_score(rr)
    assert r.is_af
    assert r.score > 0.7
    assert r.label == "Possible atrial fibrillation"
    assert "irregularly irregular" in r.evidence
    assert "COSEn" in r.evidence and "nRMSSD" in r.evidence and "TPR" in r.evidence


def test_bigeminy_not_af_and_evidence_mentions_pattern():
    rr = np.tile([600.0, 1000.0], 20) + np.random.default_rng(4).normal(0, 8, 40)
    r = af.af_score(rr)
    assert not r.is_af
    assert r.score <= 0.3
    assert r.metrics["regularly_irregular"] is True
    assert "regularly irregular" in r.evidence.lower() and "bigeminy" in r.evidence.lower()
    assert any("regularly irregular" in w for w in r.warnings)
    # trigeminy too
    r3 = af.af_score(np.tile([700.0, 700.0, 1100.0], 14))
    assert not r3.is_af


def test_frequent_random_pvcs_not_af():
    # sinus 800 ms with random premature beats (coupling 480 ms) + compensatory pauses
    rr = [800.0] * 45
    for i in (3, 9, 16, 24, 31, 38):
        rr[i], rr[i + 1] = 480.0, 1120.0
    r = af.af_score(np.array(rr) + np.random.default_rng(5).normal(0, 6, len(rr)))
    assert not r.is_af, (r.score, r.evidence)
    assert r.metrics["ectopic_pairs"] >= 5


def test_too_few_rr_undetermined():
    r = af.af_score([800.0] * 11)
    assert r.label == "Undetermined"
    assert not r.is_af
    assert math.isnan(r.score)
    assert any("too few" in w for w in r.warnings)


@pytest.mark.parametrize("bad", [[], None, "garbage", [np.nan] * 30, [np.inf, -np.inf] * 10,
                                 [0.0] * 30, [1e9] * 30, np.zeros((5, 5)), object()])
def test_garbage_input_no_exception(bad):
    r = af.af_score(bad)
    assert isinstance(r, af.AFResult)
    assert r.label == "Undetermined"
    assert not r.is_af
    json.dumps(r.metrics)  # nan is JSON-serialisable via allow_nan (default)


def test_p_wave_evidence_shifts_score():
    rr = _af(seed=11)
    s_true = af.af_score(rr, p_consistent=True).score
    s_none = af.af_score(rr, p_consistent=None).score
    s_false = af.af_score(rr, p_consistent=False).score
    assert s_true < s_none <= s_false
    # borderline series: consistent P waves should flip the call
    rr_b = 800 + np.random.default_rng(12).normal(0, 60, 40)
    r_none, r_p = af.af_score(rr_b), af.af_score(rr_b, p_consistent=True)
    assert r_p.score < r_none.score
    assert "consistent P waves" in r_p.evidence
    assert "P waves not detected" in af.af_score(rr_b, p_consistent=False).evidence


def test_low_sqi_undetermined_unless_overwhelming():
    r = af.af_score(_sinus(), sqi=0.3)
    assert r.label == "Undetermined" and not r.is_af
    r2 = af.af_score(_af(seed=7), sqi=0.4)          # overwhelming AF-like -> still called
    assert r2.is_af
    r3 = af.af_score(_af(seed=7), sqi=0.1)
    assert r3.label == "Undetermined"


def test_rr_outliers_dropped_and_counted():
    rr = list(_sinus(n=30))
    rr[5], rr[20] = 150.0, 3200.0                   # doubled / missed detections
    r = af.af_score(rr)
    assert r.metrics["n_rr_dropped"] == 2
    assert r.metrics["n_rr"] == 28
    assert not r.is_af
    assert any("dropped" in w for w in r.warnings)


def test_rr_features_finite_and_json_safe():
    m = af.rr_features(_sinus())
    for k in ("n_rr", "mean_rr_ms", "rr_cv", "rmssd_ms", "nrmssd", "pnn50", "tpr", "acf1",
              "poincare_sd1_ms", "poincare_sd2_ms", "poincare_sd1_sd2", "sampen", "cosen",
              "resid_cosen", "resid_dev_frac"):
        assert k in m, k
        assert isinstance(m[k], (int, float)) and math.isfinite(m[k]), (k, m[k])
    assert m["n_rr"] == 40
    assert m["cosen"] < af.COSEN_THRESHOLD           # regular -> low entropy
    assert af.rr_features(_af())["cosen"] > af.COSEN_THRESHOLD
    assert isinstance(m["regularly_irregular"], bool)
    txt = json.dumps(m)                                # plain python types only
    assert json.loads(txt)["n_rr"] == 40
    assert not any(isinstance(v, np.generic) for v in m.values())


def test_cosen_unit_invariance_and_short_series():
    rr = _af(seed=3)
    c_ms = af.rr_features(rr)["cosen"]
    # 12-beat classic window works on its own
    m12 = af.rr_features(rr[:12])
    assert math.isfinite(m12["cosen"])
    assert math.isfinite(c_ms)
    # constant series: SampEn ~0 -> COSEn = ln(2r/meanRR)
    m = af.rr_features(np.full(20, 800.0))
    assert m["cosen"] == pytest.approx(math.log(60.0 / 800.0), abs=1e-6)


def test_metrics_include_af_evidence_score():
    r = af.af_score(_af())
    assert 0.0 <= r.metrics["af_evidence_score"] <= 1.0
    r2 = af.af_score(_af(), p_consistent=True)
    assert r2.metrics["af_evidence_score"] == pytest.approx(r.metrics["af_evidence_score"])
