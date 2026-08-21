"""Tests for the on-disk reading library."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from open_kardia import dsp
from open_kardia.library import INDEX_NAME, ReadingLibrary, ReadingRecord

SIX = ["I", "II", "III", "aVR", "aVL", "aVF"]


def _six_lead(secs=8.0, fs=300.0, hr=72.0):
    lead1 = dsp.synthetic_ecg(secs, fs, hr_bpm=hr, noise_uv=15.0, seed=1)
    lead2 = dsp.synthetic_ecg(secs, fs, hr_bpm=hr, noise_uv=15.0, seed=2, amplitude_mv=1.3)
    III = lead2 - lead1
    aVR = -(lead1 + lead2) / 2
    aVL = lead1 - lead2 / 2
    aVF = lead2 - lead1 / 2
    return np.column_stack([lead1, lead2, III, aVR, aVL, aVF]), fs


def _single_lead(secs=6.0, fs=300.0, hr=60.0):
    return dsp.synthetic_ecg(secs, fs, hr_bpm=hr, noise_uv=10.0, seed=3)[:, None], fs


def test_add_list_newest_first_and_fields(tmp_path):
    lib = ReadingLibrary(tmp_path)
    x6, fs = _six_lead(secs=8.0, hr=72.0)
    r1 = lib.add(x6, fs, SIX, name="first six", meta={"source": "sim"})
    x1, fs1 = _single_lead(secs=6.0, hr=60.0)
    r2 = lib.add(x1, fs1, ["I"], name="single", meta={"source": "live"})

    recs = lib.list()
    assert [r.id for r in recs] == sorted([r1.id, r2.id], reverse=True)
    # newest is the last added (ids are timestamp+collision slugs, monotonic)
    assert recs[0].id == r2.id

    six = lib.get(r1.id)
    assert six.n_leads == 6
    assert six.lead_names == SIX
    assert six.duration_s == pytest.approx(8.0, abs=0.02)
    assert six.hr_bpm == pytest.approx(72.0, abs=2.0)
    assert not math.isnan(six.sqi)
    assert six.extra.get("source") == "sim"

    one = lib.get(r2.id)
    assert one.n_leads == 1 and one.lead_names == ["I"]
    assert one.duration_s == pytest.approx(6.0, abs=0.02)


def test_load_roundtrips_signal(tmp_path):
    lib = ReadingLibrary(tmp_path)
    x6, fs = _six_lead()
    rec = lib.add(x6, fs, SIX)
    fs2, names, x2 = lib.load(rec.id)
    assert fs2 == pytest.approx(fs)
    assert names == SIX
    assert x2.shape == x6.shape
    np.testing.assert_allclose(x2, x6, atol=1e-3)


def test_hr_and_sqi_from_meta_override(tmp_path):
    lib = ReadingLibrary(tmp_path)
    x1, fs = _single_lead()
    rec = lib.add(x1, fs, ["I"], meta={"hr_bpm": 55.5, "sqi": 0.99, "rhythm": "Normal Sinus"})
    assert rec.hr_bpm == pytest.approx(55.5)
    assert rec.sqi == pytest.approx(0.99)
    assert rec.rhythm == "Normal Sinus"


def test_persist_across_instances(tmp_path):
    lib = ReadingLibrary(tmp_path)
    x1, fs = _single_lead()
    rec = lib.add(x1, fs, ["I"], name="orig")
    lib.rename(rec.id, "renamed")
    lib.set_note(rec.id, "looks clean")

    fresh = ReadingLibrary(tmp_path)          # must read from index.json / meta.json
    got = fresh.get(rec.id)
    assert got.name == "renamed"
    assert got.note == "looks clean"
    # the on-disk meta.json reflects the change too
    meta = json.loads((tmp_path / rec.id / "meta.json").read_text())
    assert meta["name"] == "renamed" and meta["note"] == "looks clean"


def test_delete_removes_folder_and_index_entry(tmp_path):
    lib = ReadingLibrary(tmp_path)
    a = lib.add(*_single_lead(), lead_names=["I"], name="a")
    b = lib.add(*_six_lead(), lead_names=SIX, name="b")
    assert lib.delete(a.id) is True
    assert not (tmp_path / a.id).exists()
    ids = [r.id for r in lib.list()]
    assert a.id not in ids and b.id in ids
    assert lib.get(a.id) is None
    # deleting a non-existent id is a no-op returning False
    assert lib.delete("nope-does-not-exist") is False


def test_rebuild_index_after_deleting_it(tmp_path):
    lib = ReadingLibrary(tmp_path)
    r1 = lib.add(*_single_lead(), lead_names=["I"], name="x")
    r2 = lib.add(*_six_lead(), lead_names=SIX, name="y")
    idx = tmp_path / INDEX_NAME
    assert idx.exists()
    idx.unlink()

    # list() must transparently rebuild the index from the folders
    recs = ReadingLibrary(tmp_path).list()
    assert {r.id for r in recs} == {r1.id, r2.id}
    assert idx.exists()  # rebuilt and re-cached

    # explicit rebuild also works and matches
    lib.rebuild_index()
    assert {r.id for r in ReadingLibrary(tmp_path).list()} == {r1.id, r2.id}


def test_corrupt_index_is_rebuilt(tmp_path):
    lib = ReadingLibrary(tmp_path)
    r1 = lib.add(*_six_lead(), lead_names=SIX, name="c")
    (tmp_path / INDEX_NAME).write_text("{ this is not valid json ]")
    recs = ReadingLibrary(tmp_path).list()
    assert [r.id for r in recs] == [r1.id]


def test_record_paths(tmp_path):
    lib = ReadingLibrary(tmp_path)
    rec = lib.add(*_single_lead(), lead_names=["I"])
    assert rec.csv_path().endswith("signal.csv")
    assert rec.wfdb_path().endswith("signal.hea")
    assert rec.meta_path().endswith("meta.json")
    from pathlib import Path
    assert Path(rec.csv_path()).exists()
    assert Path(rec.meta_path()).exists()


def test_readingrecord_nan_json_roundtrip():
    rec = ReadingRecord(
        id="20260101-000000", name="n", created_at="2026-01-01T00:00:00+00:00",
        duration_s=5.0, fs=300.0, lead_names=["I"], n_leads=1,
        hr_bpm=math.nan, rhythm=None, sqi=math.nan, note="", dir="/tmp/x",
    )
    d = rec.to_json()
    assert d["hr_bpm"] is None and d["sqi"] is None
    back = ReadingRecord.from_json(d)
    assert math.isnan(back.hr_bpm) and math.isnan(back.sqi)
    assert back.rhythm is None
