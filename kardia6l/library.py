"""On-disk library of saved ECG readings.

A small filesystem-backed store the desktop app uses to save, list, reload, rename,
annotate, and delete captured 6-lead (or single-lead) readings. Pure Python: stdlib +
numpy + this package's own sinks. No GUI, no Bluetooth.

Layout (default root ``~/.kardia6l/readings/``)::

    <root>/
        index.json                 # cached list of records (rebuildable)
        20260816-175712/
            signal.csv             # via kardia6l.sinks.csv_sink.CSVSink
            signal.dat / signal.hea  # via kardia6l.sinks.wfdb_sink.WFDBSink
            meta.json              # the ReadingRecord, serialised

Each reading id is a UTC timestamp slug (``YYYYMMDD-HHMMSS``) with ``-2``/``-3`` suffixes
on collision. ``index.json`` is a convenience cache; if it is missing or corrupt it is
rebuilt by scanning the ``meta.json`` of every sub-folder.

NOTE: this is a research tool. A saved reading is NOT a medical record and must not be used
for diagnosis.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import dsp
from .sinks.csv_sink import CSVSink, read_csv
from .sinks.wfdb_sink import WFDBSink

DEFAULT_ROOT = Path.home() / ".kardia6l" / "readings"
INDEX_NAME = "index.json"
CSV_NAME = "signal.csv"
WFDB_STEM = "signal"
META_NAME = "meta.json"


def _num(v) -> float:
    """JSON-safe float: NaN -> None on dump, None/NaN-tolerant on load."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


@dataclass
class ReadingRecord:
    id: str
    name: str                       # user-facing label; default = human date/time
    created_at: str                 # ISO 8601 (UTC)
    duration_s: float
    fs: float
    lead_names: list[str]
    n_leads: int
    hr_bpm: float                   # nan if unknown
    rhythm: str | None
    sqi: float                      # nan if unknown
    note: str
    dir: str                        # absolute path to the reading folder
    extra: dict = field(default_factory=dict)  # any extra meta the caller passed through

    # ---- paths -------------------------------------------------------------
    def csv_path(self) -> str:
        return str(Path(self.dir) / CSV_NAME)

    def wfdb_path(self) -> str:
        """Path to the WFDB header (.hea); the .dat sits beside it."""
        return str(Path(self.dir) / f"{WFDB_STEM}.hea")

    def meta_path(self) -> str:
        return str(Path(self.dir) / META_NAME)

    # ---- (de)serialisation -------------------------------------------------
    def to_json(self) -> dict:
        d = asdict(self)
        # NaN is not valid JSON; store as null and restore on load.
        for k in ("hr_bpm", "sqi"):
            if d[k] is None or (isinstance(d[k], float) and math.isnan(d[k])):
                d[k] = None
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ReadingRecord":
        d = dict(d)
        d.setdefault("extra", {})
        d["hr_bpm"] = _num(d.get("hr_bpm"))
        d["sqi"] = _num(d.get("sqi"))
        if d.get("rhythm") == "":
            d["rhythm"] = None
        known = {f for f in cls.__dataclass_fields__}  # noqa: E501
        return cls(**{k: v for k, v in d.items() if k in known})


class ReadingLibrary:
    """Manage a folder of saved readings."""

    def __init__(self, root: str | os.PathLike | None = None):
        self._root = Path(root) if root is not None else DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # ---- ids ---------------------------------------------------------------
    def _new_id(self) -> str:
        base = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rid = base
        n = 2
        while (self._root / rid).exists():
            rid = f"{base}-{n}"
            n += 1
        return rid

    # ---- add ---------------------------------------------------------------
    def add(self, x, fs, lead_names, meta: dict | None = None,
            name: str | None = None) -> ReadingRecord:
        """Persist a reading and return its record.

        x: (n_samples, n_leads) mV. meta may carry hr_bpm/rhythm/sqi/note/source/etc.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        n_samples, n_leads = x.shape
        lead_names = list(lead_names)
        if len(lead_names) != n_leads:
            raise ValueError(f"lead_names has {len(lead_names)} entries but x has {n_leads} columns")
        meta = dict(meta or {})
        fs = float(fs)
        duration_s = n_samples / fs if fs else 0.0

        # HR / SQI: use provided values, else derive from lead 0.
        hr = _num(meta.get("hr_bpm", math.nan))
        sqi = _num(meta.get("sqi", math.nan))
        mains = float(meta.get("mains", 60.0))
        enough = bool(fs) and n_samples >= int(2 * fs)
        if enough:
            if math.isnan(hr):
                try:
                    hr = float(dsp.heart_rate(x[:, 0], fs))
                except Exception:  # noqa: BLE001
                    hr = math.nan
            if math.isnan(sqi):
                try:
                    q = dsp.signal_quality(x[:, 0], fs)
                    sqi = float(q.sqi)
                    if "readable" not in meta:
                        meta["readable"] = bool(q.readable)
                except Exception:  # noqa: BLE001
                    sqi = math.nan

        rid = self._new_id()
        folder = self._root / rid
        folder.mkdir(parents=True, exist_ok=True)
        created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        if name:
            label = name
        else:
            try:
                label = datetime.now().strftime("%b %d, %Y %I:%M %p")
            except Exception:  # noqa: BLE001
                label = rid

        # write signals
        t = np.arange(n_samples) / fs if fs else np.arange(n_samples, dtype=float)
        try:
            cs = CSVSink(folder / CSV_NAME)
            cs.open(fs, lead_names)
            cs.write(t, x)
            cs.close()
        except Exception:  # noqa: BLE001 - CSV is the source of truth; re-raise
            raise
        try:
            ws = WFDBSink(folder / WFDB_STEM)
            ws.open(fs, lead_names)
            ws.write(t, x)
            ws.close()
        except Exception:  # noqa: BLE001 - WFDB optional (needs `wfdb`); keep the reading
            pass

        # strip keys we store as first-class columns out of the free-form extra bag
        first_class = {"hr_bpm", "sqi", "rhythm", "note", "duration_s"}
        extra = {k: v for k, v in meta.items() if k not in first_class}

        rec = ReadingRecord(
            id=rid,
            name=label,
            created_at=created,
            duration_s=duration_s,
            fs=fs,
            lead_names=lead_names,
            n_leads=n_leads,
            hr_bpm=hr,
            rhythm=meta.get("rhythm"),
            sqi=sqi,
            note=str(meta.get("note", "")),
            dir=str(folder.resolve()),
            extra=extra,
        )
        self._write_meta(rec)
        self._append_index(rec)
        return rec

    # ---- listing / lookup --------------------------------------------------
    def list(self) -> list[ReadingRecord]:
        """All records, newest first (by id, which is a UTC timestamp slug)."""
        recs = self._read_index()
        if recs is None:
            recs = self._scan()
            self._write_index(recs)
        recs.sort(key=lambda r: r.id, reverse=True)
        return recs

    def get(self, reading_id: str) -> ReadingRecord | None:
        for r in self.list():
            if r.id == reading_id:
                return r
        # fall back to reading meta.json directly (index may be stale)
        meta = self._root / reading_id / META_NAME
        if meta.exists():
            try:
                return ReadingRecord.from_json(json.loads(meta.read_text()))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def load(self, reading_id: str) -> tuple[float, list[str], np.ndarray]:
        """Reload the signal from CSV -> (fs, names, x[n, ch])."""
        rec = self.get(reading_id)
        if rec is None:
            raise KeyError(f"no reading {reading_id!r}")
        fs, names, _t, x = read_csv(rec.csv_path())
        return fs, names, x

    # ---- mutation ----------------------------------------------------------
    def rename(self, reading_id: str, name: str) -> ReadingRecord:
        return self._update(reading_id, name=name)

    def set_note(self, reading_id: str, note: str) -> ReadingRecord:
        return self._update(reading_id, note=note)

    def _update(self, reading_id: str, **fields) -> ReadingRecord:
        rec = self.get(reading_id)
        if rec is None:
            raise KeyError(f"no reading {reading_id!r}")
        for k, v in fields.items():
            setattr(rec, k, v)
        self._write_meta(rec)
        self.rebuild_index()
        return rec

    def delete(self, reading_id: str) -> bool:
        folder = self._root / reading_id
        existed = folder.exists()
        if existed:
            shutil.rmtree(folder, ignore_errors=True)
        # drop from index regardless
        recs = self._read_index() or self._scan()
        recs = [r for r in recs if r.id != reading_id]
        self._write_index(recs)
        return existed

    # ---- index maintenance -------------------------------------------------
    def rebuild_index(self) -> None:
        self._write_index(self._scan())

    def _scan(self) -> list[ReadingRecord]:
        out: list[ReadingRecord] = []
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            meta = child / META_NAME
            if not meta.exists():
                continue
            try:
                rec = ReadingRecord.from_json(json.loads(meta.read_text()))
                rec.dir = str(child.resolve())
                out.append(rec)
            except (json.JSONDecodeError, OSError, TypeError):
                continue
        out.sort(key=lambda r: r.id, reverse=True)
        return out

    def _index_path(self) -> Path:
        return self._root / INDEX_NAME

    def _read_index(self) -> list[ReadingRecord] | None:
        p = self._index_path()
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            recs = [ReadingRecord.from_json(d) for d in data]
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            return None
        # verify the referenced folders still exist; if any are gone, force a rescan
        for r in recs:
            if not (self._root / r.id / META_NAME).exists():
                return None
        return recs

    def _write_index(self, recs: list[ReadingRecord]) -> None:
        recs = sorted(recs, key=lambda r: r.id, reverse=True)
        self._atomic_write(self._index_path(), [r.to_json() for r in recs])

    def _append_index(self, rec: ReadingRecord) -> None:
        recs = self._read_index()
        if recs is None:
            recs = self._scan()
        else:
            recs = [r for r in recs if r.id != rec.id] + [rec]
        self._write_index(recs)

    def _write_meta(self, rec: ReadingRecord) -> None:
        self._atomic_write(Path(rec.meta_path()), rec.to_json())

    @staticmethod
    def _atomic_write(path: Path, obj) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, path)
