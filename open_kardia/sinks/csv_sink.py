"""CSV sink: one row per sample, columns t_s + one per lead (mV)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class CSVSink:
    def __init__(self, path: str | Path, flush_every: int = 500):
        self.path = Path(path)
        self.flush_every = flush_every
        self._fh = None
        self._w = None
        self._n = 0

    def open(self, fs: float, channels: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._fh.write(f"# open_kardia research capture — NOT a medical device; fs={fs} Hz; units=mV\n")
        self._w.writerow(["t_s", *channels])

    def write(self, t: np.ndarray, x: np.ndarray) -> None:
        for ti, row in zip(t, x):
            self._w.writerow([f"{ti:.6f}", *(f"{v:.5f}" for v in row)])
        self._n += len(t)
        if self._n % self.flush_every < len(t):
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def read_csv(path: str | Path) -> tuple[float, list[str], np.ndarray, np.ndarray]:
    """Read a CSVSink file -> (fs, names, t, x[n, ch])."""
    fs = None
    with open(path) as fh:
        first = fh.readline()
        if first.startswith("#") and "fs=" in first:
            fs = float(first.split("fs=")[1].split()[0])
        else:
            fh.seek(0)
        r = csv.reader(fh)
        header = next(r)
        rows = [list(map(float, row)) for row in r if row]
    a = np.array(rows)
    t, x = a[:, 0], a[:, 1:]
    if fs is None and len(t) > 1:
        fs = 1.0 / np.median(np.diff(t))
    return fs, header[1:], t, x
