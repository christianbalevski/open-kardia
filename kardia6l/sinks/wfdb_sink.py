"""WFDB (.dat/.hea) and EDF writers. These buffer in memory and write on close
(both formats want to know the record length / physical ranges up front)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class _Buffered:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fs = None
        self.names: list[str] = []
        self._chunks: list[np.ndarray] = []

    def open(self, fs: float, channels: list[str]) -> None:
        self.fs, self.names = fs, list(channels)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, t: np.ndarray, x: np.ndarray) -> None:
        self._chunks.append(np.asarray(x, dtype=np.float64))

    def data(self) -> np.ndarray:
        return np.vstack(self._chunks) if self._chunks else np.zeros((0, len(self.names)))


class WFDBSink(_Buffered):
    """Writes <path>.dat + <path>.hea (format 16, adc gain 1000/mV -> µV resolution)."""

    def close(self) -> None:
        import wfdb
        x = self.data()
        if len(x) == 0:
            return
        rec = self.path.stem
        # 1000 counts/mV (1 µV resolution) unless the data would overflow int16.
        peak = float(np.max(np.abs(x))) if x.size else 1.0
        gain = 1000.0 if peak * 1000.0 < 32000 else 32000.0 / max(peak, 1e-9)
        wfdb.wrsamp(record_name=rec, fs=self.fs, units=["mV"] * x.shape[1], sig_name=self.names,
                    p_signal=x, fmt=["16"] * x.shape[1], adc_gain=[gain] * x.shape[1],
                    baseline=[0] * x.shape[1], write_dir=str(self.path.parent),
                    comments=["kardia6l research capture - NOT a medical device"])


class EDFSink(_Buffered):
    def close(self) -> None:
        import pyedflib
        x = self.data()
        if len(x) == 0:
            return
        fs = int(round(self.fs))
        # EDF wants an integer number of samples per data record; pad to a whole second.
        n = len(x)
        pad = (-n) % fs
        if pad:
            x = np.vstack([x, np.repeat(x[-1:], pad, axis=0)])
        f = pyedflib.EdfWriter(str(self.path), x.shape[1], file_type=pyedflib.FILETYPE_EDFPLUS)
        headers = []
        for name in self.names:
            headers.append({"label": name[:16], "dimension": "mV", "sample_frequency": fs,
                            "physical_max": float(max(5.0, np.max(np.abs(x)) * 1.05)),
                            "physical_min": -float(max(5.0, np.max(np.abs(x)) * 1.05)),
                            "digital_max": 32767, "digital_min": -32768,
                            "transducer": "KardiaMobile 6L (research)", "prefilter": ""})
        f.setSignalHeaders(headers)
        f.setPatientAdditional("kardia6l_research_capture_NOT_a_medical_device")
        f.writeSamples([x[:, i].copy() for i in range(x.shape[1])])
        f.close()
