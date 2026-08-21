"""JSON-lines firehose to a stream (stdout by default): one object per notification
batch: {"t0": first sample time, "fs": fs, "leads": [...], "mv": [[...], ...]}."""
from __future__ import annotations

import json
import sys

import numpy as np


class JSONLSink:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.fs = None
        self.names = None

    def open(self, fs: float, channels: list[str]) -> None:
        self.fs, self.names = fs, channels
        self.stream.write(json.dumps({"event": "start", "fs": fs, "leads": channels,
                                      "disclaimer": "research signal, not a medical device"}) + "\n")

    def write(self, t: np.ndarray, x: np.ndarray) -> None:
        self.stream.write(json.dumps({"t0": float(t[0]), "n": int(len(t)),
                                      "mv": np.round(x, 5).tolist()}) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.write(json.dumps({"event": "stop"}) + "\n")
        self.stream.flush()
