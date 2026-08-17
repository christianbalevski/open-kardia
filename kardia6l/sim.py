"""Synthetic 6L stand-in: emits notifications in a given FrameFormat at the given
sample rate, so the pipeline/sinks/CLI can be exercised without hardware."""
from __future__ import annotations

import asyncio
import time

import numpy as np

from .dsp import synthetic_ecg
from .protocol import FrameFormat, encode


class SimDevice:
    def __init__(self, fmt: FrameFormat, fs: float, hr_bpm: float = 72.0, per_notification: int = 25,
                 gain_mv: float | None = None, seconds: float = 3600.0, drop_every: int = 0):
        self.fmt = fmt
        self.fs = fs
        self.per = per_notification
        self.drop_every = drop_every
        gain = gain_mv or (fmt.gain_mv if fmt.gain_mv != 1.0 else 0.0025)
        ecg1 = synthetic_ecg(seconds, fs, hr_bpm=hr_bpm, noise_uv=20.0)
        cols = [np.round(ecg1 / gain + fmt.offset_counts)]
        if fmt.channels >= 2:
            ecg2 = synthetic_ecg(seconds, fs, hr_bpm=hr_bpm, noise_uv=20.0, seed=7, amplitude_mv=1.4)
            cols.append(np.round(ecg2 / gain + fmt.offset_counts))
        for _ in range(fmt.channels - 2):
            cols.append(np.zeros_like(cols[0]))
        counts = np.column_stack(cols).astype(int)
        if not fmt.signed:
            counts += 0  # caller should set offset_counts for unsigned formats
        self.packets = encode(counts, fmt, per_notification)
        self.uuid = "sim-ecg-characteristic"

    async def run(self, cb):
        interval = self.per / self.fs
        t_next = time.monotonic()
        for i, p in enumerate(self.packets):
            t_next += interval
            await asyncio.sleep(max(0.0, t_next - time.monotonic()))
            if self.drop_every and i % self.drop_every == self.drop_every - 1:
                continue
            cb(self.uuid, time.monotonic(), p)
