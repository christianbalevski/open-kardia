"""Async pipeline: BLE notify -> queue -> Decoder -> ring buffer -> sinks.

Keep the BLE callback minimal (append only). All decoding/DSP happens in the
consumer task.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .protocol import Decoder, FrameFormat

log = logging.getLogger("kardia6l.stream")


class Sink(Protocol):
    def open(self, fs: float, channels: list[str]) -> None: ...
    def write(self, t: np.ndarray, samples: np.ndarray) -> None: ...   # (n,), (n, ch) mV
    def close(self) -> None: ...


class RingBuffer:
    """Fixed-capacity buffer of (n, ch) float samples with monotonically increasing
    sample index. `latest(n)` returns the last n samples (copy)."""

    def __init__(self, capacity: int, channels: int):
        self.cap = capacity
        self.ch = channels
        self.buf = np.zeros((capacity, channels))
        self.t = np.zeros(capacity)
        self.n_written = 0

    def append(self, t: np.ndarray, x: np.ndarray) -> None:
        n = len(x)
        if n == 0:
            return
        if n >= self.cap:
            x, t = x[-self.cap:], t[-self.cap:]
            n = self.cap
        i = self.n_written % self.cap
        end = i + n
        if end <= self.cap:
            self.buf[i:end] = x
            self.t[i:end] = t
        else:
            k = self.cap - i
            self.buf[i:] = x[:k]
            self.t[i:] = t[:k]
            self.buf[: n - k] = x[k:]
            self.t[: n - k] = t[k:]
        self.n_written += n

    def latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        n = min(n, self.n_written, self.cap)
        if n == 0:
            return np.zeros(0), np.zeros((0, self.ch))
        end = self.n_written % self.cap
        idx = (np.arange(end - n, end)) % self.cap
        return self.t[idx].copy(), self.buf[idx].copy()


@dataclass
class StreamStats:
    notifications: int = 0
    samples: int = 0
    dropped_queue: int = 0
    started: float = 0.0
    last_t: float = 0.0
    seq_gaps: int = 0
    stalls: int = 0
    rate_est: float = 0.0
    first_t: float = 0.0
    first_n: int = 0
    sizes: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, fmt: FrameFormat, fs: float, sinks: list[Sink], channel_names: list[str],
                 ring_seconds: float = 30.0, queue_max: int = 10000, derive_leads: bool = False,
                 raw_log: str | None = None):
        self.fmt = fmt
        self._raw = open(raw_log, "a") if raw_log else None
        self.fs = fs
        self.decoder = Decoder(fmt, to_mv=True)
        self.sinks = sinks
        self.channel_names = channel_names
        self.derive = derive_leads
        n_out = 6 if (derive_leads and fmt.channels >= 2) else fmt.channels
        self.ring = RingBuffer(int(ring_seconds * fs), n_out)
        self.q: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self.stats = StreamStats()
        self._task: asyncio.Task | None = None
        self._sample_index = 0
        self._t0: float | None = None
        self._stop = False

    # BLE thread/loop callback — must be cheap.
    def on_notify(self, char_uuid: str, t: float, payload: bytes) -> None:
        if self._raw is not None:
            import json
            self._raw.write(json.dumps({"t": time.time(), "uuid": char_uuid, "len": len(payload),
                                        "hex": payload.hex()}) + "\n")
        try:
            self.q.put_nowait((t, payload))
        except asyncio.QueueFull:
            self.stats.dropped_queue += 1

    def start(self):
        out_names = self.channel_names
        if self.derive and self.fmt.channels >= 2:
            from .leads import LEAD_NAMES_6
            out_names = list(LEAD_NAMES_6)
        for s in self.sinks:
            s.open(self.fs, out_names)
        self.stats.started = time.monotonic()
        self._task = asyncio.create_task(self._consume())

    async def _consume(self):
        while not self._stop:
            t, payload = await self.q.get()
            if payload is None:
                break
            self._handle(t, payload)

    def _handle(self, t_arrival: float, payload: bytes):
        st = self.stats
        st.notifications += 1
        st.sizes[len(payload)] = st.sizes.get(len(payload), 0) + 1
        gaps_before = self.decoder.stats.gaps
        x = self.decoder.decode(payload)            # (n, ch) mV
        st.seq_gaps = self.decoder.stats.gaps
        n = len(x)
        if n == 0:
            return
        if self._t0 is None:
            self._t0 = t_arrival
        # Timestamps derived from the sample counter (BLE arrival jitter is large);
        # if a sequence gap was detected we advance the index by the estimated loss.
        if self.decoder.stats.gaps > gaps_before:
            self._sample_index += self.decoder.stats.lost_frames_est * n  # rough: frames*samples/frame
        idx = np.arange(self._sample_index, self._sample_index + n)
        t = idx / self.fs
        self._sample_index += n
        if st.samples == 0:
            st.first_t, st.first_n = t_arrival, n
        st.samples += n
        st.last_t = t_arrival
        el = t_arrival - st.first_t
        st.rate_est = (st.samples - st.first_n) / el if el > 0 else 0.0

        if self.derive and x.shape[1] >= 2:
            from .leads import derive_leads
            L = derive_leads(x[:, 0], x[:, 1])
            x = L.as_matrix()
        self.ring.append(t, x)
        for s in self.sinks:
            try:
                s.write(t, x)
            except Exception as e:  # noqa: BLE001
                log.exception("sink %s failed: %s", type(s).__name__, e)

    async def stop(self):
        self._stop = True
        await self.q.put((0.0, None))
        if self._task:
            try:
                await asyncio.wait_for(self._task, 5)
            except asyncio.TimeoutError:
                self._task.cancel()
        for s in self.sinks:
            try:
                s.close()
            except Exception as e:  # noqa: BLE001
                log.exception("sink close failed: %s", e)
        if self._raw is not None:
            self._raw.close()
