"""Live scrolling plot (matplotlib). Runs the matplotlib event loop cooperatively
from the asyncio loop via periodic `pump()` calls (see cli.stream)."""
from __future__ import annotations

import numpy as np


class PlotSink:
    def __init__(self, seconds: float = 6.0, leads: list[str] | None = None, mains: float = 60.0,
                 filtered: bool = True):
        self.seconds = seconds
        self.want = leads
        self.mains = mains
        self.filtered = filtered
        self.fs = None
        self.names: list[str] = []
        self._buf = None
        self._fig = None
        self._lines = []
        self._axes = []
        self._dirty = False

    def open(self, fs: float, channels: list[str]) -> None:
        import matplotlib
        try:
            matplotlib.use("MacOSX")
        except Exception:  # noqa: BLE001
            pass
        import matplotlib.pyplot as plt
        self.fs = fs
        self.names = [c for c in channels if (self.want is None or c in self.want)]
        self._idx = [channels.index(c) for c in self.names]
        n = int(self.seconds * fs)
        self._buf = np.full((n, len(self.names)), np.nan)
        plt.ion()
        self._fig, axes = plt.subplots(len(self.names), 1, sharex=True,
                                       figsize=(10, 2.2 * len(self.names)))
        self._axes = list(np.atleast_1d(axes))
        t = np.arange(n) / fs
        for ax, name in zip(self._axes, self.names):
            (ln,) = ax.plot(t, self._buf[:, 0], lw=0.8)
            self._lines.append(ln)
            ax.set_ylabel(f"{name} (mV)")
            ax.set_ylim(-2, 2)
            ax.grid(True, alpha=0.3)
        self._axes[-1].set_xlabel("s")
        self._fig.suptitle("open_kardia live — research signal, NOT a medical device", fontsize=9)
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def write(self, t: np.ndarray, x: np.ndarray) -> None:
        x = x[:, self._idx]
        n = len(x)
        if n >= len(self._buf):
            self._buf[:] = x[-len(self._buf):]
        else:
            self._buf = np.roll(self._buf, -n, axis=0)
            self._buf[-n:] = x
        self._dirty = True

    def pump(self) -> None:
        """Redraw if new data; call from the asyncio loop ~20 Hz."""
        if not self._dirty or self._fig is None:
            return
        self._dirty = False
        data = self._buf
        if self.filtered and np.isfinite(data).all():
            from ..dsp import clean
            data = np.column_stack([clean(data[:, i], self.fs, self.mains) for i in range(data.shape[1])])
        for i, ln in enumerate(self._lines):
            ln.set_ydata(data[:, i])
            col = data[:, i]
            if np.isfinite(col).any():
                lo, hi = np.nanpercentile(col, [1, 99])
                pad = max(0.2, 0.2 * (hi - lo))
                self._axes[i].set_ylim(lo - pad, hi + pad)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        pass
