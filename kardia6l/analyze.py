"""Offline analysis of a `sniff` capture (JSONL of {t, uuid, len, hex}).

Per characteristic: notification count, cadence, size histogram, byte entropy
(§5 diagnosis), and the ranked candidate byte layouts (§4.3).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .protocol import CANDIDATES, Decoder, entropy_report, rank_formats


def load_capture(path: str | Path):
    by: dict[str, list[tuple[float, bytes]]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by[r["uuid"]].append((float(r["t"]), bytes.fromhex(r["hex"])))
    return by


def analyze_capture(path, top: int = 10, char: str | None = None, fs_hint: float | None = None,
                    out_json: str | None = None, plot: bool = False):
    by = load_capture(path)
    if not by:
        sys.exit("empty capture")
    results = {}
    for uuid, items in by.items():
        if char and char.lower() not in uuid.lower():
            continue
        items.sort()
        ts = np.array([t for t, _ in items])
        pk = [p for _, p in items]
        dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        sizes = {}
        for p in pk:
            sizes[len(p)] = sizes.get(len(p), 0) + 1
        gaps = np.diff(ts) if len(ts) > 1 else np.array([])
        print(f"\n=== {uuid}")
        print(f"  notifications: {len(pk)}  span: {dur:.1f}s  rate: {len(pk) / max(dur, 1e-9):.2f}/s  "
              f"sizes: {dict(sorted(sizes.items()))}")
        if len(gaps):
            print(f"  inter-arrival ms: median={np.median(gaps) * 1e3:.1f} p95={np.percentile(gaps, 95) * 1e3:.1f} "
                  f"max={gaps.max() * 1e3:.1f}")
            bps = sum(len(p) for p in pk) / max(dur, 1e-9)
            print(f"  bytes/s: {bps:.1f}  -> if int16 x1ch: {bps / 2:.0f} Hz | int16 x2ch: {bps / 4:.0f} Hz | "
                  f"int24 x1ch: {bps / 3:.0f} Hz | int24 x2ch: {bps / 6:.0f} Hz")
        # First-bytes view helps spot headers/counters by eye.
        print("  first 6 payloads:")
        for p in pk[:6]:
            print(f"    {p.hex()}")
        rep = entropy_report(pk)
        print(f"  entropy: H={rep.shannon_bits:.3f} bits/byte  zlib_ratio={rep.compress_ratio:.3f}  "
              f"lag1_autocorr={rep.byte_lag1_autocorr:+.3f}  -> {rep.verdict.upper()}")
        res = {"n": len(pk), "duration_s": dur, "sizes": sizes, "entropy": rep.__dict__, "ranking": []}
        if rep.verdict == "ciphertext-like":
            print("  !! payload looks like ciphertext (high entropy, incompressible). If this persists after "
                  "pairing/bonding (--pair), the link is application-layer encrypted: STOP per spec §5.")
        if len(pk) >= 20 and dur > 2:
            ranked = rank_formats(pk, dur, CANDIDATES, fs_hint)
            print(f"  top {top} candidate layouts (score: higher = more ECG-like; <5 = not convincing, "
                  f">7 = clear ECG; if #1 and #2 are c1 vs c2i aliases, the app-trace correlation decides):")
            for s in ranked[:top]:
                print(f"    {s.fmt.name:18s} score={s.score:5.2f} fs~{s.sample_rate_est:6.1f}Hz "
                      f"HR~{s.hr_bpm:5.1f} peaks={s.n_peaks:3d} band={s.band_ratio:.2f} {s.notes}")
                res["ranking"].append({"format": s.fmt.name, "score": s.score, "fs": s.sample_rate_est,
                                       "hr": s.hr_bpm, "n_peaks": s.n_peaks, "band": s.band_ratio,
                                       "notes": s.notes})
            if plot and ranked:
                _plot(uuid, ranked[:3], pk)
        results[uuid] = res
    if out_json:
        Path(out_json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {out_json}")
    return results


def _plot(uuid, ranked, pk):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(ranked), 1, figsize=(12, 2.5 * len(ranked)))
    for ax, s in zip(np.atleast_1d(axes), ranked):
        x = Decoder(s.fmt, to_mv=False).decode_many(pk)[:, 0]
        n = min(len(x), int(10 * s.sample_rate_est))
        ax.plot(np.arange(n) / s.sample_rate_est, x[:n], lw=0.7)
        ax.set_title(f"{uuid[:8]} {s.fmt.name} score={s.score:.2f} fs~{s.sample_rate_est:.0f} HR~{s.hr_bpm:.0f}", fontsize=9)
    fig.suptitle("candidate decodes — research only, NOT a medical device", fontsize=9)
    fig.tight_layout()
    plt.show()
