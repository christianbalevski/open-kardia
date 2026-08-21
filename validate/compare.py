#!/usr/bin/env python
"""Validation harness (spec §9).

  python -m validate.compare --tool rec.csv --ref app_export.csv [--ref-fs 300] [--lead I]

* `--tool`: a open_kardia CSV recording (t_s, I, II, ...).
* `--ref` : the same-session reference trace exported from the Kardia app, as CSV
            (one column of mV, or t,mV) — or another open_kardia CSV.
Time-aligns via cross-correlation, reports max correlation, RMS error (µV) and HR
delta, and exits non-zero unless corr >= 0.95 and |ΔHR| <= 2 bpm.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_kardia import dsp, print_disclaimer  # noqa: E402
from open_kardia.sinks.csv_sink import read_csv  # noqa: E402


def load_ref(path, fs_hint):
    try:
        fs, names, t, x = read_csv(path)
        return fs, x[:, 0]
    except Exception:  # noqa: BLE001
        a = np.loadtxt(path, delimiter=",", ndmin=2)
        if a.shape[1] >= 2:
            t, x = a[:, 0], a[:, 1]
            fs = 1.0 / np.median(np.diff(t))
        else:
            x = a[:, 0]
            fs = fs_hint
        if not fs:
            sys.exit("--ref-fs required for a single-column reference")
        return fs, x


def align(a, b, fs, max_lag_s=5.0):
    """Return (lag_samples, corr) maximising normalised cross-correlation of b vs a."""
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    cc = signal.correlate(a, b, mode="full", method="fft") / n
    lags = signal.correlation_lags(n, n, mode="full")
    m = np.abs(lags) <= max_lag_s * fs
    i = np.argmax(cc[m])
    return int(lags[m][i]), float(cc[m][i])


def compare(tool_x, tool_fs, ref_x, ref_fs, lead_name="I", mains=60.0):
    fs = tool_fs
    if abs(ref_fs - tool_fs) > 1e-6:
        ref_x = signal.resample_poly(ref_x, int(round(tool_fs * 100)), int(round(ref_fs * 100)))
    a = dsp.clean(tool_x, fs, mains)
    b = dsp.clean(ref_x, fs, mains)
    lag, _ = align(a, b, fs)
    if lag > 0:
        a2, b2 = a[lag:], b[: len(b) - lag] if lag else b
    else:
        a2, b2 = a[: len(a) + lag] if lag else a, b[-lag:]
    n = min(len(a2), len(b2))
    a2, b2 = a2[:n], b2[:n]
    corr = float(np.corrcoef(a2, b2)[0, 1])
    # scale-invariant RMS: fit gain of tool onto ref
    g = float(np.dot(a2, b2) / (np.dot(a2, a2) + 1e-12))
    rms_uv = float(np.sqrt(np.mean((g * a2 - b2) ** 2)) * 1000.0)
    hr_tool = dsp.heart_rate(a2, fs)
    hr_ref = dsp.heart_rate(b2, fs)
    return {"lead": lead_name, "lag_s": lag / fs, "corr": corr, "gain_fit": g, "rms_uv": rms_uv,
            "hr_tool": hr_tool, "hr_ref": hr_ref, "hr_delta": hr_tool - hr_ref, "n": n,
            "pass_corr": corr >= 0.95, "pass_hr": abs(hr_tool - hr_ref) <= 2.0}


def main(argv=None):
    print_disclaimer()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--ref-fs", type=float)
    p.add_argument("--lead", default="I")
    p.add_argument("--mains", type=float, default=60.0)
    a = p.parse_args(argv)
    fs, names, t, x = read_csv(a.tool)
    if a.lead not in names:
        sys.exit(f"lead {a.lead} not in {names}")
    tool_x = x[:, names.index(a.lead)]
    ref_fs, ref_x = load_ref(a.ref, a.ref_fs)
    r = compare(tool_x, fs, ref_x, ref_fs, a.lead, a.mains)
    for k, v in r.items():
        print(f"{k:10s}: {v:.4f}" if isinstance(v, float) else f"{k:10s}: {v}")
    ok = r["pass_corr"] and r["pass_hr"]
    print("\nRESULT:", "PASS" if ok else "FAIL (need corr>=0.95 and |dHR|<=2 bpm)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
