#!/usr/bin/env python
"""Validate open_kardia.arrhythmia beat labels against MIT-BIH Arrhythmia Database annotations.

  python -m validate.mitdb_arrhythmia [--records 106 119 200 ...] [--dir CACHE_DIR] [--window 0]

Reads each record (channel 0 = MLII, 360 Hz), cleans with dsp.clean(mains=60), detects
R-peaks with dsp.detect_r_peaks, runs detect_arrhythmias (whole record or in --window
second chunks) and matches detected beats to reference annotations within ±150 ms.

Reference symbol mapping: N,L,R,e,j -> N ; A,a,J,S -> S ; V,E -> V ; F fusion (ignored);
everything else (paced, Q, non-beat) ignored. Reports per-record PVC/PAC sensitivity, PPV,
false-PVC rate on normal beats, pause count, and the runtime.

Research tool only — NOT a medical device. MIT-BIH data is downloaded to --dir (default:
~/.cache/open_kardia/mitdb or $MITDB_DIR) and must not be committed into the repo.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_kardia import dsp  # noqa: E402
from open_kardia.arrhythmia import detect_arrhythmias  # noqa: E402

DEFAULT_RECORDS = ["100", "101", "103", "106", "119", "200", "208", "209", "221", "232", "233"]
SYM_MAP = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
           "A": "S", "a": "S", "J": "S", "S": "S",
           "V": "V", "E": "V", "F": "F"}


def load(rec: str, d: str):
    import wfdb
    p = os.path.join(d, rec)
    if not os.path.exists(p + ".dat"):
        wfdb.dl_database("mitdb", dl_dir=d, records=[rec], annotators=["atr"])
    r = wfdb.rdrecord(p)
    a = wfdb.rdann(p, "atr")
    x = r.p_signal[:, 0].astype(float)
    return x, float(r.fs), np.asarray(a.sample), np.asarray(a.symbol)


def run_record(rec: str, d: str, window_s: float, verbose: bool = False):
    x, fs, ann_s, ann_sym = load(rec, d)
    t0 = time.time()
    clean = dsp.clean(x, fs, mains=60.0)
    rp_all, lab_all, findings_all = [], [], []
    n = len(x)
    step = int(window_s * fs) if window_s > 0 else n
    for s0 in range(0, n, step):
        seg = clean[s0:min(n, s0 + step)]
        if len(seg) < 4 * fs:
            continue
        rp = dsp.detect_r_peaks(seg, fs)
        rep = detect_arrhythmias(seg, fs, rp)
        rp_all.append(rp + s0)
        lab_all.append(rep.beat_labels)
        findings_all.extend(rep.findings)
    rp = np.concatenate(rp_all) if rp_all else np.array([], dtype=int)
    lab = np.concatenate(lab_all) if lab_all else np.array([], dtype="<U1")
    dt = time.time() - t0

    # match detections to reference beats within ±150 ms
    tol = int(0.15 * fs)
    ref_cls = np.array([SYM_MAP.get(s, "") for s in ann_sym])
    keep = ref_cls != ""
    ref_s, ref_c = ann_s[keep], ref_cls[keep]
    conf = {c: {"N": 0, "S": 0, "V": 0, "?": 0, "miss": 0} for c in ("N", "S", "V", "F")}
    j = np.searchsorted(rp, ref_s)
    for i, (s, c) in enumerate(zip(ref_s, ref_c)):
        cands = []
        for k in (j[i] - 1, j[i]):
            if 0 <= k < len(rp) and abs(rp[k] - s) <= tol:
                cands.append(k)
        if not cands:
            conf[c]["miss"] += 1
            continue
        k = min(cands, key=lambda k: abs(rp[k] - s))
        conf[c][lab[k]] += 1
    tp_v = conf["V"]["V"]; ref_v = sum(conf["V"].values()) - conf["V"]["miss"]
    fp_v = conf["N"]["V"] + conf["S"]["V"]
    tp_s = conf["S"]["S"]; ref_sc = sum(conf["S"].values()) - conf["S"]["miss"]
    fp_s = conf["N"]["S"] + conf["V"]["S"]
    n_ref_n = sum(conf["N"].values()) - conf["N"]["miss"]
    codes = sorted({f.code for f in findings_all})
    n_pause = sum(f.count for f in findings_all if f.code == "pause")
    ref_pauses = int(np.sum(np.diff(ann_s[keep]) / fs > 2.0))
    row = {
        "rec": rec, "beats": len(rp), "ref_V": ref_v, "V_se": tp_v / ref_v if ref_v else np.nan,
        "V_ppv": tp_v / (tp_v + fp_v) if (tp_v + fp_v) else np.nan,
        "falseV_on_N%": 100.0 * conf["N"]["V"] / n_ref_n if n_ref_n else np.nan,
        "ref_S": ref_sc, "S_se": tp_s / ref_sc if ref_sc else np.nan,
        "S_ppv": tp_s / (tp_s + fp_s) if (tp_s + fp_s) else np.nan,
        "falseS_on_N%": 100.0 * conf["N"]["S"] / n_ref_n if n_ref_n else np.nan,
        "pauses(det/ref>2s)": f"{n_pause}/{ref_pauses}", "missed_det": conf["N"]["miss"] + conf["V"]["miss"] + conf["S"]["miss"],
        "sec": dt, "codes": ",".join(codes),
    }
    if verbose:
        print(rec, conf)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="*", default=DEFAULT_RECORDS)
    ap.add_argument("--dir", default=os.environ.get("MITDB_DIR", os.path.expanduser("~/.cache/open_kardia/mitdb")))
    ap.add_argument("--window", type=float, default=0.0, help="process in windows of N seconds (0 = whole record)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.dir, exist_ok=True)
    rows = [run_record(r, a.dir, a.window, a.verbose) for r in a.records]
    cols = ["rec", "beats", "ref_V", "V_se", "V_ppv", "falseV_on_N%", "ref_S", "S_se", "S_ppv",
            "falseS_on_N%", "pauses(det/ref>2s)", "missed_det", "sec", "codes"]
    def fmt(v):
        if isinstance(v, float):
            return "--" if not np.isfinite(v) else f"{v:.2f}"
        return str(v)
    print("  ".join(f"{c:>{max(6,len(c))}}" for c in cols))
    for r in rows:
        print("  ".join(f"{fmt(r[c]):>{max(6,len(c))}}" for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
