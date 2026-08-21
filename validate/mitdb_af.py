#!/usr/bin/env python
"""MIT-BIH validation of kardia6l.af (RR-based AF detector).

  python -m validate.mitdb_af [--dir CACHE] [--db mitdb|afdb] [--records 201,202,...]
                              [--window 30] [--minutes N] [--verbose]

Downloads (once) the requested PhysioNet records into --dir with wfdb, cuts each record
into non-overlapping --window s windows, labels a window AF when >= 50 % of its samples
lie inside an '(AFIB' rhythm annotation, runs dsp.clean + dsp.detect_r_peaks at the
record's fs (end-to-end path), scores the resulting RR series with af.af_score, and
prints per-record and pooled sensitivity/specificity.  RRs from the reference beat
annotations are scored too ("ann" columns) to separate detector error from scorer error.
NOT a clinical validation; research aid only.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kardia6l import af, dsp  # noqa: E402

DEFAULT_DIR = ("/private/tmp/claude-501/-Users-cbalevski-repos-open-kardia/"
               "1237e89b-b913-49bb-863a-2619ef8c452e/scratchpad")
MITDB_AF = ["201", "202", "203", "210", "219", "221", "222"]
MITDB_CTRL = ["100", "101", "103", "106", "119", "208", "233"]
AFDB = ["04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091", "05121",
        "05261", "06426", "06453", "06995", "07162", "07859", "07879", "07910", "08215",
        "08219", "08378", "08405", "08434", "08455"]
BEAT_SYMS = set("NLRBAaJSVrFejnE/fQ?")


def rhythm_mask(ann, n):
    """Per-sample masks: (is_afib, is_aflutter) from the '(XXX' rhythm annotations."""
    afib, afl = np.zeros(n, bool), np.zeros(n, bool)
    cur, start = None, 0
    events = [(int(s), a) for s, a in zip(ann.sample, ann.aux_note) if a and a.startswith("(")]
    events.append((n, "(END"))
    for s, a in events:
        if cur is not None:
            afib[start:s] = cur.startswith("(AFIB")
            afl[start:s] = cur.startswith("(AFL")
        cur, start = a.strip("\x00"), s
    return afib, afl


def run_record(db, rec, ddir, win_s, minutes, verbose):
    import wfdb
    path = os.path.join(ddir, rec)
    if os.path.exists(path + ".hea"):
        kw = {}
    else:                       # not cached: stream just the needed byte range from PhysioNet
        path, kw = rec, {"pn_dir": db}
    sampto = int(minutes * 60 * wfdb.rdheader(path, **kw).fs) if minutes else None
    r = wfdb.rdrecord(path, sampto=sampto, channels=[0], **kw)
    ann = wfdb.rdann(path, "atr", sampto=sampto, **kw)
    fs, x = float(r.fs), r.p_signal[:, 0]
    n = len(x)
    afib, afl = rhythm_mask(ann, n)
    if db == "afdb":            # afdb: rhythm in .atr, beats in the separate .qrs file
        try:
            ann_b = wfdb.rdann(path, "qrs", sampto=sampto, **kw)
        except Exception:  # noqa: BLE001
            ann_b = ann
    else:
        ann_b = ann
    beats = ann_b.sample[np.isin(ann_b.symbol, list(BEAT_SYMS))]
    xc = dsp.clean(x, fs, mains=60.0)
    peaks = dsp.detect_r_peaks(xc, fs)
    w = int(win_s * fs)
    rows = []
    for k in range(n // w):
        a, b = k * w, (k + 1) * w
        truth = bool(np.mean(afib[a:b]) >= 0.5)
        is_afl = bool(np.mean(afl[a:b]) >= 0.5)      # reported separately (irregular too)
        pk = peaks[(peaks >= a) & (peaks < b)]
        rr = np.diff(pk) / fs * 1000.0
        res = af.af_score(rr)
        bk = beats[(beats >= a) & (beats < b)]
        res_ann = af.af_score(np.diff(bk) / fs * 1000.0)
        rows.append((truth, res, res_ann, is_afl))
        if verbose and truth != res.is_af:
            print(f"    {rec} win{k:3d} t={a/fs:6.0f}s truth={'AF ' if truth else 'nAF'} "
                  f"score={res.score:.2f} ann={res_ann.score:.2f} {res.evidence}")
    return rows


def confusion(rows, which=1):
    tp = sum(1 for t, *r in rows if t and r[which - 1].is_af)
    fn = sum(1 for t, *r in rows if t and not r[which - 1].is_af)
    fp = sum(1 for t, *r in rows if not t and r[which - 1].is_af)
    tn = sum(1 for t, *r in rows if not t and not r[which - 1].is_af)
    return tp, fn, fp, tn


def summary_line(name, rows):
    tp, fn, fp, tn = confusion(rows, 1)
    atp, afn, afp, atn = confusion(rows, 2)
    return (f"{name:>6} {len(rows):4d} {tp+fn:5d} | {tp:3d} {fn:3d} {fp:3d} {tn:3d} {pct(tp, tp+fn)} {pct(tn, tn+fp)} "
            f"| {pct(atp, atp+afn):>6} {pct(atn, atn+afp):>6}")


def pct(a, b):
    return f"{100.0 * a / b:5.1f}" if b else "  n/a"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--db", default="mitdb", choices=["mitdb", "afdb"])
    ap.add_argument("--records", default=None, help="comma list; default = AF + control set")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--minutes", type=float, default=None, help="only read the first N minutes")
    ap.add_argument("--verbose", action="store_true", help="print each misclassified window")
    a = ap.parse_args(argv)
    import wfdb
    recs = a.records.split(",") if a.records else (MITDB_AF + MITDB_CTRL if a.db == "mitdb" else AFDB)
    ddir = os.path.join(a.dir, a.db)
    os.makedirs(ddir, exist_ok=True)
    if a.db == "afdb" and a.minutes is None:
        a.minutes = 30.0            # afdb records are 10 h; only the first N minutes
    missing = [r for r in recs if not os.path.exists(os.path.join(ddir, r + ".hea"))]
    if missing and a.minutes is None:   # full records wanted: cache them locally
        wfdb.dl_database(a.db, dl_dir=ddir, records=missing)
    print(f"{a.db}: {a.window:.0f} s windows, AF = >=50% '(AFIB'.  det = dsp.detect_r_peaks RRs, ann = reference-beat RRs")
    print(f"{'rec':>6} {'win':>4} {'AFwin':>5} | {'TP':>3} {'FN':>3} {'FP':>3} {'TN':>3} {'Se%':>5} {'Sp%':>5} | {'annSe%':>6} {'annSp%':>6}")
    pooled = []
    for rec in recs:
        try:
            rows = run_record(a.db, rec, ddir, a.window, a.minutes, a.verbose)
        except Exception as e:  # noqa: BLE001
            print(f"{rec:>6}  failed: {e}")
            continue
        pooled += rows
        print(summary_line(rec, rows))
    print(summary_line("ALL", pooled))
    n_afl = sum(1 for r in pooled if r[3] and not r[0])
    if n_afl:
        # atrial flutter with variable block is also irregular; an RR-only detector cannot
        # separate it from AF, so report the pooled numbers without those windows too
        print(summary_line("-AFL", [r for r in pooled if not (r[3] and not r[0])])
              + f"   ({n_afl} non-AF windows with >=50% '(AFL' excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
