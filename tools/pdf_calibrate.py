#!/usr/bin/env python
"""Calibrate open_kardia's mV gain from a Kardia app **PDF** export (the app offers no CSV).

A KardiaMobile 6L PDF is a standard rhythm strip: 25 mm/s, 10 mm/mV, with a 1 mV
calibration pulse at the start of each lead strip (measured here at 28.35 pt = 1.000 mV,
confirming the scale to <0.2%). This tool reads the printed trace back into absolute
millivolts. Three sub-commands:

  extract   app.pdf --lead I [--out leadI.csv]   -> dump one lead as (t_s, mV)
  fit       --pdf app.pdf --rec rec.csv --lead I  -> gain from ensemble beats vs a open_kardia CSV
  fit-pklg  --pdf app.pdf --pklg session.pklg     -> DEFINITIVE same-recording gain, if the
                                                     PDF and the PacketLogger capture are the
                                                     SAME recording (same beats -> exact fit)

Notes / accuracy:
  * The app applies an "Enhanced Filter" before rendering, so PDF amplitudes are not the raw
    voltage. `fit-pklg` on the SAME recording is exact (per-sample regression after alignment);
    `fit` across different sessions is only good to ~2x (contact + filtering vary).
  * Kardia 6L PDF layout: page 0 is a cover; strip pages stack 6 leads (I,II,III,aVR,aVL,aVF
    top->bottom); only the first strip page carries the 1 mV cal pulses (linewidth ~1.5);
    the trace is the ~0.4-linewidth curves. Layout is identical on every strip page.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_kardia import dsp, print_disclaimer  # noqa: E402

PT_PER_MM = 72.0 / 25.4
PT_PER_MV = 10.0 * PT_PER_MM      # 10 mm/mV
PT_PER_S = 25.0 * PT_PER_MM       # 25 mm/s
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]


def _bands(page1):
    """Return per-lead (baseline_pt, center_pt) from the 6 cal pulses on the first strip page,
    ordered I..aVF (top of page first). PDF y is measured upward, so top = larger y."""
    cals = [c for c in page1.curves if c.get("linewidth", 0) > 1.0]
    if len(cals) < 6:
        raise SystemExit(f"expected 6 calibration pulses on the first strip page, found {len(cals)}")
    b = []
    for c in cals:
        ys = [p[1] for p in c["pts"]]
        b.append((min(ys), (min(ys) + max(ys)) / 2, max(ys) - min(ys)))
    b.sort(key=lambda x: -x[1])
    heights = [h for *_, h in b]
    return [(base, cy) for base, cy, _ in b], float(np.median(heights))


def _lead_on_page(page, cy, base):
    """Reconstruct one lead's (t_s, mV) on one page by concatenating trace segments in x
    order (keeping x strictly increasing), using the cal-pulse bottom as 0 mV."""
    segs = [c for c in page.curves
            if abs(c.get("linewidth", 0) - 0.4) < 0.2
            and abs(np.mean([p[1] for p in c["pts"]]) - cy) < 45]
    segs.sort(key=lambda c: min(p[0] for p in c["pts"]))
    X, Y = [], []
    for c in segs:
        for x, y in sorted(c["pts"]):
            if X and x <= X[-1]:
                continue
            X.append(x); Y.append(y)
    X, Y = np.array(X), np.array(Y)
    if len(X) == 0:
        return np.zeros(0), np.zeros(0)
    m = X > (X.min() + 0.5)                 # drop the cal-pulse x region
    X, Y = X[m], Y[m]
    return (X - X[0]) / PT_PER_S, (Y - base) / PT_PER_MV


def extract_lead(pdf_path, lead="I", max_pages=None):
    """Concatenate one lead across all strip pages -> (t_s, mV, fs_est)."""
    import pdfplumber
    li = LEADS.index(lead)
    with pdfplumber.open(pdf_path) as pdf:
        # first strip page = first page with 6 thick cal pulses
        strip0 = next(i for i, p in enumerate(pdf.pages)
                      if len([c for c in p.curves if c.get("linewidth", 0) > 1.0]) >= 6)
        bands, _ = _bands(pdf.pages[strip0])
        base, cy = bands[li]
        T, MV, toff = [], [], 0.0
        pages = range(strip0, len(pdf.pages))
        for pi in pages:
            t, mv = _lead_on_page(pdf.pages[pi], cy, base)
            if len(t) < 10:                 # analysis/averaged-beat tail pages
                continue
            T.append(t + toff); MV.append(mv); toff += t[-1] + 1.0 / 150
            if max_pages and len(T) >= max_pages:
                break
    t = np.concatenate(T); mv = np.concatenate(MV)
    fs = 1.0 / np.median(np.diff(t))
    return t, mv, fs


def _median_beat(x, fs, pre=0.3, post=0.5):
    xc = dsp.clean(x, fs)
    pk = dsp.detect_r_peaks(xc, fs)
    a, b = int(pre * fs), int(post * fs)
    beats = [xc[p - a:p + b] for p in pk if p - a >= 0 and p + b < len(xc)]
    if len(beats) < 3:
        return None, 0
    return np.median(np.vstack(beats), axis=0), len(beats)


def _orient(a):
    c = a - np.median(a)
    return a if np.max(c) >= -np.min(c) else (2 * np.median(a) - a)


def cmd_extract(a):
    t, mv, fs = extract_lead(a.pdf, a.lead)
    print(f"Lead {a.lead}: {t[-1]:.1f}s, ~{fs:.0f} Hz effective, "
          f"range [{mv.min():.2f},{mv.max():.2f}] mV, R~{np.max(dsp.clean(mv,fs)):.2f} mV")
    if a.out:
        import csv
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t_s", "mV"])
            for ti, vi in zip(t, mv):
                w.writerow([f"{ti:.5f}", f"{vi:.5f}"])
        print(f"wrote {a.out}")


def _fit_templates(mv, fs_pdf, counts, fs, label):
    from scipy.signal import resample_poly
    mv300 = resample_poly(mv, int(fs * 10), int(round(fs_pdf * 10)))
    tb, nb = _median_beat(mv300, fs)
    cb, ncb = _median_beat(counts, fs)
    if tb is None or cb is None:
        sys.exit("not enough beats to form a template")
    n = min(len(tb), len(cb)); tb, cb = _orient(tb[:n]), _orient(cb[:n])
    shift = int(np.argmax(np.abs(tb - np.median(tb))) - np.argmax(np.abs(cb - np.median(cb))))
    cb = np.roll(cb, shift)
    A = np.column_stack([cb, np.ones(n)])
    (gain, off), *_ = np.linalg.lstsq(A, tb, rcond=None)
    corr = float(np.corrcoef(A @ [gain, off], tb)[0, 1])
    print(f"  {label}: PDF beats={nb} rec beats={ncb}  template corr={corr:.3f}  gain={gain:.4g} mV/LSB")
    return gain, corr


def cmd_fit(a):
    from open_kardia.sinks.csv_sink import read_csv
    from open_kardia.protocol import KARDIA_6L_GAIN_MV
    t, mv, fs_pdf = extract_lead(a.pdf, a.lead)
    fs, names, _, x = read_csv(a.rec)
    counts = x[:, names.index(a.lead)] / KARDIA_6L_GAIN_MV
    print_disclaimer()
    gain, corr = _fit_templates(mv, fs_pdf, counts, fs, f"Lead {a.lead}")
    print(f"\n-> gain {gain:.4g} mV/LSB (cross-session ensemble fit; ~2x uncertain unless same "
          f"recording). Current constant: {KARDIA_6L_GAIN_MV:.4g}.")


def cmd_fit_pklg(a):
    """Definitive same-recording fit: decode the app's own ECG stream from the .pklg (raw
    counts), extract the PDF (mV), align by cross-correlation, and regress per sample."""
    import json
    from scipy import signal
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pklg_att import parse  # noqa
    from open_kardia.protocol import KARDIA_6L_DATA_CHAR, KNOWN_FORMAT, Decoder
    # decode counts from the pklg's notifications on the ECG characteristic
    tmp = "captures/_pklg_ecg.jsonl"
    parse(a.pklg, out=tmp, quiet=True, silent=True)
    rows = [json.loads(l) for l in open(tmp)]
    data = [r for r in rows if r.get("kind") in ("notify", "indicate") and str(r.get("uuid", "")).startswith("ac060003")]
    if not data:
        sys.exit("no ac060003 ECG notifications in the pklg")
    counts = Decoder(KNOWN_FORMAT, to_mv=False).decode_many([bytes.fromhex(r["hex"]) for r in data])
    fs = 300.0
    print_disclaimer()
    print(f"pklg: {len(counts)} samples ({len(counts)/fs:.1f}s)")
    gains = []
    for lead, ch in ((a.lead, 0), ("II", 1)):
        t, mv, fs_pdf = extract_lead(a.pdf, lead)
        mv300 = signal.resample_poly(mv, int(fs * 10), int(round(fs_pdf * 10)))
        c = dsp.clean(counts[:, ch].astype(float), fs); p = dsp.clean(mv300, fs)
        cn = (c - c.mean()) / c.std(); pn = (p - p.mean()) / p.std()
        n = min(len(cn), len(pn))
        cc = signal.correlate(cn[:n], pn[:n], mode="full"); lags = signal.correlation_lags(n, n, "full")
        m = np.abs(lags) < int(8 * fs); i = int(np.argmax(cc[m])); lag = int(lags[m][i]); corr = float(cc[m][i] / n)
        if corr < 0.5:
            print(f"  Lead {lead}: corr only {corr:.2f} at lag {lag/fs:+.1f}s -> PDF and pklg are NOT the "
                  f"same recording; use `fit` (ensemble) instead.")
            continue
        # align and regress mV = gain*counts + off over the overlap
        cc_seg = c[lag:lag + n] if lag >= 0 else c[:n + lag]
        pp_seg = p[:len(cc_seg)] if lag >= 0 else p[-len(cc_seg):]
        k = min(len(cc_seg), len(pp_seg)); A = np.column_stack([cc_seg[:k], np.ones(k)])
        (gain, off), *_ = np.linalg.lstsq(A, pp_seg[:k], rcond=None)
        r2 = float(np.corrcoef(A @ [gain, off], pp_seg[:k])[0, 1])
        print(f"  Lead {lead} (ch{ch}): same-recording corr={corr:.3f} lag={lag/fs:+.2f}s  "
              f"gain={gain:.4g} mV/LSB  fit r={r2:.3f}")
        gains.append(gain)
    if gains:
        g = float(np.mean(gains))
        print(f"\nDEFINITIVE gain = {g:.5g} mV/LSB = 1/{1/g:.0f}")
        print(f"-> set KARDIA_6L_GAIN_MV = {g:.5g} in open_kardia/protocol.py")


def main():
    print_disclaimer()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("pdf"); e.add_argument("--lead", default="I", choices=LEADS)
    e.add_argument("--out"); e.set_defaults(fn=cmd_extract)
    f = sp.add_parser("fit"); f.add_argument("--pdf", required=True); f.add_argument("--rec", required=True)
    f.add_argument("--lead", default="I", choices=LEADS); f.set_defaults(fn=cmd_fit)
    g = sp.add_parser("fit-pklg"); g.add_argument("--pdf", required=True); g.add_argument("--pklg", required=True)
    g.add_argument("--lead", default="I", choices=LEADS); g.set_defaults(fn=cmd_fit_pklg)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
