# kardia6l

Research-grade macOS Bluetooth-LE interface for the **AliveCor KardiaMobile 6L** (AC-019).
Discovers the device, replays the app's start handshake, and streams the **live 6-lead ECG**
to CSV / WFDB / EDF and a scrolling live plot — no phone app in the loop.

> ⚠️ **NOT A MEDICAL DEVICE.** kardia6l is an independent interoperability project. Its output
> is a *research signal*: not clinically validated, not for diagnosis, monitoring, or treatment.
> Use it only with a device **you own**. See *Limitations*.

> _Sample output (a live 6-lead capture, HR/interval report, and the desktop app) is
> generated locally under `captures/`; recordings are personal health data and are kept out
> of version control (see `.gitignore`)._

## What works (verified 2026-08-16 on fw 3.0.1 / hw 19SC03.03)

* **Discover / connect / enumerate** over CoreBluetooth (via `bleak`), addressing the
  peripheral by its macOS UUID and caching it in `~/.kardia6l/known.json`.
* **Decoded wire format** (reverse-engineered — see `tools/capture_decode_notes.md`):
  vendor service `ac060001-…`, ECG on `ac060003` (notify). 36-byte notifications at 33.4/s =
  **signed int16, little-endian, 2 channels interleaved `[Lead I, Lead II]`, 300 Hz**, no
  header. **Plaintext** — no application-layer encryption, no bonding required.
* **Start handshake** (ASCII writes to `ac060002`): `K`+token → `M2` → subscribe. Sent
  automatically by `record`/`stream`.
* **6 leads**: I & II are measured; III, aVR, aVL, aVF are derived (Einthoven/Goldberger).
* **DSP**: 0.5–40 Hz bandpass, 60 Hz notch, baseline removal, Pan–Tompkins R-peaks, HR,
  signal-quality index. HR matches `neurokit2` to 0.1 bpm on real captures.
* **Outputs**: CSV, WFDB (`.dat/.hea`), EDF, JSON firehose, live matplotlib plot.

## Install

Requires a **python.org (framework) Python** on macOS (see *macOS Bluetooth permission*).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or ".[plot,files,validate]" for a runtime subset
```

## Use

```bash
kardia6l scan                         # find the device (wake it: touch both electrodes)
kardia6l dump                         # full GATT enumeration -> captures/gatt_dump.json
kardia6l record --secs 30 --out rec --formats csv,wfdb --derive   # 6-lead capture to file
kardia6l stream --plot --derive       # live scrolling plot (Lead I/II + derived)
kardia6l stream --json                # JSON-lines firehose on stdout
```

`record`/`stream` use the built-in 6L format and handshake automatically. `--sim` runs the
whole pipeline against a synthetic device with no hardware. Every command prints the
non-diagnostic disclaimer once.

## Desktop app

A Tkinter window: one click to scan/connect (auto-picks the 6L and runs the start
handshake), a live scrolling 6-lead plot, large heart-rate and signal-quality readouts, a
record toggle that streams to CSV/WFDB, and a snapshot-save of the on-screen buffer. It also
includes:

- **Quality-gated timed recording** — pick a duration (15/30/60 s) and a signal-quality
  threshold (default 90 %); the app waits until the signal is sustained above the threshold,
  then captures exactly that duration, so you don't record noise.
- **Automatic analysis** — each completed reading is classified (rhythm + confidence) and
  measured: HR, PR, QRS, QT, QTc (Bazett & Fridericia), P/T duration, ST level, RR mean±SD,
  frontal QRS axis, and HRV (SDNN/RMSSD/pNN50), shown in a results panel.
- **Saved-readings library** — every reading is stored (CSV + WFDB + metadata) and listed in
  the window; re-open its analysis, or export a printable **PDF/PNG report** (standard
  25 mm/s · 10 mm/mV grid, intervals table, median-beat panel).
- **Edge-stable live trace** — the plot filters a longer buffer and shows an interior slice,
  so the window ends don't ring.
- **Single-lead mode** — check "Single lead (hands only)" when you hold the device in both
  hands without left-leg contact: only Lead I is valid then, so the app shows/saves/analyzes
  Lead I alone and skips the derived limb leads and QRS axis (which need Lead II).

```bash
./bin/kardia6l-app            # from the repo (uses the venv / signed BT helper bundle)
kardia6l-app                  # after: pip install -e ".[app]"
python -m kardia6l.app        # module form
python -m kardia6l.app --connect   # real device, auto-connect on launch
./bin/kardia6l-app --sim      # no-hardware demo with a synthetic 6L
```

The analysis, report, and library pieces are also usable headlessly:
`kardia6l.analysis.analyze_recording(x, fs, lead_names)`, `kardia6l.report.render_report(...)`,
and `kardia6l.library.ReadingLibrary` (stores under `~/.kardia6l/readings/`). **The rhythm
labels and interval measurements are a research aid, not a diagnosis.**

Launch it **from Terminal.app** so Bluetooth works (see *macOS Bluetooth permission* below).
It's the same engine as the CLI — it shares `connect` / `stream` / `protocol` via the
`appcore.AppController` bridge — so the same non-diagnostic disclaimer applies: this is a
research signal, **not a medical device**.

### macOS Bluetooth permission (important)

A CLI Python that calls CoreBluetooth is killed by macOS TCC (`SIGABRT`) unless the process
is attributed to a Bluetooth-permitted app. Simplest fix: **run kardia6l from Apple's
Terminal.app** and allow the "Terminal would like to use Bluetooth" prompt on first scan.
(`tools/setup_macos_bt.sh` builds a signed helper bundle with the usage string as a
fallback.) macOS addresses BLE peripherals by an opaque UUID, not a MAC — reconnection uses
the cached UUID.

## Reverse-engineering it yourself / other units

The token in the handshake may differ per unit or firmware. If a live replay is rejected,
re-capture the app's session and re-derive:

1. **Track A** (macOS): `kardia6l dump`, then `kardia6l sniff --secs 45` (fingers on
   electrodes) → `kardia6l analyze <capture>.jsonl` scores candidate byte layouts and runs
   the §5 entropy (plaintext vs ciphertext) diagnostic.
2. **Track B** (authoritative): capture the phone app with **iOS PacketLogger** (or Android
   HCI snoop) during a real recording, then `python tools/pklg_att.py session.pklg --out
   app_session.jsonl` prints the exact write sequence and the ECG notifications.

Full runbook and the current findings live in `tools/capture_decode_notes.md`.

## Validation & mV calibration (works with only a PDF export)

The Kardia app only exports **PDF**, never CSV — which is the whole reason this project
exists. That's fine: a Kardia ECG PDF is printed at the standard **25 mm/s, 10 mm/mV**
scale, so the paper itself is an absolute millivolt reference.

```bash
# 1) record ~30 s in the Kardia app and export the PDF to captures/app.pdf
# 2) record ~30 s with kardia6l (same sitting):
kardia6l record --secs 30 --out captures/rec --formats csv --derive
# 3) fit the gain from the PDF's calibrated scale (beat-template match; no CSV needed):
python tools/pdf_calibrate.py fit --pdf captures/app.pdf --rec captures/rec.csv --lead I
```

`pdf_calibrate.py` reads the printed trace back to absolute mV via the PDF's 1 mV cal pulse
and 10 mm/mV scale. `fit` matches ensemble-averaged beats between the PDF and a kardia6l CSV
(good to ~2x across sessions). **For an exact gain**, capture one recording with PacketLogger
running and export *that same* recording's PDF, then `fit-pklg --pdf x.pdf --pklg x.pklg`
decodes the app's own ECG stream from the .pklg and regresses it per-sample against the PDF
(same beats). It reports the correlation and refuses if the two aren't the same recording.

If you *do* have a CSV reference (e.g. from another tool), `validate/compare.py` does the
classic same-beat cross-correlation / RMS / HR-delta gate (≥ 0.95, ±2 bpm).

## Limitations & open items

* **mV gain is an ESTIMATE (~2x).** Rate (300 Hz), endianness, interleave and 0 offset are
  verified. The absolute mV scale (`protocol.KARDIA_6L_GAIN_MV = 1.27e-4` mV/LSB) is derived
  from the app PDF's 1 mV calibration pulse (exact) but only cross-session amplitude-matched
  (the app filters amplitudes and no same-recording PDF+counts pair was available), so it is
  bracketed, not pinned. For an exact value, capture one recording with **PacketLogger
  running** and export **that same recording's** PDF, then
  `python tools/pdf_calibrate.py fit-pklg --pdf x.pdf --pklg x.pklg`. Morphology, timing, HR,
  rhythm and lead math are unaffected by this scalar.
* Verified on one unit (AC-019, fw 3.0.1). The handshake token and format may vary across
  firmware; re-derive with the runbook above if needed.
* The device sleeps unless the electrodes are touched and ends the link a few seconds after
  a recording; `connect.py` reconnects by cached UUID with backoff and flags notification
  stalls, but a capture needs continuous skin contact.
* Ethics: reads only your own device's **plaintext** characteristics. It never touches the
  Nordic DFU characteristic, never impersonates the device to AliveCor's cloud, and uploads
  nothing. If a future firmware encrypts the payload behind a server-negotiated key, the
  project stops at "documented, not decoded" (spec §5) rather than defeating crypto.

## Layout

`kardia6l/` — `scan`, `connect`, `protocol` (decoder + `KNOWN_FORMAT`), `leads`, `dsp`
(filters, R-peaks, edge-stable display filter), `stream`, `sinks/`, `sim`, `cli`,
`analysis` (rhythm + intervals + HRV + axis), `report` (PNG/PDF ECG report), `library`
(saved-readings store), `app` (Tkinter GUI), `appcore` (threaded async↔GUI controller +
quality-gated recording). `bin/` — `kardia6l-app` (desktop launcher), `btpython`. `tools/` —
`gatt_dump.py`, `pklg_att.py`, `replay_start.py`, `pdf_calibrate.py`, `probe_*.py`,
`capture_decode_notes.md`. `validate/compare.py` — the validation harness.
`tests/` — 158 tests (lead math, DSP, decoder round-trips, format scorer, pipeline, file
round-trips, validation, and the real-device golden vector).
