# capture_decode_notes.md — living log of the 6L reverse-engineering findings

> Research notes for personal interoperability with hardware the author owns.
> Nothing here is a medical device, nothing is clinically validated.

## Status board

| Item | Status | Evidence |
|---|---|---|
| Advertised name / service UUIDs of the retail 6L (AC-019) | **CAPTURED 2026-08-16** | `Kardia6L 26E7`; svcs 0x180A, 0x180F, `ac060001-328c-a28f-9846-5a8aa212661b`; no mfg data |
| GATT map (services / characteristics / properties) | **CAPTURED** | `captures/gatt_dump.json` (see table below) |
| ECG notify characteristic | **`ac060003` (notify)** ✅ | streams after handshake; handle 0x0013 |
| Command channel | `ac060002` write+indicate | start handshake below |
| Link security | **plaintext, no bonding** ✅ | entropy H=5.6, zlib 0.68 → PLAINTEXT-LIKE; §5 good case |
| Byte layout | **s16 LE, 2ch interleaved [I,II], no header** ✅ | scorer + golden vector |
| Sample rate | **300 Hz** ✅ (measured 300.0–300.5) | 9 samples/ch × 36-B notif × 33.4/s |
| mV gain & offset | **offset 0; gain ESTIMATE 1.27e-4 mV/LSB (~2x)** ⚠️ | PDF cal pulse exact; needs same-recording pklg+PDF for an exact fit |
| `protocol.KNOWN_FORMAT` | **set** (`open-kardia_6l`) ✅ | golden vector in tests/test_protocol.py |

**Nothing in `open_kardia/protocol.py` is a verified constant.** `CANDIDATES` is the hypothesis
space from spec §4.3, not a claim about the device.

## Session log

### 2026-08-16 (decisive) — Track B (iOS PacketLogger) cracked the protocol
* Captured the iOS Kardia app doing a 60 s recording (`captures/kardia_app_session.pklg`);
  parsed it with `tools/pklg_att.py` (Apple's newer .pklg is **little-endian** — parser
  auto-detects). Handle map: cmd `ac060002`=0x0010, data `ac060003`=0x0013.
* **Start handshake** (writes to `ac060002`, ASCII), then enable notify on `ac060003`:
  1. `4b 45303743354645303145394438353244` = `"K"` + `"E07C5FE01E9D852D"` → indicate `01`
  2. `4d 32` = `"M2"` → indicate `01`
  3. write `0100` to `ac060003` CCCD (handle 0x0014) to subscribe
  (The token is why blind `4b…` probes returned `04`. `M2` likely selects mode/2-channel.)
* **ECG stream** on `ac060003`: 36-byte notifications, 33.4/s. Decoded as signed int16
  LE, **2 channels interleaved [Lead I, Lead II]**, **300 Hz**, no header/sequence.
  Entropy → PLAINTEXT. Scorer top pick `h0_s16le_c2i` score 6.7, HR 66, leads distinct
  (r=0.83). Set `protocol.KNOWN_FORMAT`; froze `tests/golden_6l_notifications.json`.
* **Live replay** (`tools/replay_start.py`) of the exact handshake **worked** — the token
  is replayable. Streamed 689 notifications, no disconnect; decoded HR 70 bpm, SQI 0.92,
  R≈1.2 mV. Then wired the handshake into `open-kardia record/stream` and captured a clean
  30 s **6-lead** recording via the CLI (`captures/rec_cli.*`, `rec_cli_6leads.png`):
  8550 samples @ 300.03 Hz, 0 gaps, HR 61.4 bpm == neurokit2's 61.4 bpm. Textbook P-QRS-T,
  aVR inverted, derived leads consistent. **Phase 1 complete; streaming client works.**
* Open item: **mV gain is provisional** (assumed ~1 mV R-wave → 1/3840 mV/LSB). To finalise,
  see the calibration entry below — the app exports only PDF, and `tools/pdf_calibrate.py`
  reads absolute mV off the PDF's 25 mm/s · 10 mm/mV scale.

### 2026-08-16 — desktop GUI added
* `open_kardia/app.py` (Tkinter) + `open_kardia/appcore.py` (thread-safe controller bridging the
  async BLE pipeline to the GUI). Launch via `./bin/open-kardia-app` (from Terminal.app for
  Bluetooth) or `python -m open_kardia.app`; `--sim` for a no-hardware demo. Connect, live
  6-lead plot, HR + SQI readouts, record to CSV/WFDB, snapshot save. 158 tests still pass;
  sim run exits clean. Same engine as the CLI (connect/stream/protocol reused).

### 2026-08-16 — mV calibration from the app PDF (partial)
* User's PDF `ecg-20260816-174935.pdf`: 6-lead, **Scale: 25mm/s, 10mm/mV**, Enhanced Filter,
  60 Hz mains, "Normal Sinus Rhythm", 66 BPM. Layout: cover page + strip pages stacking
  I/II/III/aVR/aVL/aVF; only the first strip page carries the 1 mV **calibration pulses**.
* Cal pulse measured **28.35 pt tall = 1.000 mV** at 10 mm/mV → the PDF's mV axis is exact
  (<0.2%). `tools/pdf_calibrate.py extract` reconstructs any lead in absolute mV from the
  vector trace (concatenate ~0.4-width curve segments in x-order; cal-pulse bottom = 0 mV).
* BUT: the PDF and every raw capture I hold are from **different sessions** (cross-corr ~0.1
  between the PDF and `kardia_app_session.pklg`), and the app's Enhanced Filter alters
  amplitudes. So amplitude-matching only brackets the gain: Lead I ~8.9e-5, Lead II
  ~7.3e-5..1.8e-4 mV/LSB. Set `KARDIA_6L_GAIN_MV = 1.27e-4` (geo-mean) as a documented
  ESTIMATE (~2x). Morphology/timing/HR/lead-math are unaffected by this scalar.
* **To finalise exactly:** capture ONE recording with **PacketLogger running** (its .pklg
  carries the app's ac060003 ECG stream = raw counts) AND export **that same recording's**
  PDF (= mV). Then `python tools/pdf_calibrate.py fit-pklg --pdf that.pdf --pklg that.pklg`
  aligns them per-sample (same beats) and prints the exact gain. The tool refuses (and says
  so) if the PDF and pklg are not the same recording, as it did for the pair on hand.

### 2026-08-16 (later) — hardware reached from a Terminal.app-hosted session
* Relaunching the agent from Terminal.app fixed TCC. Scan/connect/enumerate work.
* Full GATT dump captured; command channel probed exhaustively at 1–2 bytes (see below).
* No ECG data obtained: the stream is gated behind an unknown command/handshake.

### 2026-08-16 — scaffold + tooling; hardware capture blocked in this session
* Built the package, decoder-candidate framework, ECG-likeness scorer, entropy diagnostic,
  pipeline, sinks, CLI, validation harness; 157 tests pass on synthetic data.
* Attempted Track A (`tools/gatt_dump.py scan`) from the agent's shell on this Mac:
  the process was killed with **SIGABRT by macOS TCC** — the python.org framework's
  `Python.app` has no `NSBluetoothAlwaysUsageDescription`, and the shell used by the agent
  is not a Bluetooth-permitted terminal. Editing `/Library/Frameworks/.../Info.plist` was
  not permitted in this session. Created `tools/setup_macos_bt.sh` + `bin/btpython`
  (a copied, re-signed `Python.app` bundle carrying the key); it still aborted from this
  shell — most likely because TCC attributes CLI processes to the *responsible*
  terminal app, so it must be run from Terminal.app / iTerm2 with Bluetooth granted.
  → **Phase 1 hardware capture must be run interactively by the owner** (it also needs
  ≥30 s of fingers on the electrodes). See the runbook below.
* Nothing about the 6L wire format was learned. Do not proceed to `record`/`stream` on real
  hardware without completing the runbook and filling in the status board.

## Runbook — Track A (macOS, passive enumeration)

Do this in **Terminal.app** (Apple's Terminal already carries the Bluetooth usage string; the
first run pops a "Terminal would like to use Bluetooth" dialog — allow it).

```bash
cd ~/repos/open-kardia && source .venv/bin/activate
# 0. make sure the phone app is closed / not connected to the 6L (only one central at a time)
# 1. wake the device (touch both top electrodes) and scan
open-kardia scan --secs 15            # look for a name containing Kardia/AliveCor; note the UUID
open-kardia scan --secs 15 --all      # if nothing: list everything, look for an unnamed device
                                    #   whose RSSI tracks the 6L, or manufacturer data
# 2. full GATT enumeration (device must still be awake)
open-kardia dump                      # -> captures/gatt_dump.json ; commit this file
# 3. subscribe to every notify/indicate characteristic and log raw bytes, 45 s,
#    fingers on both electrodes the whole time (Lead I). If the device seems to need
#    a bond, add --pair.
open-kardia sniff --secs 45 --out captures/sniff_leadI_01.jsonl
# 3b. some devices need a "start" write. If nothing streams, look in the dump for a
#     write/write-without-response characteristic on the same vendor service and try
#     candidate commands (start with 0x01, 0x0100, 0x0101 ...):
open-kardia sniff --secs 45 --write <WRITE_UUID>=01 --out captures/sniff_start01.jsonl
# 4. analyze
open-kardia analyze captures/sniff_leadI_01.jsonl --top 12 --out captures/analysis_01.json --plot
```

`analyze` prints per characteristic: rate, size histogram, first payloads (spot headers /
counters by eye), the **§5 entropy verdict** and the ranked byte-layout candidates with an
estimated fs and HR. Interpret:

* `PLAINTEXT-LIKE` + a candidate with score > 7, physiologic HR, fs in ~250–600 Hz →
  proceed. Note the c1 vs c2-interleaved ambiguity (aliases): pick c2 only if channels
  0/1 differ in morphology (r < ~0.99); the app-trace correlation (Track B / validate)
  is the tie-breaker.
* `CIPHERTEXT-LIKE` **without** `--pair` → repeat with `--pair` (bond). If it turns
  plaintext, note "link-layer encryption only" — the good case.
* `CIPHERTEXT-LIKE` **with** `--pair` → **application-layer encryption. STOP.** Write it up
  here and in README §Limitations. Do not attempt key extraction (spec §5).

If a candidate wins: freeze it —

```bash
# record 30 s using the winning layout, the measured fs and the ECG characteristic UUID
open-kardia record --format h2_s16le_c1i --fs 300 --char <ECG_UUID> --secs 30 --out captures/rec_01 --formats csv,wfdb
```
(`record` keeps `captures/rec_01.raw.jsonl` — raw bytes — so any capture can be re-decoded
if the layout is refined later.)

## Runbook — Track B (phone-side ground truth)

* **Android:** Settings → Developer options → *Enable Bluetooth HCI snoop log* → run a
  real 6-lead recording in Kardia → `adb bugreport` (or `adb pull` the btsnoop file) →
  Wireshark: filter `btatt.opcode == 0x1b` (Handle Value Notification), note the handle,
  export the payloads (`tshark -r x.log -Y 'btatt.opcode==0x1b' -T fields -e frame.time_epoch -e btatt.handle -e btatt.value`),
  and convert to the sniff JSONL shape `{t, uuid, len, hex}` to feed `open-kardia analyze`.
* **iOS:** install the Bluetooth logging profile + PacketLogger (Additional Tools for
  Xcode), record, export .pklg, open in Wireshark, same filter.
* Export the recording from the app (History → share). If only a PDF is available,
  the PDF trace can be digitised (25 mm/s, 10 mm/mV) into a CSV for `validate/compare.py`.

## Deriving the format (spec §4.3) — what to fill in here

```
characteristic:        <uuid>          properties: <notify|indicate>  size(s): <n bytes>
notifications/s:       <x>             bytes/s: <y>
header:                <k bytes> (<counter at offset i, w bytes> | none)
sample:                <int16 le|be|int24...>, <signed|unsigned>
channels:              <1|2>, <interleaved|blocked>
sample rate (measured):<fs> Hz  (samples / span over ≥30 s; std of estimate <..>)
gain / offset (fitted): mV = (raw - <offset>) * <gain>   (R² = .., from validate/compare)
noise floor (RMS, µV): <..>  (IFU sanity bound: ≤ 47 µV RMS vs 12-lead)
sequence-gap behaviour: <..>
```
Then: set `protocol.KNOWN_FORMAT`, add `captures/golden_notifications.jsonl` + expected
samples to `tests/test_protocol.py::test_golden_vector_real_device`, and remove its skip.

## GATT map (from captures/gatt_dump.json, fw 3.0.1, hw 19SC03.03, MTU 247)

| Service | Characteristic | Props | Notes |
|---|---|---|---|
| `ac060001-328c-a28f-9846-5a8aa212661b` (vendor) | `ac060002-…` | write, indicate | **command / response channel** (1-byte replies observed) |
| | `ac060003-…` | notify | presumed ECG stream — never produced data yet |
| | `ac060005-…` | read | 2 B LE, ~0x0c03–0x0c1c (3075–3100) — looks like battery mV |
| | `ac060006-…` | read | 2 B LE, ~0x0bc1–0x0bc6, slowly rising — another ADC value |
| 0x180A Device Information | 2A29/2A24/2A25/2A27/2A26 | read | AliveCor / AC-019 / serial / 19SC03.03 / 3.0.1 |
| 0x180F Battery | 2A19 | read, notify | 100 → 98 % |
| 0xFE59 Nordic DFU | `8ec90003-f315-4f60-9fb8-838830daea50` | write, indicate | Buttonless DFU (**never write to this**) — device is nRF52-based |

Behaviour: the device advertises only while electrodes are touched / shortly after; it
drops the connection ~9–14 s after connect regardless of what is written; enabling
notify/indicate on 0002/0003/2A19 works without any pairing prompt.

### Command-channel probing (2026-08-16, `tools/probe_*.py`, logs in `captures/probe_*.jsonl`)

Every write to `ac060002` is answered by a single indicated byte:

| Written | Reply | Notes |
|---|---|---|
| `00` (+ any bytes) | `01` | |
| `20` / `20 00` / `20 00 …` | `01` | `20 xx` with xx≠00 → `03`; `20 01 xx` (len-framing) → `03` |
| `2c` / `2c 00` / `2c 00 …` | `01` | same as 0x20 |
| `4b` ('K') + anything (0–20 B, any 2nd byte) | `04` | |
| every other first byte 0x01–0xFF | `03` | |

Interpretation (hypothesis): Nordic-style response codes — `01` SUCCESS, `03` INVALID
PARAMETER/OPCODE, `04` INSUFFICIENT RESOURCES or CRC/AUTH failure. `00`, `20 00`, `2c 00`
are "accepted" but produce **no data on `ac060003`** and no visible state change in
`0005/0006`. `4b` is special (distinct code for all payloads) — possibly a framed/CRC'd or
authenticated message class. **The start/handshake sequence is unknown; blind guessing
past 2 bytes is impractical → Track B (capture the phone app's session) is required.**
