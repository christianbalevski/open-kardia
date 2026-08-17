"""kardia6l CLI.

  kardia6l scan                        list nearby Kardia devices
  kardia6l dump [--address U]          full GATT enumeration -> JSON
  kardia6l sniff [--secs 30]           subscribe to all notify chars, log raw bytes (Phase 1)
  kardia6l analyze captures/raw.jsonl  score candidate byte layouts + entropy (§4.3/§5)
  kardia6l record --secs 30 --out rec  capture to CSV/WFDB/EDF (needs a known/--format layout)
  kardia6l stream --plot [--json]      live plot / JSON firehose
Add --sim to record/stream to run against a synthetic device (no hardware).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__, print_disclaimer
from .protocol import CANDIDATES, KNOWN_FORMAT, FrameFormat

log = logging.getLogger("kardia6l")


# --------------------------------------------------------------------------- helpers
def _resolve_format(a) -> FrameFormat:
    if getattr(a, "format", None):
        spec = a.format
        if spec.endswith(".json"):
            d = json.loads(Path(spec).read_text())
            return FrameFormat(**d)
        for c in CANDIDATES:
            if c.name == spec:
                f = c
                break
        else:
            sys.exit(f"unknown --format {spec!r}; use a CANDIDATES name (see `kardia6l formats`) "
                     f"or a JSON file")
        return f
    if KNOWN_FORMAT is not None:
        return KNOWN_FORMAT
    sys.exit("The 6L wire format has not been established yet (Phase 1). Run `kardia6l sniff` with "
             "fingers on the electrodes, then `kardia6l analyze`, then pass --format <name> "
             "--fs <hz> --char <uuid> (see tools/capture_decode_notes.md).")


def _make_sinks(a, fmt: FrameFormat):
    from .sinks.csv_sink import CSVSink
    sinks = []
    out = getattr(a, "out", None)
    if out:
        p = Path(out)
        exts = (a.formats or "csv").split(",")
        for e in exts:
            e = e.strip().lower()
            if e == "csv":
                sinks.append(CSVSink(p.with_suffix(".csv")))
            elif e == "wfdb":
                from .sinks.wfdb_sink import WFDBSink
                sinks.append(WFDBSink(p.with_suffix("")))
            elif e == "edf":
                from .sinks.wfdb_sink import EDFSink
                sinks.append(EDFSink(p.with_suffix(".edf")))
            else:
                sys.exit(f"unknown output format {e}")
    if getattr(a, "json", False):
        from .sinks.json_sink import JSONLSink
        sinks.append(JSONLSink())
    plot = None
    if getattr(a, "plot", False):
        from .sinks.plot_sink import PlotSink
        plot = PlotSink(seconds=a.plot_secs, mains=a.mains)
        sinks.append(plot)
    return sinks, plot


async def _run_pipeline(a, fmt: FrameFormat, fs: float):
    from .stream import Pipeline
    sinks, plot = _make_sinks(a, fmt)
    names = ["I", "II"][: fmt.channels] if fmt.channels <= 2 else [f"ch{i}" for i in range(fmt.channels)]
    raw_log = None
    if not a.sim and getattr(a, "out", None) and not a.no_raw:
        raw_log = str(Path(a.out).with_suffix("")) + ".raw.jsonl"
        print(f"raw notifications -> {raw_log}", file=sys.stderr)
    pipe = Pipeline(fmt, fs, sinks, names, derive_leads=a.derive, raw_log=raw_log)
    pipe.start()

    stop_at = time.monotonic() + a.secs if a.secs else None
    if a.sim:
        from dataclasses import replace
        from .sim import SimDevice
        if fmt.gain_mv == 1.0:          # uncalibrated layout: give the sim a plausible gain
            fmt = replace(fmt, gain_mv=0.0025)
            pipe.fmt = fmt
            pipe.decoder.fmt = fmt
        src = SimDevice(fmt, fs, hr_bpm=a.sim_hr, per_notification=a.sim_batch)
        task = asyncio.create_task(src.run(pipe.on_notify))
        closer = None
    else:
        from .connect import Connection, KARDIA_6L_START
        from .protocol import KARDIA_6L_CMD_CHAR, KARDIA_6L_DATA_CHAR
        conn = Connection(address=a.address, pair=a.pair, stall_ms=a.stall_ms,
                          on_stall=lambda ms: log.warning("DROPOUT: %.0f ms", ms))
        await conn.connect()
        chars = a.char or [KARDIA_6L_DATA_CHAR]      # default to the 6L ECG characteristic
        for c in chars:
            await conn.subscribe(c, pipe.on_notify)
        # Send the start handshake: explicit --write overrides; otherwise the captured 6L
        # sequence unless --no-start is given.
        if a.write:
            u, hx = a.write.split("=", 1)
            await conn.write(u, bytes.fromhex(hx))
        elif not a.no_start:
            cmd_char = a.cmd_char or KARDIA_6L_CMD_CHAR
            for cmd in KARDIA_6L_START:
                await conn.write(cmd_char, cmd)
                await asyncio.sleep(0.3)
            log.info("sent 6L start handshake to %s", cmd_char)
        conn.start_watchdog()
        task = None
        closer = conn.close

    try:
        last_report = time.monotonic()
        while True:
            await asyncio.sleep(0.05)
            if plot:
                plot.pump()
            now = time.monotonic()
            if now - last_report >= 2.0:
                last_report = now
                st = pipe.stats
                t, x = pipe.ring.latest(int(5 * fs))
                hr = float("nan")
                if len(x) >= 2 * fs:
                    from .dsp import heart_rate
                    hr = heart_rate(x[:, 0], fs)
                print(f"[{now - st.started:6.1f}s] notif={st.notifications} samples={st.samples} "
                      f"rate={st.rate_est:.1f} Hz gaps={st.seq_gaps} qdrop={st.dropped_queue} "
                      f"HR~{hr:.0f}", file=sys.stderr)
            if stop_at and now >= stop_at:
                break
            if not a.sim and not conn.state.connected:
                log.error("device disconnected")
                break
    except KeyboardInterrupt:
        pass
    finally:
        if task:
            task.cancel()
        if closer:
            await closer()
        await pipe.stop()
        st = pipe.stats
        print(f"done: {st.samples} samples, {st.notifications} notifications, {st.seq_gaps} seq gaps, "
              f"measured rate {st.rate_est:.2f} Hz", file=sys.stderr)


# --------------------------------------------------------------------------- commands
def cmd_scan(a):
    from .scan import scan

    async def go():
        found = await scan(a.secs, only_kardia=not a.all)
        if not found:
            print("nothing found — see README 'macOS Bluetooth permission'.")
        for f in found:
            tag = "KARDIA" if f.looks_like_kardia() else "      "
            print(f"{tag} {f.address} rssi={f.rssi} name={f.name!r} svcs={f.service_uuids} "
                  f"mfg={f.manufacturer_data}")
    asyncio.run(go())


def cmd_dump(a):
    from .scan import find_kardia, gatt_dump, notify_candidates, save_known

    async def go():
        dev = await find_kardia(a.timeout, a.address)
        if not dev:
            sys.exit("Kardia not found (wake it by touching the electrodes)")
        save_known(dev)
        d = await gatt_dump(dev, read_values=True, pair=a.pair)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(d, indent=2))
        print(json.dumps(d, indent=2))
        print(f"\nwrote {a.out}\nnotify candidates:", file=sys.stderr)
        for c in notify_candidates(d):
            print(f"  {'VENDOR' if c['vendor'] else 'SIG   '} {c['uuid']} {c['properties']}", file=sys.stderr)
    asyncio.run(go())


def cmd_sniff(a):
    # Delegate to the standalone tool (kept in tools/ so it works without install).
    from importlib import import_module
    sys.argv = ["gatt_dump.py", "sniff", *(["--address", a.address] if a.address else []),
                "--secs", str(a.secs), "--out", a.out, *(["--pair"] if a.pair else []),
                *(["--echo"] if a.echo else []), *(["--only", a.only] if a.only else []),
                *(["--write", a.write] if a.write else [])]
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    import_module("gatt_dump").main()


def cmd_analyze(a):
    from .analyze import analyze_capture
    analyze_capture(a.path, top=a.top, char=a.char, fs_hint=a.fs, out_json=a.out, plot=a.plot)


def cmd_formats(_a):
    for c in CANDIDATES:
        print(c.name)


def cmd_record(a):
    fmt = _resolve_format(a)
    fs = a.fs or fmt.sample_rate_hz
    if not fs:
        sys.exit("--fs is required until the sample rate is established (measure with `analyze`)")
    if not a.out:
        sys.exit("--out required")
    asyncio.run(_run_pipeline(a, fmt, fs))


def cmd_stream(a):
    fmt = _resolve_format(a)
    fs = a.fs or fmt.sample_rate_hz
    if not fs:
        sys.exit("--fs is required until the sample rate is established")
    asyncio.run(_run_pipeline(a, fmt, fs))


def cmd_validate(a):
    from validate.compare import main as vmain  # type: ignore
    sys.argv = ["compare", *a.rest]
    vmain()


# --------------------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(prog="kardia6l", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"kardia6l {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("scan"); s.add_argument("--secs", type=float, default=10); s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_scan)

    d = sp.add_parser("dump"); d.add_argument("--address"); d.add_argument("--timeout", type=float, default=20)
    d.add_argument("--pair", action="store_true"); d.add_argument("--out", default="captures/gatt_dump.json")
    d.set_defaults(fn=cmd_dump)

    n = sp.add_parser("sniff"); n.add_argument("--address"); n.add_argument("--secs", type=float, default=30)
    n.add_argument("--out", default="captures/raw_notifications.jsonl"); n.add_argument("--pair", action="store_true")
    n.add_argument("--echo", action="store_true"); n.add_argument("--only"); n.add_argument("--write")
    n.set_defaults(fn=cmd_sniff)

    an = sp.add_parser("analyze"); an.add_argument("path"); an.add_argument("--top", type=int, default=10)
    an.add_argument("--char", help="only this characteristic uuid (substring)"); an.add_argument("--fs", type=float)
    an.add_argument("--out", help="write ranking JSON here"); an.add_argument("--plot", action="store_true")
    an.set_defaults(fn=cmd_analyze)

    fm = sp.add_parser("formats"); fm.set_defaults(fn=cmd_formats)

    for name, fn in (("record", cmd_record), ("stream", cmd_stream)):
        r = sp.add_parser(name)
        r.add_argument("--secs", type=float, default=30 if name == "record" else 0)
        r.add_argument("--out"); r.add_argument("--formats", default="csv", help="csv,wfdb,edf")
        r.add_argument("--format", help="FrameFormat name or JSON file (until KNOWN_FORMAT is set)")
        r.add_argument("--fs", type=float); r.add_argument("--char", action="append")
        r.add_argument("--address"); r.add_argument("--pair", action="store_true")
        r.add_argument("--write", help="UUID=hex start command (overrides the built-in 6L handshake)")
        r.add_argument("--no-start", action="store_true", help="don't send the 6L start handshake")
        r.add_argument("--cmd-char", help="command characteristic UUID (default: 6L vendor cmd char)")
        r.add_argument("--stall-ms", type=float, default=1000)
        r.add_argument("--derive", action="store_true", help="derive III/aVR/aVL/aVF from I,II")
        r.add_argument("--json", action="store_true"); r.add_argument("--plot", action="store_true")
        r.add_argument("--plot-secs", type=float, default=6); r.add_argument("--mains", type=float, default=60)
        r.add_argument("--sim", action="store_true", help="synthetic device (no hardware)")
        r.add_argument("--no-raw", action="store_true", help="don't keep a raw-notification side log")
        r.add_argument("--sim-hr", type=float, default=72); r.add_argument("--sim-batch", type=int, default=25)
        r.set_defaults(fn=fn)

    v = sp.add_parser("validate"); v.add_argument("rest", nargs=argparse.REMAINDER); v.set_defaults(fn=cmd_validate)
    return p


def main(argv=None):
    print_disclaimer()
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    a.fn(a)


if __name__ == "__main__":
    main()
