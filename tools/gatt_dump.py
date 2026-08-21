#!/usr/bin/env python
"""Standalone Phase-1 tool: scan, connect to the Kardia, enumerate every
characteristic, optionally subscribe to all notify candidates and log raw bytes.

Usage:
  python tools/gatt_dump.py scan [--secs 10] [--all]
  python tools/gatt_dump.py dump [--address UUID] [--pair] [--out captures/gatt.json]
  python tools/gatt_dump.py sniff [--address UUID] [--secs 30] [--out captures/raw.jsonl]
        (subscribes to every notify/indicate characteristic and logs (t, uuid, hex))
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bleak import BleakClient  # noqa: E402

from open_kardia import print_disclaimer  # noqa: E402
from open_kardia.scan import find_kardia, gatt_dump, notify_candidates, save_known, scan  # noqa: E402


async def cmd_scan(a):
    print(f"scanning {a.secs}s ...", file=sys.stderr)
    found = await scan(a.secs, only_kardia=not a.all)
    if not found:
        print("nothing found. Check: Bluetooth on, device awake (touch electrodes), and "
              "the terminal has Bluetooth permission in System Settings > Privacy & Security.")
        return
    for f in found:
        tag = "*KARDIA*" if f.looks_like_kardia() else "        "
        print(f"{tag} {f.address}  rssi={f.rssi}  name={f.name!r}  svcs={f.service_uuids}  "
              f"mfg={f.manufacturer_data}  n={f.n_adv}")


async def _pick(a):
    dev = await find_kardia(a.timeout, a.address)
    if not dev:
        print("Kardia not found. Wake it (touch the electrodes) and retry, or pass --address.",
              file=sys.stderr)
        sys.exit(2)
    print(f"using {dev.address} name={dev.name!r}", file=sys.stderr)
    save_known(dev)
    return dev


async def cmd_dump(a):
    dev = await _pick(a)
    d = await gatt_dump(dev, read_values=not a.no_read, pair=a.pair)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2))
    print(json.dumps(d, indent=2))
    print(f"\nwrote {a.out}", file=sys.stderr)
    print("\nnotify/indicate candidates:", file=sys.stderr)
    for c in notify_candidates(d):
        print(f"  {'VENDOR ' if c['vendor'] else 'SIG    '}{c['uuid']}  props={c['properties']}",
              file=sys.stderr)


async def cmd_sniff(a):
    dev = await _pick(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "a")
    counts: dict[str, int] = {}
    sizes: dict[str, set] = {}
    t0 = time.time()

    def mk(uuid):
        def cb(_, data: bytearray):
            t = time.time()
            counts[uuid] = counts.get(uuid, 0) + 1
            sizes.setdefault(uuid, set()).add(len(data))
            fh.write(json.dumps({"t": t, "uuid": uuid, "len": len(data), "hex": data.hex()}) + "\n")
            if a.echo:
                print(f"{t - t0:8.3f} {uuid[:8]} len={len(data):3d} {data.hex()}")
        return cb

    async with BleakClient(dev, timeout=a.timeout) as client:
        if a.pair:
            try:
                print("pair ->", await client.pair(), file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print("pair error:", e, file=sys.stderr)
        subs = []
        for svc in client.services:
            for ch in svc.characteristics:
                if {"notify", "indicate"} & set(ch.properties):
                    if a.only and a.only.lower() not in ch.uuid.lower():
                        continue
                    try:
                        await client.start_notify(ch, mk(ch.uuid))
                        subs.append(ch.uuid)
                        print(f"subscribed {ch.uuid} ({','.join(ch.properties)})", file=sys.stderr)
                    except Exception as e:  # noqa: BLE001
                        print(f"FAILED subscribe {ch.uuid}: {e}", file=sys.stderr)
        if a.write:
            uuid, hexdata = a.write.split("=", 1)
            print(f"writing {hexdata} -> {uuid}", file=sys.stderr)
            await client.write_gatt_char(uuid, bytes.fromhex(hexdata), response=True)
        print(f"listening {a.secs}s — put fingers on the electrodes now", file=sys.stderr)
        end = time.time() + a.secs
        while time.time() < end and client.is_connected:
            await asyncio.sleep(1.0)
            el = time.time() - t0
            summary = "  ".join(f"{u[:8]}:{n} ({n/el:.1f}/s)" for u, n in counts.items())
            print(f"[{el:5.1f}s] {summary or 'no notifications yet'}", file=sys.stderr)
        for u in subs:
            try:
                await client.stop_notify(u)
            except Exception:  # noqa: BLE001
                pass
    fh.close()
    print("\nsummary:", file=sys.stderr)
    for u, n in counts.items():
        print(f"  {u}: {n} notifications, sizes={sorted(sizes[u])}, "
              f"rate={n / a.secs:.2f}/s, bytes/s={sum(sizes[u]) / len(sizes[u]) * n / a.secs:.1f}",
              file=sys.stderr)
    print(f"wrote {a.out}", file=sys.stderr)


def main():
    print_disclaimer()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("--secs", type=float, default=10); s.add_argument("--all", action="store_true")
    for name in ("dump", "sniff"):
        q = sp.add_parser(name)
        q.add_argument("--address"); q.add_argument("--timeout", type=float, default=20)
        q.add_argument("--pair", action="store_true")
        if name == "dump":
            q.add_argument("--out", default="captures/gatt_dump.json"); q.add_argument("--no-read", action="store_true")
        else:
            q.add_argument("--secs", type=float, default=30); q.add_argument("--out", default="captures/raw_notifications.jsonl")
            q.add_argument("--echo", action="store_true"); q.add_argument("--only", help="uuid substring filter")
            q.add_argument("--write", help="UUID=hexbytes to write after subscribing (e.g. a start command)")
    a = p.parse_args()
    asyncio.run({"scan": cmd_scan, "dump": cmd_dump, "sniff": cmd_sniff}[a.cmd](a))


if __name__ == "__main__":
    main()
