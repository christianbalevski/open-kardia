#!/usr/bin/env python
"""Replay the captured 6L start handshake and capture ECG notifications live.
Proves the streaming client works without the Kardia app. Writes only to the vendor
command characteristic (never DFU)."""
import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bleak import BleakClient, BleakScanner
from open_kardia import print_disclaimer
from open_kardia.connect import KARDIA_6L_START
from open_kardia.protocol import KARDIA_6L_CMD_CHAR as CMD, KARDIA_6L_DATA_CHAR as DATA

ADDR = sys.argv[1]
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

async def main():
    print_disclaimer()
    dev = await BleakScanner.find_device_by_address(ADDR, timeout=30)
    if not dev:
        sys.exit("device not advertising — touch the electrodes")
    out = open("captures/replay_ecg.jsonl", "w")
    t0 = time.time(); n = {"CMD": 0, "DATA": 0}; first_data = [None]
    def cb(tag):
        def f(_, d):
            n[tag] += 1
            if tag == "DATA":
                if first_data[0] is None:
                    first_data[0] = time.time() - t0
                out.write(json.dumps({"t": time.time(), "uuid": DATA, "len": len(d), "hex": bytes(d).hex()}) + "\n")
            else:
                print(f"  CMD <- {bytes(d).hex()}")
        return f
    async with BleakClient(dev, timeout=30) as c:
        print("connected mtu", c.mtu_size)
        await c.start_notify(CMD, cb("CMD"))
        await c.start_notify(DATA, cb("DATA"))
        for cmd in KARDIA_6L_START:
            print(f"-> write {cmd.hex()} ({cmd!r})")
            await c.write_gatt_char(CMD, cmd, response=True)
            await asyncio.sleep(0.3)
        print(f"streaming for {SECS}s — fingers on the electrodes")
        end = time.time() + SECS
        while time.time() < end and c.is_connected:
            await asyncio.sleep(1)
            el = time.time() - t0
            print(f"  [{el:5.1f}s] CMD={n['CMD']} DATA={n['DATA']} ({n['DATA']/max(el,1e-9):.1f}/s)")
        if not c.is_connected:
            print("!! device disconnected")
    out.close()
    print(f"\nfirst DATA at {first_data[0]}s, {n['DATA']} notifications -> captures/replay_ecg.jsonl")
    if n["DATA"] == 0:
        print("NO ECG DATA: the token after 'K' is likely session/device-specific; re-capture with PacketLogger.")
asyncio.run(main())
