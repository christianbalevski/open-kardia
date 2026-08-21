#!/usr/bin/env python
"""Opcode sweep over the vendor command characteristic; reconnects as needed.
Logs (cmd -> reply) to captures/probe_sweep.jsonl. Never touches DFU."""
import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bleak import BleakClient, BleakScanner
from open_kardia import print_disclaimer
ADDR = sys.argv[1]
START = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0
END = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x100
PREFIX = bytes.fromhex(sys.argv[4]) if len(sys.argv) > 4 else b""
CMD = "ac060002-328c-a28f-9846-5a8aa212661b"; DATA = "ac060003-328c-a28f-9846-5a8aa212661b"
WAIT = 0.35

async def main():
    print_disclaimer()
    out = open("captures/probe_sweep.jsonl", "a")
    results = {}
    op = START
    while op < END:
        dev = await BleakScanner.find_device_by_address(ADDR, timeout=30)
        if not dev:
            print("device not advertising — touch electrodes"); await asyncio.sleep(2); continue
        pending = {}
        got = asyncio.Queue()
        def cb(tag):
            def f(_, d):
                got.put_nowait((tag, bytes(d)))
            return f
        try:
            async with BleakClient(dev, timeout=30) as c:
                await c.start_notify(CMD, cb("CMD")); await c.start_notify(DATA, cb("DATA"))
                t_conn = time.time()
                while op < END and c.is_connected:
                    cmd = PREFIX + bytes([op])
                    try:
                        await c.write_gatt_char(CMD, cmd, response=True)
                    except Exception as e:
                        print(f"{cmd.hex()}: write err {e}"); break
                    await asyncio.sleep(WAIT)
                    reps = []
                    while not got.empty():
                        reps.append(got.get_nowait())
                    rs = " ".join(f"{t}:{d.hex()}" for t, d in reps)
                    results[cmd.hex()] = rs
                    print(f"{cmd.hex():>6} -> {rs}")
                    out.write(json.dumps({"t": time.time(), "cmd": cmd.hex(), "replies": [[t, d.hex()] for t, d in reps]}) + "\n"); out.flush()
                    op += 1
                    if any(t == "DATA" for t, _ in reps):
                        print("!!! DATA notification received — holding 8 s to collect"); await asyncio.sleep(8)
                        while not got.empty():
                            t, d = got.get_nowait(); print(f"   {t} {d.hex()}")
                            out.write(json.dumps({"t": time.time(), "cmd": cmd.hex(), "extra": [t, d.hex()]}) + "\n")
                print(f"(connection lasted {time.time()-t_conn:.1f}s)")
        except Exception as e:
            print("conn error:", e); await asyncio.sleep(1)
    print("\nnon-03 replies:")
    for k, v in results.items():
        if v and "CMD:03" not in v:
            print(f"  {k} -> {v}")
asyncio.run(main())
