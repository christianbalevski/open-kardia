#!/usr/bin/env python
"""Write an explicit list of hex commands to the vendor cmd characteristic (reconnecting
as needed) and print all replies/notifications. Never touches DFU."""
import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bleak import BleakClient, BleakScanner
from open_kardia import print_disclaimer
ADDR = sys.argv[1]
CMDS = [bytes.fromhex(h) for h in sys.argv[2].split(",")]
WAIT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
HOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
CMD = "ac060002-328c-a28f-9846-5a8aa212661b"; DATA = "ac060003-328c-a28f-9846-5a8aa212661b"

async def main():
    print_disclaimer()
    out = open("captures/probe_list.jsonl", "a")
    i = 0
    while i < len(CMDS):
        dev = await BleakScanner.find_device_by_address(ADDR, timeout=30)
        if not dev:
            print("device not advertising — touch electrodes"); await asyncio.sleep(2); continue
        got = asyncio.Queue()
        def cb(tag):
            def f(_, d): got.put_nowait((time.time(), tag, bytes(d)))
            return f
        try:
            async with BleakClient(dev, timeout=30) as c:
                await c.start_notify(CMD, cb("CMD")); await c.start_notify(DATA, cb("DATA"))
                t_conn = time.time()
                while i < len(CMDS) and c.is_connected:
                    cmd = CMDS[i]
                    try:
                        await c.write_gatt_char(CMD, cmd, response=True)
                    except Exception as e:
                        print(f"{cmd.hex()}: write err {e}"); break
                    await asyncio.sleep(WAIT)
                    reps = []
                    while not got.empty(): reps.append(got.get_nowait())
                    rs = " ".join(f"{t}:{d.hex()}" for _, t, d in reps)
                    print(f"{cmd.hex():>20} -> {rs}")
                    out.write(json.dumps({"t": time.time(), "cmd": cmd.hex(), "replies": [[t, d.hex()] for _, t, d in reps]}) + "\n"); out.flush()
                    i += 1
                    if any(t == "DATA" for _, t, d in reps):
                        print(f"!!! DATA notification — holding {HOLD}s"); await asyncio.sleep(HOLD)
                        n = 0
                        while not got.empty():
                            ts, t, d = got.get_nowait(); n += 1
                            if n <= 12: print(f"   {ts-t_conn:6.2f} {t} len={len(d)} {d.hex()}")
                            out.write(json.dumps({"t": ts, "uuid": t, "len": len(d), "hex": d.hex()}) + "\n")
                        print(f"   ... {n} packets total during hold")
                print(f"(connection lasted {time.time()-t_conn:.1f}s)")
        except Exception as e:
            print("conn error:", e); await asyncio.sleep(1)
asyncio.run(main())
