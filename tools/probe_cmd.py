#!/usr/bin/env python
"""Phase-1 probe: connect, subscribe to the vendor cmd (indicate) + data (notify)
characteristics, then write candidate commands to the cmd characteristic and log
every byte that comes back. Never touches the DFU characteristic."""
import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bleak import BleakClient, BleakScanner
from open_kardia import print_disclaimer

ADDR = sys.argv[1]
CMD = "ac060002-328c-a28f-9846-5a8aa212661b"
DATA = "ac060003-328c-a28f-9846-5a8aa212661b"
CMDS = [bytes.fromhex(h) for h in sys.argv[2].split(",")] if len(sys.argv) > 2 else \
       [b"\x01", b"\x00", b"\x02", b"\x01\x00", b"\x01\x01", b"\x00\x01", b"\x03", b"\x10", b"\xa0", b"\x01\x00\x00\x00"]
PAIR = "--pair" in sys.argv
WAIT = 2.5

async def main():
    print_disclaimer()
    dev = await BleakScanner.find_device_by_address(ADDR, timeout=30)
    if not dev:
        sys.exit("not found — wake it")
    out = open("captures/probe_cmd.jsonl", "a")
    t0 = time.time()
    def mk(tag):
        def cb(_, d: bytearray):
            print(f"  <- {time.time()-t0:6.2f}s {tag} len={len(d)} {bytes(d).hex()}")
            out.write(json.dumps({"t": time.time(), "uuid": tag, "len": len(d), "hex": bytes(d).hex()}) + "\n"); out.flush()
        return cb
    async with BleakClient(dev, timeout=30) as c:
        print("connected mtu", c.mtu_size)
        if PAIR:
            try: print("pair ->", await c.pair())
            except Exception as e: print("pair err", e)
        await c.start_notify(CMD, mk("CMD"))
        await c.start_notify(DATA, mk("DATA"))
        for cmd in CMDS:
            if not c.is_connected:
                print("disconnected"); break
            print(f"-> write {cmd.hex()} to CMD")
            out.write(json.dumps({"t": time.time(), "uuid": "WRITE", "hex": cmd.hex()}) + "\n")
            try:
                await c.write_gatt_char(CMD, cmd, response=True)
            except Exception as e:
                print("   write error:", e)
            await asyncio.sleep(WAIT)
        print("holding 5s more"); await asyncio.sleep(5)
asyncio.run(main())
