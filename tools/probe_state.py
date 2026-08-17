import asyncio, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bleak import BleakClient, BleakScanner
ADDR = sys.argv[1]
CMD = "ac060002-328c-a28f-9846-5a8aa212661b"; DATA = "ac060003-328c-a28f-9846-5a8aa212661b"
R5 = "ac060005-328c-a28f-9846-5a8aa212661b"; R6 = "ac060006-328c-a28f-9846-5a8aa212661b"; BAT="00002a19-0000-1000-8000-00805f9b34fb"
async def main():
    dev = await BleakScanner.find_device_by_address(ADDR, timeout=30)
    t0 = time.time()
    async with BleakClient(dev, timeout=30) as c:
        def cb(tag):
            def f(_, d): print(f"  <- {time.time()-t0:5.2f} {tag} {bytes(d).hex()}")
            return f
        await c.start_notify(CMD, cb("CMD")); await c.start_notify(DATA, cb("DATA"))
        async def rd(label):
            a = (await c.read_gatt_char(R5)).hex(); b = (await c.read_gatt_char(R6)).hex(); bat = (await c.read_gatt_char(BAT)).hex()
            print(f"  [{label}] 0005={a} 0006={b} bat={bat}  t={time.time()-t0:.2f}")
        await rd("initial")
        for cmd in ("2000", "2c00", "00", "2000"):
            print(f"-> {cmd}"); await c.write_gatt_char(CMD, bytes.fromhex(cmd), response=True); await asyncio.sleep(0.5); await rd("after " + cmd)
        print("holding..."); 
        while c.is_connected and time.time() - t0 < 30:
            await asyncio.sleep(1)
        print(f"disconnected/ended at {time.time()-t0:.1f}s")
asyncio.run(main())
