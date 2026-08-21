"""BLE discovery + full GATT enumeration (Phase 1 tooling).

macOS note: CoreBluetooth exposes peripherals by a system-assigned UUID, not a
MAC address. `BLEDevice.address` on macOS *is* that UUID; we cache it in
~/.open-kardia/known.json for reconnects.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

from . import data_dir

KNOWN_PATH = data_dir() / "known.json"

# Name fragments the 6L (and other AliveCor devices) have been observed / expected
# to advertise under. Case-insensitive substring match. Extend as discovered.
NAME_HINTS = ("kardia", "alivecor", "ac-019", "km6l", "6l")


@dataclass
class Found:
    address: str                    # macOS: peripheral UUID
    name: str | None
    rssi: int | None
    service_uuids: list[str] = field(default_factory=list)
    manufacturer_data: dict[str, str] = field(default_factory=dict)  # company_id -> hex
    service_data: dict[str, str] = field(default_factory=dict)
    tx_power: int | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    n_adv: int = 0

    def looks_like_kardia(self) -> bool:
        n = (self.name or "").lower()
        return any(h in n for h in NAME_HINTS)


def _adv_to_found(dev: BLEDevice, adv: AdvertisementData, prev: Found | None) -> Found:
    now = time.time()
    f = Found(
        address=dev.address,
        name=adv.local_name or dev.name,
        rssi=adv.rssi,
        service_uuids=sorted(set(adv.service_uuids or [])),
        manufacturer_data={hex(k): v.hex() for k, v in (adv.manufacturer_data or {}).items()},
        service_data={k: v.hex() for k, v in (adv.service_data or {}).items()},
        tx_power=adv.tx_power,
        first_seen=prev.first_seen if prev else now,
        last_seen=now,
        n_adv=(prev.n_adv if prev else 0) + 1,
    )
    if prev:
        f.service_uuids = sorted(set(prev.service_uuids) | set(f.service_uuids))
        f.manufacturer_data = {**prev.manufacturer_data, **f.manufacturer_data}
        f.service_data = {**prev.service_data, **f.service_data}
        if not f.name:
            f.name = prev.name
    return f


async def scan(timeout: float = 10.0, only_kardia: bool = False, verbose_cb=None) -> list[Found]:
    """Scan for `timeout` seconds and return everything seen (merged per address)."""
    seen: dict[str, Found] = {}

    def cb(dev: BLEDevice, adv: AdvertisementData):
        f = _adv_to_found(dev, adv, seen.get(dev.address))
        seen[dev.address] = f
        if verbose_cb:
            verbose_cb(f)

    async with BleakScanner(cb) as _:
        await asyncio.sleep(timeout)

    out = list(seen.values())
    if only_kardia:
        out = [f for f in out if f.looks_like_kardia()]
    out.sort(key=lambda f: (not f.looks_like_kardia(), -(f.rssi or -999)))
    return out


async def find_kardia(timeout: float = 15.0, address: str | None = None) -> BLEDevice | None:
    """Return the first BLEDevice that looks like a Kardia (or matches `address`)."""
    if address:
        return await BleakScanner.find_device_by_address(address, timeout=timeout)

    def flt(dev: BLEDevice, adv: AdvertisementData) -> bool:
        n = (adv.local_name or dev.name or "").lower()
        return any(h in n for h in NAME_HINTS)

    return await BleakScanner.find_device_by_filter(flt, timeout=timeout)


# --------------------------------------------------------------------------- GATT dump
async def gatt_dump(device: BLEDevice | str, read_values: bool = True, timeout: float = 20.0,
                    pair: bool = False) -> dict[str, Any]:
    """Connect and enumerate every service/characteristic/descriptor.

    Returns a JSON-serialisable dict. Reads readable characteristic values (hex).
    """
    out: dict[str, Any] = {"address": device if isinstance(device, str) else device.address,
                           "name": None if isinstance(device, str) else device.name,
                           "timestamp": time.time(), "services": []}
    async with BleakClient(device, timeout=timeout) as client:
        out["connected"] = client.is_connected
        try:
            out["mtu"] = client.mtu_size
        except Exception:
            out["mtu"] = None
        if pair:
            try:
                out["paired"] = await client.pair()
            except Exception as e:  # noqa: BLE001
                out["paired"] = f"error: {e}"
        for svc in client.services:
            s: dict[str, Any] = {"uuid": svc.uuid, "handle": svc.handle,
                                 "description": svc.description, "characteristics": []}
            for ch in svc.characteristics:
                c: dict[str, Any] = {"uuid": ch.uuid, "handle": ch.handle,
                                     "description": ch.description,
                                     "properties": list(ch.properties), "descriptors": []}
                if read_values and "read" in ch.properties:
                    try:
                        val = await client.read_gatt_char(ch)
                        c["value_hex"] = val.hex()
                        try:
                            c["value_utf8"] = val.decode("utf-8")
                        except UnicodeDecodeError:
                            pass
                    except Exception as e:  # noqa: BLE001
                        c["value_error"] = str(e)
                for d in ch.descriptors:
                    dd: dict[str, Any] = {"uuid": d.uuid, "handle": d.handle,
                                          "description": d.description}
                    if read_values:
                        try:
                            v = await client.read_gatt_descriptor(d.handle)
                            dd["value_hex"] = v.hex()
                        except Exception as e:  # noqa: BLE001
                            dd["value_error"] = str(e)
                    c["descriptors"].append(dd)
                s["characteristics"].append(c)
            out["services"].append(s)
    return out


def notify_candidates(dump: dict[str, Any]) -> list[dict[str, Any]]:
    """Characteristics that can notify/indicate, vendor-specific first."""
    cands = []
    for s in dump.get("services", []):
        for c in s.get("characteristics", []):
            props = set(c.get("properties", []))
            if props & {"notify", "indicate"}:
                cands.append({"service": s["uuid"], "uuid": c["uuid"], "handle": c["handle"],
                              "properties": sorted(props),
                              "vendor": not _is_sig_uuid(c["uuid"])})
    cands.sort(key=lambda c: (not c["vendor"], c["handle"]))
    return cands


def _is_sig_uuid(u: str) -> bool:
    u = u.lower()
    return u.endswith("-0000-1000-8000-00805f9b34fb") and u.startswith("0000")


# --------------------------------------------------------------------------- known cache
def load_known() -> dict[str, Any]:
    if KNOWN_PATH.exists():
        try:
            return json.loads(KNOWN_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_known(found: Found | BLEDevice, extra: dict[str, Any] | None = None) -> None:
    KNOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    known = load_known()
    if isinstance(found, Found):
        rec = asdict(found)
    else:
        rec = {"address": found.address, "name": found.name}
    rec["saved_at"] = time.time()
    if extra:
        rec.update(extra)
    known[rec["address"]] = rec
    known["last"] = rec["address"]
    KNOWN_PATH.write_text(json.dumps(known, indent=2))
