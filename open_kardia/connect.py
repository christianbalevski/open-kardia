"""Async connection manager for the 6L.

* discover by name hint or cached macOS peripheral UUID (~/.open-kardia/known.json)
* optional pair/bond (spec §5: link-layer encryption is transparent once bonded)
* reconnect with exponential backoff
* watchdog: flag notification stalls > `stall_ms` mid-recording
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from bleak import BleakClient
from bleak.backends.device import BLEDevice

from .scan import find_kardia, load_known, save_known

# 6L start handshake (ASCII writes to the command characteristic), captured from the app.
# NOTE: the 16-hex-char token after 'K' may be session/device-specific; if a live replay
# is rejected, re-capture. Verified by live replay on 2026-08-16.
KARDIA_6L_START = [bytes.fromhex("4b45303743354645303145394438353244"), b"M2"]

log = logging.getLogger("open_kardia.connect")

NotifyCB = Callable[[str, float, bytes], None]  # (char_uuid, t_monotonic, payload)


@dataclass
class ConnState:
    device: BLEDevice | None = None
    connected: bool = False
    connects: int = 0
    disconnects: int = 0
    stalls: int = 0
    last_notify_t: float | None = None
    subscribed: list[str] = field(default_factory=list)


class Connection:
    def __init__(self, address: str | None = None, pair: bool = False, timeout: float = 20.0,
                 stall_ms: float = 1000.0, max_backoff: float = 30.0,
                 on_stall: Callable[[float], None] | None = None,
                 on_disconnect: Callable[[], None] | None = None):
        self.address = address
        self.pair = pair
        self.timeout = timeout
        self.stall_ms = stall_ms
        self.max_backoff = max_backoff
        self.on_stall = on_stall
        self.on_disconnect = on_disconnect
        self.state = ConnState()
        self._client: BleakClient | None = None
        self._disconnected = asyncio.Event()
        self._watchdog: asyncio.Task | None = None
        self._recording = False

    # ------------------------------------------------------------------ discovery
    async def discover(self, scan_timeout: float = 15.0) -> BLEDevice:
        addr = self.address
        if addr is None:
            known = load_known()
            addr = known.get("last")
        dev = None
        if addr:
            log.info("looking for cached peripheral %s", addr)
            dev = await find_kardia(scan_timeout, address=addr)
            if dev is None:
                log.info("cached peripheral not seen; scanning by name")
        if dev is None:
            dev = await find_kardia(scan_timeout)
        if dev is None:
            raise RuntimeError("KardiaMobile not found. Wake it (touch both electrodes), make sure "
                               "the Kardia phone app is NOT connected to it, and retry.")
        save_known(dev)
        self.state.device = dev
        self.address = dev.address
        return dev

    # ------------------------------------------------------------------ connect
    def _on_disc(self, _client):
        self.state.connected = False
        self.state.disconnects += 1
        log.warning("disconnected")
        self._disconnected.set()
        if self.on_disconnect:
            self.on_disconnect()

    async def connect(self) -> BleakClient:
        if self.state.device is None:
            await self.discover()
        backoff = 1.0
        while True:
            try:
                client = BleakClient(self.state.device, timeout=self.timeout,
                                     disconnected_callback=self._on_disc)
                await client.connect()
                if self.pair:
                    try:
                        ok = await client.pair()
                        log.info("pair -> %s", ok)
                    except Exception as e:  # noqa: BLE001
                        log.warning("pair failed/unsupported: %s", e)
                self._client = client
                self._disconnected.clear()
                self.state.connected = True
                self.state.connects += 1
                log.info("connected to %s (mtu=%s)", self.state.device.address, client.mtu_size)
                return client
            except Exception as e:  # noqa: BLE001
                log.warning("connect failed: %s — retry in %.0fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(self.max_backoff, backoff * 2)
                # device may have gone to sleep / changed; rescan
                try:
                    await self.discover()
                except Exception as e2:  # noqa: BLE001
                    log.warning("rediscover failed: %s", e2)

    @property
    def client(self) -> BleakClient:
        if self._client is None:
            raise RuntimeError("not connected")
        return self._client

    # ------------------------------------------------------------------ notify
    async def subscribe(self, char_uuid: str, cb: NotifyCB) -> None:
        def _cb(_sender, data: bytearray):
            self.state.last_notify_t = time.monotonic()
            cb(char_uuid, self.state.last_notify_t, bytes(data))
        await self.client.start_notify(char_uuid, _cb)
        self.state.subscribed.append(char_uuid)

    async def write(self, char_uuid: str, data: bytes, response: bool = True) -> None:
        await self.client.write_gatt_char(char_uuid, data, response=response)

    async def read(self, char_uuid: str) -> bytes:
        return bytes(await self.client.read_gatt_char(char_uuid))

    # ------------------------------------------------------------------ watchdog
    def start_watchdog(self):
        self._recording = True
        self._watchdog = asyncio.create_task(self._watch())

    async def _watch(self):
        stalled = False
        while self._recording:
            await asyncio.sleep(self.stall_ms / 1000.0 / 2)
            t = self.state.last_notify_t
            if t is None or not self.state.connected:
                continue
            gap_ms = (time.monotonic() - t) * 1000.0
            if gap_ms > self.stall_ms and not stalled:
                stalled = True
                self.state.stalls += 1
                log.warning("notification stall: %.0f ms without data", gap_ms)
                if self.on_stall:
                    self.on_stall(gap_ms)
            elif gap_ms <= self.stall_ms:
                stalled = False

    async def wait_disconnected(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._disconnected.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def close(self):
        self._recording = False
        if self._watchdog:
            self._watchdog.cancel()
        if self._client is not None:
            for u in list(self.state.subscribed):
                try:
                    await self._client.stop_notify(u)
                except Exception:  # noqa: BLE001
                    pass
            try:
                await self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        self.state.connected = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()


async def run_with_reconnect(conn: Connection, session: Callable[[Connection], Awaitable[None]],
                             max_reconnects: int = 5) -> None:
    """Run `session(conn)`; on unexpected disconnect, reconnect and re-run up to
    `max_reconnects` times. `session` should be idempotent w.r.t. subscriptions."""
    tries = 0
    while True:
        await conn.connect()
        try:
            await session(conn)
            return
        except Exception as e:  # noqa: BLE001
            if conn.state.connected:
                raise
            tries += 1
            if tries > max_reconnects:
                raise RuntimeError(f"gave up after {tries - 1} reconnects: {e}") from e
            log.warning("session interrupted (%s); reconnecting (%d/%d)", e, tries, max_reconnects)
