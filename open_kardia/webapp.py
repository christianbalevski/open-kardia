"""open_kardia heart app — browser front-end with a live 3D heart that beats with the ECG.

Research tool. NOT a medical device, NOT for diagnosis.

Architecture
------------
* This module is a small aiohttp server on 127.0.0.1 that runs the open_kardia pipeline
  through :class:`open_kardia.appcore.AppController` on the MAIN-thread asyncio loop (bleak /
  CoreBluetooth on macOS want the main thread) and serves the single-page UI in
  ``open_kardia/web/`` (vanilla JS + three.js, all vendored — no CDN, works offline).
* Everything on the LEFT of the page (connect, readouts, scrolling 6-lead ECG, quality-gated
  recording, saved-readings library, playback transport) talks to this server over one
  WebSocket (``/ws``) plus a few JSON endpoints (``/api/...``).
* The RIGHT side is the heart. It is driven purely by *timing*: R-peak times plus the
  P/QRS/T fiducials of the reading (see :mod:`open_kardia.beats`), so the atria contract after
  the P wave, the ventricles after the QRS, and relaxation follows the T wave — for the live
  stream and for replayed readings alike, at any playback speed.

Live sync: the server ships display-filtered samples that lag the newest sample by
``LIVE_LAG_S`` (so the zero-phase display filter and the R-peak detector are both settled),
and the browser holds a small extra jitter buffer. Total on-screen latency is < 1 s, and the
ECG trace and the heart are always rendered off the same clock, so they cannot drift apart.

Run:
    python -m open_kardia.webapp             # opens http://127.0.0.1:8765 in your browser
    python -m open_kardia.webapp --sim       # start with the simulated device pre-selected
    python -m open_kardia.webapp --no-browser --port 9000
"""
from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from . import beats as beats_mod
from . import data_dir
from . import dsp, print_disclaimer
from .appcore import AppController

WEB_DIR = Path(__file__).parent / "web"
DEFAULT_PORT = 8765

LIVE_TICK_S = 0.05        # 20 Hz sample push
LIVE_LAG_S = 0.5          # ship filtered samples this far behind the newest one
LIVE_WINDOW_S = 12.0      # ring slice fetched per tick (filter context)
BEATS_EVERY_TICKS = 5     # R-peak refresh (250 ms)
STATS_EVERY_TICKS = 8     # HR/SQI/elapsed refresh (400 ms)
DELINEATE_EVERY_S = 4.0   # median-beat delineation refresh (runs in a worker thread)


# ----------------------------------------------------------------------------- json helpers
def _jsonable(o):
    """Recursively convert dataclasses / numpy / NaN into JSON-safe values."""
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if is_dataclass(o) and not isinstance(o, type):
        if hasattr(o, "to_json"):
            return _jsonable(o.to_json())
        return {k: _jsonable(v) for k, v in asdict(o).items()}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    return str(o)


def analysis_to_json(res) -> dict | None:
    """AnalysisResult -> the dict the browser's results panel renders."""
    if res is None:
        return None
    rhythm = getattr(res, "rhythm", None)
    rows = []
    try:
        rows = list(res.rows())
    except Exception:  # noqa: BLE001
        rows = []
    iv = getattr(res, "intervals", None)
    hrv = getattr(res, "hrv", {}) or {}
    return _jsonable({
        "rhythm": {
            "label": getattr(rhythm, "label", "No analysis") if rhythm else "No analysis",
            "confidence": getattr(rhythm, "confidence", 0.0) if rhythm else 0.0,
            "rationale": getattr(rhythm, "rationale", "") if rhythm else "",
            "flags": list(getattr(rhythm, "flags", []) or []) if rhythm else [],
        },
        "rows": rows,
        "axis_label": getattr(res, "axis_label", ""),
        "context": {
            "rr_mean_ms": getattr(iv, "rr_mean_ms", None), "rr_sd_ms": getattr(iv, "rr_sd_ms", None),
            "hr_min": getattr(iv, "hr_min", None), "hr_max": getattr(iv, "hr_max", None),
            "SDNN": hrv.get("SDNN"), "RMSSD": hrv.get("RMSSD"), "pNN50": hrv.get("pNN50"),
            "n_beats": getattr(res, "n_beats", None), "lead_used": getattr(res, "lead_used", ""),
            "quality_sqi": getattr(res, "quality_sqi", None),
        },
        "warnings": list(getattr(res, "warnings", []) or []),
        "summary": res.summary() if hasattr(res, "summary") else "",
        "arrhythmia": _arrhythmia_json(getattr(res, "arrhythmia", None), getattr(res, "af", None)),
        "af": _af_json(getattr(res, "af", None)),
    })


def _af_json(af) -> dict | None:
    if af is None:
        return None
    return {"score": getattr(af, "score", None), "is_af": bool(getattr(af, "is_af", False)),
            "episodes": [(float(a), float(b), float(c)) for a, b, c in (getattr(af, "episodes", []) or [])],
            "confidence": getattr(af, "confidence", None), "label": getattr(af, "label", ""),
            "evidence": getattr(af, "evidence", ""), "metrics": dict(getattr(af, "metrics", {}) or {}),
            "warnings": list(getattr(af, "warnings", []) or [])}


def _arrhythmia_json(arr, af=None) -> dict | None:
    """Findings list for the UI: the beat-level screen's findings plus the AF screen as a
    finding of its own (so one list tells the whole story)."""
    if arr is None and af is None:
        return None
    findings = []
    af_eps = [(float(a), float(b)) for a, b, *_ in (getattr(af, "episodes", []) or [])] if af is not None else []
    if af is not None and getattr(af, "is_af", False):
        findings.append({"code": "af", "label": getattr(af, "label", "") or "Possible atrial fibrillation", "severity": "urgent",
                         "confidence": getattr(af, "confidence", 0.5), "evidence": getattr(af, "evidence", ""),
                         "count": 0, "beats": [], "times": [], "episodes": af_eps,
                         "when": ("episodes " + ", ".join(f"{a:.1f}–{b:.1f} s" for a, b in af_eps[:6])
                                  + (f" (+{len(af_eps) - 6} more)" if len(af_eps) > 6 else "")) if af_eps else "whole recording"})
    for f in list(getattr(arr, "findings", []) or []):
        findings.append({"code": getattr(f, "code", ""), "label": getattr(f, "label", ""),
                         "severity": getattr(f, "severity", "info"), "confidence": getattr(f, "confidence", 0.0),
                         "evidence": getattr(f, "evidence", ""), "count": int(getattr(f, "count", 0) or 0),
                         "beats": [int(b) for b in (getattr(f, "beats", []) or [])][:200],
                         "times": [float(t) for t in (getattr(f, "times", []) or [])][:200],
                         "episodes": [(float(a), float(b)) for a, b in (getattr(f, "episodes", []) or [])][:200],
                         "when": f.when() if hasattr(f, "when") else ""})
    if af is not None and not getattr(af, "is_af", False) and getattr(af, "label", "") != "Undetermined":
        findings.append({"code": "af_screen", "label": "AF screen: not suggested", "severity": "info",
                         "confidence": getattr(af, "confidence", 0.0), "evidence": getattr(af, "evidence", ""),
                         "count": 0, "beats": []})
    order = {"urgent": 0, "notable": 1, "info": 2}
    findings.sort(key=lambda d: order.get(d["severity"], 3))
    labels = getattr(arr, "beat_labels", None)
    return {"findings": findings, "summary": getattr(arr, "summary", "") if arr is not None else "",
            "metrics": dict(getattr(arr, "metrics", {}) or {}) if arr is not None else {},
            "beat_labels": [str(v) for v in np.asarray(labels).tolist()] if labels is not None else None,
            "warnings": list(getattr(arr, "warnings", []) or []) if arr is not None else []}


def record_to_json(rec) -> dict | None:
    if rec is None:
        return None
    d = rec.to_json() if hasattr(rec, "to_json") else _jsonable(rec)
    d = _jsonable(d)
    d["has_report"] = (Path(rec.dir) / "report.pdf").exists() if getattr(rec, "dir", None) else False
    return d


# ----------------------------------------------------------------------------- server
class HeartServer:
    def __init__(self, sim_default: bool = False, mains: float = 60.0):
        from aiohttp import web
        self.web = web
        self.sim_default = sim_default
        self.mains = mains
        self.ctrl: AppController | None = None
        self._clients: set = set()
        self._live_task: asyncio.Task | None = None
        self._lib_ctrl: AppController | None = None
        # live push state
        self._sent_upto_t = -1.0
        self._last_fid: dict | None = None
        self._last_fid_source = "default"
        self._last_delin = 0.0
        self._delin_busy = False
        self._recording_path: str | None = None
        self._selftest = False
        # live event log (rhythm events seen by the live screen, time-stamped, de-duplicated)
        self._logged_keys: set = set()
        self._live_events: list = []
        self._events_path = data_dir() / "live_events.jsonl"

        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        r = self.app.router
        r.add_get("/", self.index)
        r.add_get("/ws", self.ws_handler)
        r.add_get("/api/state", self.api_state)
        r.add_get("/api/live-events", self.api_live_events)
        r.add_get("/api/readings", self.api_readings)
        r.add_get("/api/readings/{rid}/playback", self.api_playback)
        r.add_get("/api/readings/{rid}/analysis", self.api_analysis)
        r.add_post("/api/readings/{rid}/report", self.api_report)
        r.add_get("/api/readings/{rid}/file/{name}", self.api_file)
        r.add_post("/api/readings/{rid}/open", self.api_open)
        r.add_post("/api/readings/{rid}/delete", self.api_delete)
        r.add_post("/api/import", self.api_import)
        r.add_post("/api/import-sample", self.api_import_sample)
        r.add_static("/static/", WEB_DIR, show_index=False)
        self.app.on_shutdown.append(self._on_shutdown)

    # ------------------------------------------------------------------ helpers
    def _library_ctrl(self) -> AppController:
        if self.ctrl is not None:
            return self.ctrl
        if self._lib_ctrl is None:
            self._lib_ctrl = AppController(sim=self.sim_default, mains=self.mains)
        return self._lib_ctrl

    async def _broadcast(self, msg: dict):
        if not self._clients:
            return
        data = json.dumps(_jsonable(msg))
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_str(data)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def _state_msg(self) -> dict:
        st = self.ctrl.state if self.ctrl else "idle"
        return {"type": "status", "payload": st, "events": _jsonable(self._live_events[-200:]),
                "sim": bool(self.ctrl.sim) if self.ctrl else self.sim_default,
                "single_lead": bool(self.ctrl.single_lead) if self.ctrl else False,
                "recording_path": self._recording_path,
                "gate_state": self.ctrl.gate_state if self.ctrl else "idle",
                "device": (self.ctrl.stats().device_name if self.ctrl else None)}

    # ------------------------------------------------------------------ pages / static
    async def index(self, request):
        return self.web.FileResponse(WEB_DIR / "index.html",
                                     headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------ websocket
    async def ws_handler(self, request):
        ws = self.web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)
        try:
            await ws.send_str(json.dumps(_jsonable(self._state_msg())))
            async for msg in ws:
                if msg.type == self.web.WSMsgType.TEXT:
                    try:
                        cmd = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    try:
                        await self._handle_cmd(cmd)
                    except Exception as e:  # noqa: BLE001
                        await self._broadcast({"type": "error", "payload": str(e)})
                elif msg.type == self.web.WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _handle_cmd(self, cmd: dict):
        kind = cmd.get("cmd")
        if kind == "connect":
            await self._connect(bool(cmd.get("sim", self.sim_default)),
                                bool(cmd.get("single_lead", False)))
        elif kind == "disconnect":
            await self._disconnect()
        elif kind == "gate_start":
            if self.ctrl:
                secs = float(cmd.get("seconds", 30))
                thr = max(0.5, min(0.99, float(cmd.get("threshold", 0.9))))
                self.ctrl.start_gated_recording(seconds=secs, sqi_threshold=thr, hold_s=4.0)
        elif kind == "gate_cancel":
            if self.ctrl:
                self.ctrl.cancel_gated_recording()
        elif kind == "rec_start":
            if self.ctrl and not self._recording_path:
                root = data_dir() / "recordings"
                root.mkdir(parents=True, exist_ok=True)
                p = root / time.strftime("open_kardia-%Y%m%d-%H%M%S")
                written = self.ctrl.start_saving(str(p), ["csv", "wfdb"])
                self._recording_path = written[0] if written else str(p)
                await self._broadcast({"type": "recording", "payload": self._recording_path})
        elif kind == "rec_stop":
            if self.ctrl:
                self.ctrl.stop_saving()
            self._recording_path = None
            await self._broadcast({"type": "recording", "payload": None})
        elif kind == "snapshot":
            if self.ctrl:
                root = data_dir() / "recordings"
                root.mkdir(parents=True, exist_ok=True)
                p = root / time.strftime("open_kardia-snap-%Y%m%d-%H%M%S")
                written = self.ctrl.save_buffer(str(p), 30, ["csv", "wfdb"])
                await self._broadcast({"type": "toast",
                                       "payload": f"Snapshot saved: {written[0]}.csv" if written
                                       else "Nothing to save yet."})
        elif kind == "ping":
            pass

    # ------------------------------------------------------------------ connect / disconnect
    async def _connect(self, sim: bool, single_lead: bool):
        if self.ctrl and self.ctrl.state in ("connecting", "streaming"):
            return
        self.ctrl = AppController(sim=sim, mains=self.mains, derive=True, single_lead=single_lead)
        self.ctrl.attach_loop(asyncio.get_running_loop())
        self._sent_upto_t = -1.0
        self._last_fid = None
        self._last_fid_source = "default"
        self._last_delin = 0.0
        self._recording_path = None
        self._logged_keys = set()
        self._live_events = []
        self._session_id = time.strftime("%Y%m%d-%H%M%S")
        await self._broadcast({"type": "reset_live"})
        self.ctrl.start()
        await self._broadcast(self._state_msg())
        if self._live_task is None or self._live_task.done():
            self._live_task = asyncio.create_task(self._live_loop())

    async def _disconnect(self):
        if not self.ctrl:
            return
        try:
            self.ctrl.cancel_gated_recording()
        except Exception:  # noqa: BLE001
            pass
        if self._recording_path:
            try:
                self.ctrl.stop_saving()
            except Exception:  # noqa: BLE001
                pass
            self._recording_path = None
        self.ctrl.stop()   # external loop: teardown is scheduled; 'stopped' event follows

    # ------------------------------------------------------------------ live push loop
    async def _live_loop(self):
        tick = 0
        loop = asyncio.get_running_loop()
        try:
            while self.ctrl is not None:
                tick += 1
                # 1) forward controller events
                for kind, payload in self.ctrl.poll_events():
                    await self._forward_event(kind, payload)
                if self.ctrl.state == "streaming":
                    try:
                        await self._push_samples()
                    except Exception as e:  # noqa: BLE001
                        await self._broadcast({"type": "toast", "payload": f"live push error: {e}"})
                    if tick % BEATS_EVERY_TICKS == 0:
                        await self._push_beats(loop)
                    if tick % STATS_EVERY_TICKS == 0:
                        await self._push_stats()
                elif self.ctrl.state in ("stopped", "error", "idle"):
                    # drain remaining events once, then park the loop
                    if tick % STATS_EVERY_TICKS == 0:
                        await self._push_stats()
                    if self.ctrl.state in ("stopped", "error") and not self._recording_path:
                        for kind, payload in self.ctrl.poll_events():
                            await self._forward_event(kind, payload)
                        break
                await asyncio.sleep(LIVE_TICK_S)
        finally:
            self._live_task = None

    async def _forward_event(self, kind, payload):
        if kind == "reading_complete":
            rec = (payload or {}).get("record")
            msg = {"type": "reading_complete", "payload": {
                "record": record_to_json(rec),
                "analysis": analysis_to_json((payload or {}).get("analysis")),
                "reports": _jsonable((payload or {}).get("reports") or {}),
                "summary": (payload or {}).get("summary", ""),
            }}
            await self._broadcast(msg)
            if self._selftest:
                rid = getattr(rec, "id", "?")
                print(f"SELFTEST: reading_complete {rid}", flush=True)
                asyncio.get_running_loop().call_later(0.5, self._selftest_finish)
            return
        if kind == "connected":
            await self._broadcast({"type": "connected", "payload": list(payload) if payload else []})
            await self._broadcast(self._state_msg())
            if self._selftest:
                asyncio.get_running_loop().call_later(
                    1.5, lambda: self.ctrl and self.ctrl.start_gated_recording(
                        seconds=6, sqi_threshold=0.6, hold_s=0.6))
            return
        if kind == "status":
            await self._broadcast(self._state_msg())
            return
        await self._broadcast({"type": kind, "payload": _jsonable(payload)})

    def _selftest_finish(self):
        ev = getattr(self, "stop_event", None)
        if ev is not None:
            ev.set()

    async def _push_samples(self):
        ctrl = self.ctrl
        t, x, names = ctrl.latest(LIVE_WINDOW_S)
        fs = ctrl.fs
        n = len(x)
        if n < int(1.5 * fs):
            return
        xf = dsp.display_filter(x, fs, self.mains)
        lag = int(LIVE_LAG_S * fs)
        stable_end = n - lag
        if stable_end <= 0:
            return
        # new, settled samples: t > sent_upto
        start = int(np.searchsorted(t, self._sent_upto_t, side="right"))
        if start >= stable_end:
            return
        # first push: don't dump the whole history, start ~1 s back
        if self._sent_upto_t < 0:
            start = max(start, stable_end - int(1.0 * fs))
        seg_t = t[start:stable_end]
        seg = xf[start:stable_end]
        self._sent_upto_t = float(seg_t[-1])
        await self._broadcast({
            "type": "chunk", "t0": float(seg_t[0]), "fs": fs, "leads": list(names),
            "data": [np.round(seg[:, j], 4).tolist() for j in range(seg.shape[1])],
        })

    async def _push_beats(self, loop):
        ctrl = self.ctrl
        t, x, names = ctrl.latest(10.0)
        fs = ctrl.fs
        if len(x) < int(2.5 * fs):
            return
        li = beats_mod.primary_lead_index(list(names))
        col = x[:, li]
        # fast path: R-peaks only, reuse the last good delineation
        timing = beats_mod.beat_timing(col, fs, t0=float(t[0]), mains=self.mains,
                                       delineate=False, prev_fid=self._last_fid)
        if self._last_fid is not None:
            timing["fid_source"] = self._last_fid_source
        await self._broadcast({"type": "beats", **timing})
        # slow path: refresh the delineated fiducials in a worker thread now and then
        now = time.monotonic()
        if not self._delin_busy and now - self._last_delin > DELINEATE_EVERY_S and len(x) >= int(6 * fs):
            self._delin_busy = True
            self._last_delin = now
            col_copy = col.copy()

            t30, x30, _n30 = ctrl.latest(30.0)
            col30 = x30[:, li].copy() if len(x30) else col_copy
            t30_0 = float(t30[0]) if len(t30) else float(t[0])

            def work():
                timing = beats_mod.beat_timing(col_copy, fs, t0=float(t[0]), mains=self.mains,
                                               delineate=True, prev_fid=None)
                timing["screen"] = beats_mod.live_screen(col30, fs, mains=self.mains, t0=t30_0)
                return timing

            def done(fut):
                self._delin_busy = False
                try:
                    res = fut.result()
                except Exception:  # noqa: BLE001
                    return
                scr = res.get("screen")
                if scr is not None:
                    asyncio.ensure_future(self._broadcast({"type": "screen", **scr}))
                    try:
                        new = self._log_live_events(scr, latest_t=float(t30[-1]) if len(t30) else float(t[-1]))
                        if new:
                            asyncio.ensure_future(self._broadcast({"type": "events", "events": new}))
                    except Exception:  # noqa: BLE001
                        pass
                if res.get("fid_source") == "delineated":
                    self._last_fid = res["fid"]
                    self._last_fid_source = "delineated"
                elif self._last_fid is None:
                    self._last_fid = res["fid"]
                    self._last_fid_source = "default"

            fut = loop.run_in_executor(None, work)
            fut.add_done_callback(done)

    _LOGGED_CODES = ("af", "vt", "vt_run", "svt", "svt_run", "pause", "couplet", "bigeminy", "trigeminy", "pvc", "pac")

    def _log_live_events(self, screen: dict, latest_t: float) -> list:
        """Turn the live screen's findings into time-stamped events (one per beat / episode),
        skip the ones already logged (overlapping 30-s windows), append to the JSONL log and
        return the new ones. Only rhythm *events* are logged — interval/rate flags are not."""
        now = time.time()
        new = []
        for f in screen.get("findings", []) or []:
            code = f.get("code")
            if code not in self._LOGGED_CODES:
                continue
            eps = f.get("episodes") or [(t, t) for t in (f.get("times") or [])]
            for a, b in eps:
                key = (code, round(float(a), 1))
                if key in self._logged_keys:
                    continue
                # a beat/episode still inside the fresh 0.5 s at the window end may be re-labelled
                # on the next pass — wait for it to settle before logging
                if float(b) > latest_t - 0.6:
                    continue
                self._logged_keys.add(key)
                wall = now - (latest_t - float(a))
                ev = {"session": getattr(self, "_session_id", ""), "wall": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall)),
                      "t": round(float(a), 2), "t_end": round(float(b), 2), "code": code,
                      "label": f.get("label", code), "severity": f.get("severity", "info"),
                      "confidence": round(float(f.get("confidence", 0.0) or 0.0), 2),
                      "evidence": f.get("evidence", ""), "device": self.ctrl.stats().device_name if self.ctrl else None}
                new.append(ev)
        if new:
            self._live_events.extend(new)
            try:
                self._events_path.parent.mkdir(parents=True, exist_ok=True)
                with self._events_path.open("a") as fh:
                    for ev in new:
                        fh.write(json.dumps(_jsonable(ev)) + "\n")
            except OSError:
                pass
        return new

    async def api_live_events(self, request):
        return self.web.json_response({"events": _jsonable(self._live_events), "path": str(self._events_path)})

    async def _push_stats(self):
        s = self.ctrl.stats()
        await self._broadcast({"type": "stats", "payload": {
            "hr": s.hr_bpm if s.hr_bpm == s.hr_bpm else None,
            "sqi": s.sqi if s.sqi == s.sqi else None,
            "readable": bool(s.readable), "elapsed_s": s.elapsed_s, "samples": s.samples,
            "rate_hz": s.rate_hz, "gaps": s.seq_gaps, "dropped": s.dropped,
            "connected": bool(s.connected), "state": s.state, "device": s.device_name,
        }})

    # ------------------------------------------------------------------ REST: state / library
    async def api_state(self, request):
        return self.web.json_response(_jsonable(self._state_msg()))

    async def api_readings(self, request):
        recs = self._library_ctrl().list_readings()
        return self.web.json_response([record_to_json(r) for r in recs])

    def _record(self, rid):
        return self._library_ctrl().library.get(rid)

    async def api_playback(self, request):
        rid = request.match_info["rid"]
        rec = self._record(rid)
        if rec is None:
            raise self.web.HTTPNotFound(text="no such reading")
        loop = asyncio.get_running_loop()
        cache = Path(rec.dir) / "heart.json"
        csv = Path(rec.csv_path())
        try:
            if cache.exists() and cache.stat().st_mtime >= csv.stat().st_mtime:
                return self.web.Response(text=cache.read_text(), content_type="application/json")
        except OSError:
            pass

        def work():
            from .sinks.csv_sink import read_csv
            fs, names, _t, x = read_csv(str(csv))
            b = beats_mod.playback_bundle(x, fs, names, mains=self.mains)
            b["record"] = record_to_json(rec)
            text = json.dumps(_jsonable(b))
            try:
                cache.write_text(text)
            except OSError:
                pass
            return text

        text = await loop.run_in_executor(None, work)
        return self.web.Response(text=text, content_type="application/json")

    async def api_analysis(self, request):
        rid = request.match_info["rid"]
        ctrl = self._library_ctrl()
        loop = asyncio.get_running_loop()
        try:
            res = await loop.run_in_executor(None, ctrl.analyze_reading, rid)
        except Exception as e:  # noqa: BLE001
            return self.web.json_response({"error": str(e)}, status=500)
        rec = self._record(rid)
        return self.web.json_response({"record": record_to_json(rec),
                                       "analysis": analysis_to_json(res)})

    async def api_report(self, request):
        rid = request.match_info["rid"]
        fmt = request.query.get("fmt", "pdf")
        if fmt not in ("pdf", "png"):
            fmt = "pdf"
        rec = self._record(rid)
        if rec is None:
            raise self.web.HTTPNotFound(text="no such reading")
        ctrl = self._library_ctrl()
        loop = asyncio.get_running_loop()
        out = str(Path(rec.dir) / f"report.{fmt}")
        try:
            path = await loop.run_in_executor(None, ctrl.export_report, rid, out, fmt)
        except Exception as e:  # noqa: BLE001
            return self.web.json_response({"error": str(e)}, status=500)
        return self.web.json_response({"path": path, "url": f"/api/readings/{rid}/file/report.{fmt}"})

    async def api_file(self, request):
        rid = request.match_info["rid"]
        name = request.match_info["name"]
        rec = self._record(rid)
        if rec is None or "/" in name or name.startswith("."):
            raise self.web.HTTPNotFound()
        p = Path(rec.dir) / name
        if not p.exists():
            raise self.web.HTTPNotFound()
        return self.web.FileResponse(p)

    async def api_open(self, request):
        rid = request.match_info["rid"]
        rec = self._record(rid)
        if rec is None:
            raise self.web.HTTPNotFound()
        body = await request.json() if request.can_read_body else {}
        target = Path(rec.dir)
        name = (body or {}).get("name")
        if name and "/" not in name:
            target = target / name
        if sys.platform == "darwin":
            args = ["open", "-R", str(target)] if (body or {}).get("reveal") else ["open", str(target)]
            subprocess.run(args, check=False)
        return self.web.json_response({"ok": True, "path": str(target)})

    async def api_delete(self, request):
        rid = request.match_info["rid"]
        ok = self._library_ctrl().delete_reading(rid)
        return self.web.json_response({"ok": bool(ok)})

    async def api_import(self, request):
        """Import a CSV (open_kardia format, or 'time,lead,...' with an fs= header) into the library."""
        reader = await request.multipart()
        field = await reader.next()
        if field is None:
            return self.web.json_response({"error": "no file"}, status=400)
        data = await field.read(decode=False)
        fname = field.filename or "import.csv"
        loop = asyncio.get_running_loop()
        try:
            rec = await loop.run_in_executor(None, self._import_bytes, data, fname)
        except Exception as e:  # noqa: BLE001
            return self.web.json_response({"error": str(e)}, status=400)
        return self.web.json_response({"ok": True, "record": record_to_json(rec)})

    def _import_bytes(self, data: bytes, fname: str):
        import tempfile
        from .sinks.csv_sink import read_csv
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            fs, names, _t, x = read_csv(tmp)
        finally:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
        stem = Path(fname).stem
        return self._library_ctrl().library.add(
            x, fs, names, meta={"source": "import", "file": fname, "mains": self.mains},
            name=f"Imported {stem}")

    async def api_import_sample(self, request):
        """Add the repo's bundled real capture (captures/rec_cli.csv) to the library."""
        sample = Path(__file__).resolve().parent.parent / "captures" / "rec_cli.csv"
        if not sample.exists():
            return self.web.json_response({"error": "sample capture not found"}, status=404)
        loop = asyncio.get_running_loop()
        try:
            rec = await loop.run_in_executor(None, self._import_bytes, sample.read_bytes(),
                                             "sample-6-lead-capture.csv")
        except Exception as e:  # noqa: BLE001
            return self.web.json_response({"error": str(e)}, status=400)
        return self.web.json_response({"ok": True, "record": record_to_json(rec)})

    # ------------------------------------------------------------------ shutdown
    async def _on_shutdown(self, app):
        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self.ctrl is not None:
            try:
                await asyncio.wait_for(self.ctrl.aclose(), 6)
            except Exception:  # noqa: BLE001
                pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    print_disclaimer()
    sim = "--sim" in argv
    port = DEFAULT_PORT
    if "--port" in argv:
        try:
            port = int(argv[argv.index("--port") + 1])
        except (IndexError, ValueError):
            pass
    mains = 60.0
    if "--mains" in argv:
        try:
            mains = float(argv[argv.index("--mains") + 1])
        except (IndexError, ValueError):
            pass
    open_browser = "--no-browser" not in argv
    selftest = "--selftest" in argv
    seconds = None
    if "--seconds" in argv:
        try:
            seconds = float(argv[argv.index("--seconds") + 1])
        except (IndexError, ValueError):
            pass

    from aiohttp import web
    srv = HeartServer(sim_default=sim, mains=mains)
    srv._selftest = selftest
    url = f"http://127.0.0.1:{port}/"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()
    srv.stop_event = stop_event

    async def serve():
        runner = web.AppRunner(srv.app, shutdown_timeout=3.0)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        print(f"[open-kardia] heart app: {url}   (Ctrl-C to quit)", flush=True)
        if open_browser:
            loop.call_later(0.4, lambda: webbrowser.open(url))
        if selftest:
            # headless smoke test: connect the simulator, gated-record 6 s, exit
            loop.call_later(0.5, lambda: asyncio.ensure_future(srv._connect(True, False)))
        if seconds:
            loop.call_later(seconds, stop_event.set)
        try:
            await stop_event.wait()
        finally:
            await runner.cleanup()

    import signal
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        loop.run_until_complete(serve())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:  # noqa: BLE001
            pass
        loop.close()


if __name__ == "__main__":
    main()
