"""HTTP/WebSocket surface of the heart app (open_kardia.webapp), against a temp library."""
import asyncio
import json

import numpy as np
import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from open_kardia import dsp, webapp  # noqa: E402
from open_kardia.appcore import AppController  # noqa: E402
from open_kardia.library import ReadingLibrary  # noqa: E402

FS = 300.0


def _server(tmp_path):
    srv = webapp.HeartServer(sim_default=True)
    ctrl = AppController(sim=True)
    ctrl._library = ReadingLibrary(tmp_path / "readings")   # keep the user's library untouched
    srv._lib_ctrl = ctrl
    return srv


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_static_and_state(tmp_path):
    async def go():
        srv = _server(tmp_path)
        async with TestClient(TestServer(srv.app)) as c:
            r = await c.get("/")
            assert r.status == 200 and "importmap" in await r.text()
            r = await c.get("/static/app.js")
            assert r.status == 200
            r = await c.get("/static/vendor/three.module.min.js")
            assert r.status == 200
            r = await c.get("/api/state")
            assert r.status == 200
            st = await r.json()
            assert st["type"] == "status" and st["payload"] == "idle" and st["sim"] is True
            r = await c.get("/api/readings")
            assert await r.json() == []
    _run(go())


def test_playback_analysis_report_delete_roundtrip(tmp_path):
    async def go():
        srv = _server(tmp_path)
        lib = srv._library_ctrl().library
        x1 = dsp.synthetic_ecg(12.0, FS, hr_bpm=66.0, noise_uv=10.0, seed=3)
        rec = lib.add(np.column_stack([x1, 1.2 * x1]), FS, ["I", "II"], meta={"source": "test"})
        async with TestClient(TestServer(srv.app)) as c:
            r = await c.get("/api/readings")
            recs = await r.json()
            assert len(recs) == 1 and recs[0]["id"] == rec.id and recs[0]["has_report"] is False

            r = await c.get(f"/api/readings/{rec.id}/playback")
            assert r.status == 200
            b = await r.json()
            assert b["leads"] == ["I", "II"] and len(b["data"][0]) == len(x1)
            assert b["record"]["id"] == rec.id and len(b["beats"]) >= 10
            assert b["fid"]["p_on"] < b["fid"]["q_on"] < 0 < b["fid"]["t_off"]
            # second call is served from the cache file
            assert (tmp_path / "readings" / rec.id / "heart.json").exists()
            r2 = await c.get(f"/api/readings/{rec.id}/playback")
            assert (await r2.json())["beats"] == b["beats"]

            r = await c.get(f"/api/readings/{rec.id}/analysis")
            assert r.status == 200
            a = await r.json()
            assert a["analysis"]["rhythm"]["label"] and isinstance(a["analysis"]["rows"], list)
            json.dumps(a)                                     # JSON-safe (no NaN leaked)

            r = await c.post(f"/api/readings/{rec.id}/report?fmt=png")
            assert r.status == 200
            rep = await r.json()
            assert rep["url"].endswith("report.png")
            r = await c.get(rep["url"])
            assert r.status == 200 and r.headers["Content-Type"].startswith("image/png")
            r = await c.get(f"/api/readings/{rec.id}/file/../secret")
            assert r.status in (400, 404)

            r = await c.get("/api/readings/nope/playback")
            assert r.status == 404

            r = await c.post(f"/api/readings/{rec.id}/delete")
            assert (await r.json())["ok"] is True
            r = await c.get("/api/readings")
            assert await r.json() == []
    _run(go())


def test_import_csv(tmp_path):
    async def go():
        srv = _server(tmp_path)
        # build a open_kardia CSV via the sink
        from open_kardia.sinks.csv_sink import CSVSink
        x = dsp.synthetic_ecg(6.0, FS, hr_bpm=75.0)
        p = tmp_path / "in.csv"
        s = CSVSink(p); s.open(FS, ["I"]); s.write(np.arange(len(x)) / FS, x[:, None]); s.close()
        async with TestClient(TestServer(srv.app)) as c:
            data = aiohttp.FormData()
            data.add_field("file", p.read_bytes(), filename="in.csv", content_type="text/csv")
            r = await c.post("/api/import", data=data)
            assert r.status == 200
            j = await r.json()
            assert j["ok"] and j["record"]["lead_names"] == ["I"] and abs(j["record"]["duration_s"] - 6.0) < 1e-6
            r = await c.get("/api/readings")
            assert len(await r.json()) == 1
    _run(go())


def test_websocket_status_and_sim_stream(tmp_path):
    """Connect the simulator over the WebSocket and check that samples + beats + stats flow
    with the shape the browser expects, then disconnect cleanly."""
    async def go():
        srv = _server(tmp_path)
        async with TestClient(TestServer(srv.app)) as c:
            ws = await c.ws_connect("/ws")
            first = json.loads((await ws.receive_str()))
            assert first["type"] == "status" and first["payload"] == "idle"
            await ws.send_json({"cmd": "connect", "sim": True, "single_lead": False})
            seen = {}
            deadline = asyncio.get_event_loop().time() + 20
            while asyncio.get_event_loop().time() < deadline:
                msg = await asyncio.wait_for(ws.receive(), 5)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                m = json.loads(msg.data)
                seen.setdefault(m["type"], m)
                if m["type"] == "beats" and m.get("beats") and "chunk" in seen and "stats" in seen:
                    break
            assert "connected" in seen and "chunk" in seen and "beats" in seen and "stats" in seen
            ch = seen["chunk"]
            assert ch["fs"] == FS and ch["leads"] == ["I", "II", "III", "aVR", "aVL", "aVF"]
            assert len(ch["data"]) == 6 and len(ch["data"][0]) > 0
            assert seen["beats"]["fid"]["p_on"] < 0 < seen["beats"]["fid"]["t_off"]
            assert seen["stats"]["payload"]["connected"] is True
            await ws.send_json({"cmd": "disconnect"})
            deadline = asyncio.get_event_loop().time() + 10
            stopped = False
            while asyncio.get_event_loop().time() < deadline:
                msg = await asyncio.wait_for(ws.receive(), 5)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                m = json.loads(msg.data)
                if m["type"] == "status" and m["payload"] == "stopped":
                    stopped = True
                    break
            assert stopped
            await ws.close()
    _run(go())


def test_live_event_log_dedupes_and_persists(tmp_path):
    srv = _server(tmp_path)
    srv._events_path = tmp_path / "live_events.jsonl"
    srv._session_id = "test"
    screen = {"findings": [
        {"code": "pvc", "label": "PVCs", "severity": "notable", "confidence": 0.8, "evidence": "e",
         "times": [12.3, 20.1, 39.8], "episodes": [(12.3, 12.3), (20.1, 20.1), (39.8, 39.8)]},
        {"code": "trigeminy", "label": "Trigeminy", "severity": "notable", "confidence": 0.5, "evidence": "e",
         "times": [], "episodes": [(10.0, 25.0)]},
        {"code": "long_qtc", "label": "Long QTc", "severity": "notable", "confidence": 0.5, "evidence": "e",
         "times": [], "episodes": []},                       # interval flag: never logged
    ]}
    new = srv._log_live_events(screen, latest_t=40.0)
    codes = sorted(e["code"] for e in new)
    # the PVC at 39.8 s is still inside the fresh 0.6 s at the window end -> deferred
    assert codes == ["pvc", "pvc", "trigeminy"]
    assert all("wall" in e and e["t"] >= 0 for e in new)
    # same window again -> nothing new; window advanced -> the deferred beat is logged once
    assert srv._log_live_events(screen, latest_t=40.0) == []
    later = srv._log_live_events(screen, latest_t=45.0)
    assert [e["t"] for e in later] == [39.8]
    lines = srv._events_path.read_text().strip().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["code"] in ("pvc", "trigeminy")
    assert len(srv._live_events) == 4
