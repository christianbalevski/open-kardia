// app.js — kardia6l heart app front-end. Left: recording panel; right: the beating heart.
import { SampleStore, ECGScope } from './ecg.js';
import { CardiacCycle, PHASES } from './cycle.js';
import { HeartView } from './heart.js';

const $ = id => document.getElementById(id);
const LIVE_JITTER_S = 0.35;      // client-side buffer on top of the server's LIVE_LAG_S
const LIVE_KEEP_S = 60;

// ---------------------------------------------------------------- state
const S = {
  ws: null, wsOpen: false,
  ctrlState: 'idle', connected: false, sim: false, single: false, gating: false, recording: null,
  mode: 'idle',                        // 'idle' | 'live' | 'playback'
  live: { store: null, cycle: new CardiacCycle(), t: NaN, fs: 300, leads: [] },
  pb: { bundle: null, store: null, cycle: new CardiacCycle(), pos: 0, playing: false, rate: 1, loop: true, record: null },
  view: ['I', 'II'],
  readings: [], selected: null,
  events: [],
  lastFrame: performance.now(),
  currentReadingForModal: null,
};
const scope = new ECGScope($('scope'));
scope.setLeads(S.view);
let heart = null;
try { heart = new HeartView($('heart')); }
catch (e) { console.error(e); toast('WebGL unavailable — heart disabled: ' + e.message); }

// ---------------------------------------------------------------- websocket
function wsConnect() {
  const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
  const ws = new WebSocket(url);
  S.ws = ws;
  ws.onopen = () => { S.wsOpen = true; $('wsState').textContent = 'connected'; $('wsState').classList.add('on'); };
  ws.onclose = () => { S.wsOpen = false; $('wsState').textContent = 'offline'; $('wsState').classList.remove('on'); setTimeout(wsConnect, 1500); };
  ws.onerror = () => {};
  ws.onmessage = ev => { let m; try { m = JSON.parse(ev.data); } catch { return; } handleMsg(m); };
}
function send(obj) { if (S.wsOpen) S.ws.send(JSON.stringify(obj)); else toast('Not connected to the kardia6l server'); }

function handleMsg(m) {
  switch (m.type) {
    case 'status': onStatus(m); break;
    case 'connected': {
      S.connected = true; setStatus('Streaming — ' + (m.payload?.[0] || 'device'), 'on');
      $('gateBtn').disabled = false; $('recBtn').disabled = false; $('snapBtn').disabled = false;
      $('gateStatus').textContent = 'Ready. Hold the electrodes and press record.';
      if (S.mode !== 'playback') enterLive();
      break;
    }
    case 'reset_live': resetLive(); break;
    case 'chunk': onChunk(m); break;
    case 'beats': onBeats(m); break;
    case 'stats': onStats(m.payload); break;
    case 'gate': onGate(m.payload); break;
    case 'rec_start': $('gatePb').style.width = '0%'; $('gatePb').classList.add('rec'); S.recSecs = m.payload?.seconds || 0;
      $('gateStatus').textContent = `Recording ${fmtDur(S.recSecs)}… 0%`; $('gateStatus').style.color = 'var(--green)'; break;
    case 'rec_progress': { const f = Math.max(0, Math.min(1, +m.payload || 0)); $('gatePb').style.width = (f * 100) + '%';
      $('gateStatus').textContent = `Recording ${fmtDur(S.recSecs || 0)}… ${Math.round(f * 100)}%  (${fmtDur(f * (S.recSecs || 0))} done)`; break; }
    case 'reading_complete': onReadingComplete(m.payload); break;
    case 'gate_cancelled': resetGateUI(); $('gateStatus').textContent = 'Cancelled.'; break;
    case 'gate_error': resetGateUI(); $('gateStatus').textContent = String(m.payload); $('gateStatus').style.color = 'var(--orange)'; break;
    case 'error': toast('Error: ' + m.payload, 6000); resetIdle('Error'); break;
    case 'stopped': resetIdle('Stopped'); break;
    case 'stall': $('recStatus').textContent = `⚠ signal dropout (${Number(m.payload).toFixed(0)} ms)`; break;
    case 'recording': S.recording = m.payload; updateRecUI(); break;
    case 'saving': case 'saved': refreshLibrary(); break;
    case 'toast': toast(m.payload); break;
    case 'screen': onScreen(m); break;
    case 'events': addEvents(m.events || []); break;
  }
}

function onStatus(m) {
  S.ctrlState = m.payload; S.sim = !!m.sim; S.single = !!m.single_lead;
  if (Array.isArray(m.events) && !S.events.length) addEvents(m.events);
  if (m.recording_path !== undefined) { S.recording = m.recording_path; updateRecUI(); }
  const label = { connecting: 'Connecting…', streaming: 'Streaming', stopped: 'Stopped', error: 'Error', idle: 'Idle' }[m.payload] || m.payload;
  if (m.payload === 'connecting') { setStatus(label, 'busy'); $('connectBtn').textContent = 'Disconnect'; $('connectBtn').classList.remove('accent'); lockDeviceOpts(true); }
  else if (m.payload === 'streaming') { setStatus(m.device ? `Streaming — ${m.device}` : label, 'on'); $('connectBtn').textContent = 'Disconnect'; $('connectBtn').classList.remove('accent'); lockDeviceOpts(true);
    $('gateBtn').disabled = false; $('recBtn').disabled = false; $('snapBtn').disabled = false; S.connected = true; if (S.mode !== 'playback') enterLive(); }
  else if (m.payload === 'stopped' || m.payload === 'error' || m.payload === 'idle') resetIdle(label);
  $('simChk').checked = S.sim; $('singleChk').checked = S.single;
  applySingleLeadUI();
}
function setStatus(text, dot) { $('statusText').textContent = text; $('statusDot').className = 'dot' + (dot ? ' ' + dot : ''); }
function lockDeviceOpts(lock) { $('simChk').disabled = lock; $('singleChk').disabled = lock; }
function resetIdle(label) {
  S.connected = false; S.gating = false; S.recording = null;
  setStatus(label || 'Idle', ''); $('connectBtn').textContent = 'Connect'; $('connectBtn').classList.add('accent'); lockDeviceOpts(false);
  $('gateBtn').disabled = true; $('recBtn').disabled = true; $('snapBtn').disabled = true; resetGateUI(); updateRecUI();
  $('hrVal').textContent = '--'; $('sqiVal').textContent = '--'; $('sqiVal').className = 'val'; $('metaVal').textContent = '';
  if (S.mode === 'live') { S.mode = 'idle'; updateModeUI(); }
}

// ---------------------------------------------------------------- live data
function resetLive() {
  S.events = []; $('evList').innerHTML = ''; $('evCount').textContent = '';
  S.liveScreen = null; renderScreen(null); scope.setEpisodes([]);
  S.live.store = null; S.live.cycle.clear(); S.live.cycle.fidSource = 'default'; S.live.t = NaN;
  scope.scale = {};
}
function enterLive() {
  S.mode = 'live'; S.pb.playing = false; updateModeUI();
}
function onChunk(m) {
  const L = S.live;
  if (!L.store || L.store.fs !== m.fs || L.store.leads.join() !== m.leads.join()) {
    L.store = new SampleStore(m.fs, m.leads, LIVE_KEEP_S); L.fs = m.fs; L.leads = m.leads; L.t = NaN;
  }
  L.store.append(m.t0, m.data);
  if (S.mode === 'idle' && S.connected) enterLive();
}
function onBeats(m) {
  const C = S.live.cycle;
  const keepFrom = S.live.store ? S.live.store.tStart : undefined;
  C.mergeBeats(m.beats || [], keepFrom, m.labels || null);
  if (m.fid) C.setFid(m.fid, m.fid_source);
  if (S.mode === 'live') showFid(C);
}
function onStats(p) {
  if (!p) return;
  $('hrVal').textContent = (p.hr && p.hr > 20 && p.hr < 300) ? p.hr.toFixed(0) : '--';
  if (p.sqi !== null && p.sqi !== undefined) {
    const pct = Math.round(Math.max(0, Math.min(1, p.sqi)) * 100);
    $('sqiVal').textContent = `${pct}%  ${p.readable ? 'Readable' : 'Noisy / Unreadable'}`;
    $('sqiVal').className = 'val ' + (p.readable ? 'good' : 'bad');
  } else { $('sqiVal').textContent = '--'; $('sqiVal').className = 'val'; }
  const e = Math.max(0, p.elapsed_s || 0);
  $('elapsedVal').textContent = `${String(Math.floor(e / 60)).padStart(2, '0')}:${String(Math.floor(e % 60)).padStart(2, '0')}`;
  $('metaVal').textContent = `${(p.samples || 0).toLocaleString()} samples · ${(p.rate_hz || 0).toFixed(0)} Hz · gaps ${p.gaps} · drop ${p.dropped}`;
  $('statusDot').className = 'dot' + (p.connected ? ' on' : (S.ctrlState === 'connecting' ? ' busy' : ''));
}
const EV_SHORT = { pvc: 'PVC', pac: 'PAC', couplet: 'couplet', bigeminy: 'bigeminy', trigeminy: 'trigeminy', vt_run: 'V run', svt_run: 'S run',
                   pause: 'pause', af: 'AF episode', vt: 'VT?', svt: 'SVT?' };
function addEvents(list) {
  if (!list.length) return;
  S.events.push(...list);
  const ul = $('evList');
  for (const e of list) {
    const li = document.createElement('li'); li.className = e.severity || 'info';
    const span = (e.t_end && e.t_end > e.t + 0.05) ? `${fmtClock(e.t)}–${fmtClock(e.t_end)}` : fmtClock(e.t);
    li.innerHTML = `<span class="wall">${(e.wall || '').slice(11)}</span><span class="st">${span}</span><span class="lab">${EV_SHORT[e.code] || e.label}</span>`;
    li.title = `${e.label}\n${e.evidence || ''}\nstream time ${e.t}s`;
    ul.appendChild(li);
  }
  ul.scrollTop = ul.scrollHeight;
  $('evCount').textContent = `${S.events.length} event${S.events.length === 1 ? '' : 's'}`;
}
$('evClear').onclick = () => { S.events = []; $('evList').innerHTML = ''; $('evCount').textContent = ''; };
function onScreen(m) {
  S.liveScreen = m;
  if (S.mode === 'live') { renderScreen(m); scope.setEpisodes(episodesFromScreen(m)); }
}
const EPISODE_STYLE = { af: ['#ff5c5c', 'AF'], vt: ['#ff5c5c', 'VT?'], vt_run: ['#ff5c5c', 'V run'], svt: ['#ff8a5c', 'SVT?'], svt_run: ['#f6a609', 'S run'],
                        pause: ['#ff5c5c', 'pause'], couplet: ['#ff6a3d', 'couplet'], bigeminy: ['#ff6a3d', 'bigeminy'], trigeminy: ['#ff6a3d', 'trigeminy'],
                        pvc: ['#ff6a3d', ''], pac: ['#f6a609', ''] };
/** Convert screen findings -> shaded episodes for the scope (patterns/episodes only; single beats are already marked V/S). */
function episodesFromScreen(m) {
  const out = [];
  for (const f of (m && m.findings) || []) {
    const st = EPISODE_STYLE[f.code]; if (!st) continue;
    if (f.code === 'pvc' || f.code === 'pac') continue;               // per-beat markers cover these
    for (const [a, b] of (f.episodes || [])) out.push({ t0: a, t1: b, color: st[0], label: st[1] });
  }
  return out;
}
function renderScreen(m) {
  const el = $('screenVal');
  if (!m || (!(m.findings && m.findings.length) && !m.af)) { el.textContent = '--'; el.className = 'val small'; el.title = ''; return; }
  const parts = [];
  const fs = m.findings || [];
  const worst = fs.some(f => f.severity === 'urgent') ? 'urgent' : (fs.some(f => f.severity === 'notable') ? 'notable' : 'info');
  if (m.af && m.af.is_af) parts.push('Possible AF (' + Math.round((m.af.score || 0) * 100) + '%)');
  for (const f of fs.slice(0, 3)) parts.push(f.label.replace(/\s*\(.*?\)\s*$/, ''));
  if (!parts.length && m.af) parts.push(m.af.label);
  el.textContent = parts.join(' · ');
  el.title = [(m.summary || ''), (m.af && m.af.evidence) || ''].filter(Boolean).join('\n');
  el.className = 'val small ' + ((m.af && m.af.is_af) || worst === 'urgent' || worst === 'notable' ? 'bad' : 'good');
}
function showFid(C) {
  const F = C.fid;
  const ms = x => (x * 1000).toFixed(0);
  $('fidVal').textContent = `PR ${ms(F.q_on - F.p_on)} · QRS ${ms(F.q_off - F.q_on)} · QT ${ms(F.t_off - F.q_on)} ms`;
  $('fidSrc').textContent = C.fidSource === 'delineated'
    ? 'Heart timing: R-peaks + P/QRS/T delineated from this reading’s median beat'
    : 'Heart timing: R-peaks + textbook P/QRS/T intervals scaled to heart rate';
}

// ---------------------------------------------------------------- gate / recording UI
function onGate(p) {
  if (!p) return;
  if (p.state === 'analyzing') { $('gatePb').style.width = '100%'; $('gateStatus').textContent = 'Analyzing recording…'; $('gateStatus').style.color = 'var(--accent)'; return; }
  const pct = (p.sqi !== null && p.sqi !== undefined && isFinite(p.sqi)) ? Math.round(p.sqi * 100) : 0;
  const need = Math.round((p.threshold || 0.9) * 100), held = p.held_s || 0, hold = p.hold_s || 1.5;
  $('gatePb').classList.remove('rec'); $('gatePb').style.width = Math.max(0, Math.min(100, held / hold * 100)) + '%';
  if (held > 0) { $('gateStatus').textContent = `Signal quality ${pct}% — hold still, recording starts in ${Math.max(0, hold - held).toFixed(0)} s`; $('gateStatus').style.color = 'var(--green)'; }
  else { $('gateStatus').textContent = `Waiting for signal quality: ${pct}%  (need ≥ ${need}%)`; $('gateStatus').style.color = ''; }
}
function resetGateUI() {
  S.gating = false; $('gateBtn').textContent = '● Record (quality-gated)'; $('gateBtn').classList.remove('rec'); $('gateBtn').classList.add('accent');
  $('thrInp').disabled = false; $('gatePb').style.width = '0%'; $('gatePb').classList.remove('rec'); $('gateStatus').style.color = '';
}
function updateRecUI() {
  if (S.recording) { $('recBtn').textContent = '■ Stop recording'; $('recBtn').classList.add('rec'); $('recStatus').textContent = '● REC  ' + S.recording; }
  else { $('recBtn').textContent = '● Record to file'; $('recBtn').classList.remove('rec'); $('recStatus').textContent = ''; }
}
function onReadingComplete(p) {
  resetGateUI(); $('gateStatus').textContent = 'Reading saved.'; $('gateStatus').style.color = 'var(--green)';
  refreshLibrary().then(() => { if (p?.record?.id) selectReading(p.record.id); });
  showAnalysis(p.record, p.analysis, p.reports || {});
}

// ---------------------------------------------------------------- toolbar handlers
$('connectBtn').onclick = () => {
  if (S.ctrlState === 'connecting' || S.ctrlState === 'streaming') send({ cmd: 'disconnect' });
  else { send({ cmd: 'connect', sim: $('simChk').checked, single_lead: $('singleChk').checked }); }
};
$('singleChk').onchange = () => { S.single = $('singleChk').checked; applySingleLeadUI(); };
function applySingleLeadUI() {
  const radios = document.querySelectorAll('input[name=view]');
  radios.forEach(r => { if (r.value !== 'I') r.disabled = S.single; });
  if (S.single) { document.querySelector('input[name=view][value="I"]').checked = true; S.view = ['I']; scope.setLeads(S.view); }
}
document.querySelectorAll('input[name=view]').forEach(r => r.onchange = () => { S.view = r.value.split(','); scope.setLeads(S.view); scope.scale = {}; });
$('gateBtn').onclick = () => {
  if (S.gating) { send({ cmd: 'gate_cancel' }); $('gateStatus').textContent = 'Cancelling…'; return; }
  const secs = parseFloat($('durSel').value) || 30, thr = Math.max(50, Math.min(99, parseInt($('thrInp').value) || 90)) / 100;
  S.gating = true; $('gateBtn').textContent = 'Cancel'; $('gateBtn').classList.add('rec'); $('gateBtn').classList.remove('accent'); $('thrInp').disabled = true;
  $('gatePb').style.width = '0%'; $('gateStatus').textContent = 'Waiting for signal quality…';
  send({ cmd: 'gate_start', seconds: secs, threshold: thr });
};
$('recBtn').onclick = () => send({ cmd: S.recording ? 'rec_stop' : 'rec_start' });
$('snapBtn').onclick = () => send({ cmd: 'snapshot' });

// ---------------------------------------------------------------- library
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { let t = ''; try { t = (await r.json()).error || ''; } catch {} throw new Error(t || `${r.status} ${r.statusText}`); }
  return r.json();
}
async function refreshLibrary() {
  try { S.readings = await api('/api/readings'); } catch (e) { toast('Could not load readings: ' + e.message); S.readings = []; }
  const tb = $('libTable').querySelector('tbody'); tb.innerHTML = '';
  for (const r of S.readings) {
    const tr = document.createElement('tr'); tr.dataset.id = r.id;
    const when = String(r.created_at || r.id).slice(0, 19).replace('T', ' ');
    const hr = (r.hr_bpm !== null && r.hr_bpm !== undefined) ? Number(r.hr_bpm).toFixed(0) : '—';
    const dur = (r.duration_s !== null && r.duration_s !== undefined) ? Number(r.duration_s).toFixed(0) + ' s' : '—';
    tr.innerHTML = `<td>${when}</td><td class="r">${dur}</td><td class="r">${hr}</td><td>${r.rhythm || '—'}</td>
                    <td><button class="btn tiny accent" data-play="${r.id}">▶</button></td>`;
    tr.onclick = (ev) => { if (ev.target.dataset.play) return; selectReading(r.id); };
    tr.ondblclick = () => replayReading(r.id);
    tr.querySelector('[data-play]').onclick = () => replayReading(r.id);
    if (r.id === S.selected) tr.classList.add('sel');
    tb.appendChild(tr);
  }
  $('libEmpty').classList.toggle('hidden', S.readings.length > 0);
  if (S.selected && !S.readings.find(r => r.id === S.selected)) S.selected = null;
}
function selectReading(id) {
  S.selected = id;
  $('libTable').querySelectorAll('tbody tr').forEach(tr => tr.classList.toggle('sel', tr.dataset.id === id));
}
$('refreshBtn').onclick = refreshLibrary;
$('replayBtn').onclick = () => { if (S.selected) replayReading(S.selected); else toast('Select a reading first'); };
$('analysisBtn').onclick = async () => {
  if (!S.selected) return toast('Select a reading first');
  toast('Analyzing…');
  try { const r = await api(`/api/readings/${S.selected}/analysis`); showAnalysis(r.record, r.analysis, {}); }
  catch (e) { toast('Analysis failed: ' + e.message, 6000); }
};
$('pdfBtn').onclick = () => exportPdf(S.selected);
async function exportPdf(id) {
  if (!id) return toast('Select a reading first');
  toast('Rendering PDF…');
  try { const r = await api(`/api/readings/${id}/report?fmt=pdf`, { method: 'POST' }); window.open(r.url, '_blank'); refreshLibrary(); }
  catch (e) { toast('Export failed: ' + e.message, 6000); }
}
$('folderBtn').onclick = () => { if (S.selected) api(`/api/readings/${S.selected}/open`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(e => toast(e.message)); };
$('deleteBtn').onclick = async () => {
  if (!S.selected) return;
  if (!confirm(`Delete reading ${S.selected}?\nThis cannot be undone.`)) return;
  try { await api(`/api/readings/${S.selected}/delete`, { method: 'POST' }); } catch (e) { toast('Delete failed: ' + e.message); }
  if (S.pb.record?.id === S.selected && S.mode === 'playback') exitPlayback();
  S.selected = null; refreshLibrary();
};
$('importBtn').onclick = () => $('importFile').click();
$('importLink').onclick = (e) => { e.preventDefault(); $('importFile').click(); };
$('importFile').onchange = async () => {
  const f = $('importFile').files[0]; if (!f) return;
  const fd = new FormData(); fd.append('file', f, f.name);
  toast('Importing…');
  try { const r = await api('/api/import', { method: 'POST', body: fd }); await refreshLibrary(); selectReading(r.record.id); toast('Imported ' + f.name); }
  catch (e) { toast('Import failed: ' + e.message, 6000); }
  $('importFile').value = '';
};
$('sampleLink').onclick = async (e) => {
  e.preventDefault(); toast('Loading sample capture…');
  try { const r = await api('/api/import-sample', { method: 'POST' }); await refreshLibrary(); selectReading(r.record.id); replayReading(r.record.id); }
  catch (err) { toast('Sample not available: ' + err.message, 6000); }
};

// ---------------------------------------------------------------- playback
async function replayReading(id) {
  toast('Loading reading…');
  let b;
  try { b = await api(`/api/readings/${id}/playback`); } catch (e) { return toast('Could not load reading: ' + e.message, 6000); }
  const P = S.pb;
  P.bundle = b; P.record = b.record; P.store = new SampleStore(b.fs, b.leads, 1e9); P.store.setAll(b.t0 || 0, b.data);
  P.cycle = new CardiacCycle(); P.cycle.setBeats(b.beats || [], b.labels || null); if (b.fid) P.cycle.setFid(b.fid, b.fid_source);
  P.pos = 0; P.playing = true;
  S.mode = 'playback'; selectReading(id); scope.scale = {};
  $('scrub').max = String(b.duration_s || (b.data[0].length / b.fs));
  updateModeUI(); showFid(P.cycle); renderScreen(b.screen || null); scope.setEpisodes(episodesFromScreen(b.screen)); toast(`Replaying ${b.record?.name || id}`);
}
function exitPlayback() {
  S.pb.playing = false; S.mode = S.connected ? 'live' : 'idle'; updateModeUI();
  $('hrVal').textContent = '--'; $('elapsedVal').textContent = '00:00';
  if (S.mode === 'live') { showFid(S.live.cycle); renderScreen(S.liveScreen || null); scope.setEpisodes(episodesFromScreen(S.liveScreen)); } else { renderScreen(null); scope.setEpisodes([]); }
}
function updateModeUI() {
  const pb = S.mode === 'playback';
  $('modeBadge').textContent = pb ? 'PLAYBACK' : (S.mode === 'live' ? 'LIVE' : 'IDLE');
  $('modeBadge').className = 'pill ' + (pb ? 'play' : (S.mode === 'live' ? 'live' : ''));
  $('transport').classList.toggle('disabled', !pb);
  $('liveBtn').classList.toggle('hidden', !S.connected || !pb);
  $('scopeEmpty').classList.toggle('hidden', S.mode !== 'idle');
  scope.setWindow(pb ? 4.5 : 5.0, pb ? 1.6 : 0.15);
  $('playBtn').textContent = S.pb.playing ? '❚❚' : '▶';
  if (!pb) { $('phaseChip').querySelector('.sw').style.background = ''; }
}
$('playBtn').onclick = () => { if (S.mode !== 'playback') return; S.pb.playing = !S.pb.playing; if (S.pb.playing && S.pb.pos >= dur() - 0.01) S.pb.pos = 0; $('playBtn').textContent = S.pb.playing ? '❚❚' : '▶'; };
$('stepBack').onclick = () => seek(S.pb.pos - 1);
$('stepFwd').onclick = () => seek(S.pb.pos + 1);
$('scrub').oninput = () => { seek(parseFloat($('scrub').value)); };
$('loopChk').onchange = () => S.pb.loop = $('loopChk').checked;
$('liveBtn').onclick = exitPlayback;
function dur() { return S.pb.bundle ? (S.pb.bundle.duration_s || 0) : 0; }
function seek(t) { S.pb.pos = Math.max(0, Math.min(dur(), t)); }
// speed slider: log scale 0.05× .. 4×
function sliderToRate(v) { return 0.05 * Math.pow(80, v / 100); }
function rateToSlider(r) { return 100 * Math.log(r / 0.05) / Math.log(80); }
function setRate(r) {
  r = Math.max(0.05, Math.min(4, r)); S.pb.rate = r;
  $('rate').value = String(Math.round(rateToSlider(r)));
  $('rateLbl').textContent = (r < 0.995 ? r.toFixed(2) : r.toFixed(r >= 2 ? 0 : 2)) + '×';
  document.querySelectorAll('.presets .btn').forEach(b => b.classList.toggle('active', Math.abs(parseFloat(b.dataset.rate) - r) < 1e-3));
}
$('rate').oninput = () => setRate(sliderToRate(parseFloat($('rate').value)));
document.querySelectorAll('.presets .btn').forEach(b => b.onclick = () => setRate(parseFloat(b.dataset.rate)));
setRate(1);
window.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement?.tagName) && e.code !== 'Space') return;
  if (S.mode !== 'playback') return;
  if (e.code === 'Space') { e.preventDefault(); $('playBtn').click(); }
  else if (e.code === 'ArrowLeft') seek(S.pb.pos - (e.shiftKey ? 5 : 1));
  else if (e.code === 'ArrowRight') seek(S.pb.pos + (e.shiftKey ? 5 : 1));
  else if (e.key === '[') setRate(S.pb.rate / 2);
  else if (e.key === ']') setRate(S.pb.rate * 2);
});

// ---------------------------------------------------------------- heart HUD controls
$('xrayChk').onchange = () => heart && heart.setXray($('xrayChk').checked);
$('rotChk').onchange = () => heart && heart.setAutoRotate($('rotChk').checked);
$('resetViewBtn').onclick = () => heart && heart.resetView();

// ---------------------------------------------------------------- analysis modal
function fmt(v, unit = '', digits = 0) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return Number(v).toFixed(digits) + (unit ? ' ' + unit : '');
}
function showAnalysis(record, analysis, reports) {
  S.currentReadingForModal = record;
  const rid = record?.id || 'reading';
  $('mTitle').textContent = `Reading ${rid}` + (record?.name ? ` — ${record.name}` : '');
  const rh = analysis?.rhythm;
  $('mRhythm').textContent = rh?.label || 'No analysis';
  $('mRhythm').className = 'rhythm ' + (rh && /normal/i.test(rh.label) ? 'good' : (rh ? 'warn' : ''));
  $('mConf').textContent = rh ? `confidence ${Math.round((rh.confidence || 0) * 100)}%` : '';
  $('mRationale').textContent = rh?.rationale || '';
  const rows = [];
  for (const r of (analysis?.rows || [])) {
    let val = fmt(r.value, r.unit, r.digits || 0);
    if (r.key === 'qrs_axis_deg' && analysis.axis_label && r.value !== null) val += `  (${analysis.axis_label})`;
    rows.push([r.label, val, r.range ? 'normal ' + r.range : '', r.status || '']);
  }
  const c = analysis?.context || {};
  rows.push(['RR mean ± SD', `${fmt(c.rr_mean_ms)} ± ${fmt(c.rr_sd_ms)} ms`, '', ''],
            ['HR min–max', `${fmt(c.hr_min)}–${fmt(c.hr_max)} bpm`, '', ''],
            ['SDNN', fmt(c.SDNN, 'ms'), '', ''], ['RMSSD', fmt(c.RMSSD, 'ms'), '', ''], ['pNN50', fmt(c.pNN50, '%', 1), '', ''],
            ['Beats analysed', fmt(c.n_beats), '', ''], ['Analysis lead', c.lead_used || '—', '', '']);
  $('mRows').innerHTML = rows.map(([k, v, rng, st]) =>
    `<tr><td class="k">${k}</td><td class="v ${st}">${v}${st === 'low' || st === 'high' ? '  ' + st.toUpperCase() : ''}</td><td class="rng">${rng}</td></tr>`).join('');
  // rhythm findings (arrhythmia screen)
  const findings = analysis?.arrhythmia?.findings || [];
  $('mFindingsWrap').classList.toggle('hidden', findings.length === 0);
  $('mFindings').innerHTML = findings.map((f, i) => {
    const seekable = (f.times && f.times.length) || (f.episodes && f.episodes.length);
    return `<li class="${seekable ? 'seekable' : ''}" data-i="${i}" title="${seekable ? 'Click to replay from here' : ''}">` +
      `<span class="sev ${f.severity}">${f.severity}</span><span class="lbl">${f.label}${f.confidence ? ` <span class="muted tiny">(${Math.round(f.confidence * 100)}%)</span>` : ''}</span>` +
      `<span class="ev">${f.evidence || ''}</span>` +
      `<span class="when">${f.when || 'whole recording'}${seekable ? '  ▶' : ''}</span></li>`; }).join('');
  $('mFindings').querySelectorAll('li.seekable').forEach(li => li.onclick = async () => {
    const f = findings[parseInt(li.dataset.i)];
    const t = (f.episodes && f.episodes.length) ? f.episodes[0][0] : f.times[0];
    closeModal();
    if (!(S.mode === 'playback' && S.pb.record && S.pb.record.id === rid)) await replayReading(rid);
    seek(Math.max(0, t - 1.5)); S.pb.playing = true; $('playBtn').textContent = '❚❚';
  });
  const notes = [...(analysis?.warnings || []), ...(rh?.flags || [])];
  $('mWarn').textContent = notes.length ? '⚠ ' + notes.join('; ') : '';
  const pdf = reports?.pdf || (record?.has_report ? 'report.pdf' : null);
  $('mPdf').textContent = pdf ? 'Open report (PDF)' : 'Export PDF';
  $('mPdf').onclick = () => { if (pdf) window.open(`/api/readings/${rid}/file/report.pdf`, '_blank'); else exportPdf(rid); };
  $('mReveal').onclick = () => api(`/api/readings/${rid}/open`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reveal: !!pdf, name: pdf ? 'report.pdf' : null }) }).catch(e => toast(e.message));
  $('mReplay').onclick = () => { closeModal(); replayReading(rid); };
  $('modal').classList.remove('hidden');
}
function closeModal() { $('modal').classList.add('hidden'); }
$('mClose').onclick = closeModal;
$('modal').onclick = (e) => { if (e.target === $('modal')) closeModal(); };

// ---------------------------------------------------------------- toast
let toastTimer = null;
function toast(msg, ms = 2600) {
  const t = $('toast'); t.textContent = msg; t.classList.remove('hidden');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add('hidden'), ms);
}

// ---------------------------------------------------------------- render loop
function frame(now) {
  const dt = Math.min(0.1, (now - S.lastFrame) / 1000); S.lastFrame = now;
  let store = null, cycle = null, tNow = NaN, primary = 'II';
  if (S.mode === 'playback' && S.pb.store) {
    const P = S.pb;
    if (P.playing) {
      P.pos += dt * P.rate;
      if (P.pos >= dur()) { if (P.loop) P.pos = 0; else { P.pos = dur(); P.playing = false; $('playBtn').textContent = '▶'; } }
    }
    store = P.store; cycle = P.cycle; tNow = (P.bundle.t0 || 0) + P.pos; primary = P.bundle.primary || 'II';
    $('scrub').value = String(P.pos);
    $('timeLbl').textContent = `${fmtClock(P.pos)} / ${fmtClock(dur())}`;
    const hrNow = cycle.hrAt ? cycle.hrAt(tNow) : cycle.hr;
    $('hrVal').textContent = (isFinite(hrNow) && hrNow > 20 && hrNow < 300) ? hrNow.toFixed(0) : '--';
    $('elapsedVal').textContent = fmtClock(P.pos);
  } else if (S.mode === 'live' && S.live.store && S.live.store.n > 0) {
    const L = S.live;
    const target = L.store.tEnd - LIVE_JITTER_S;
    if (!isFinite(L.t)) L.t = target;
    else {
      L.t += dt;                                   // advance the display clock in real time
      const drift = target - L.t;
      if (Math.abs(drift) > 0.6) L.t = target;     // big jump (gap/reconnect) -> snap
      else L.t += drift * 0.06;                    // otherwise ease toward the buffer target
    }
    store = L.store; cycle = L.cycle; tNow = L.t; primary = L.leads.includes('II') ? 'II' : (L.leads[0] || 'I');
  }
  if (store) scope.draw(store, tNow, cycle, primary); else scope.clear();

  // heart
  const st = (cycle && isFinite(tNow)) ? cycle.stateAt(tNow) : null;
  if (heart) { heart.setState(st); heart.render(); }
  updateHud(st);
  requestAnimationFrame(frame);
}
function fmtDur(sec) { return sec >= 60 ? `${(sec / 60).toFixed(sec % 60 ? 1 : 0)} min` : `${sec.toFixed(0)} s`; }
function fmtClock(t) { const m = Math.floor(t / 60), s = t - m * 60; return `${m}:${s < 10 ? '0' : ''}${s.toFixed(1)}`; }
let lastPhaseKey = null;
function updateHud(st) {
  const ph = st ? st.phase : PHASES.DIA;
  if (ph.key !== lastPhaseKey) {
    lastPhaseKey = ph.key;
    $('phaseName').textContent = ph.name; $('phaseDesc').textContent = ph.desc;
    const sw = $('phaseChip').querySelector('.sw'); sw.style.background = ph.color; sw.style.boxShadow = `0 0 10px ${ph.color}`;
  }
  $('mAtria').style.width = ((st ? st.a : 0) * 100) + '%';
  $('mVent').style.width = ((st ? st.v : 0) * 100) + '%';
  $('mEject').style.width = ((st ? st.eject : 0) * 100) + '%';
}

// ---------------------------------------------------------------- boot
window.kardia = { S, seek, setRate, replayReading, exitPlayback, scope, heart, renderScreen, addEvents };   // debug handle
(async function boot() {
  wsConnect();
  await refreshLibrary();
  updateModeUI();
  requestAnimationFrame(frame);
})();
