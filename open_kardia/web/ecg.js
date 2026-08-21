// ecg.js — sample store + canvas ECG scope (live scroll or playback with a playhead).

/** Continuous per-lead sample buffers on one absolute time base (seconds). */
export class SampleStore {
  constructor(fs, leads, maxSeconds = 90) {
    this.fs = fs;
    this.leads = leads.slice();
    this.maxN = Math.round(maxSeconds * fs);
    this.t0 = null;                 // time of index 0
    this.n = 0;                     // samples stored
    this.cap = Math.round(30 * fs);
    this.data = this.leads.map(() => new Float32Array(this.cap));
  }
  get tEnd() { return this.t0 === null || this.n === 0 ? NaN : this.t0 + (this.n - 1) / this.fs; }
  get tStart() { return this.t0 === null ? NaN : this.t0; }
  _grow(need) {
    while (this.cap < need) this.cap *= 2;
    this.data = this.data.map(a => { const b = new Float32Array(this.cap); b.set(a.subarray(0, this.n)); return b; });
  }
  /** Append a chunk starting at time t0 (rows = leads, contiguous samples). */
  append(t0, cols) {
    const len = cols[0]?.length || 0;
    if (!len) return;
    if (this.t0 === null) { this.t0 = t0; this.n = 0; }
    let i0 = Math.round((t0 - this.t0) * this.fs);
    if (i0 < 0) {  // shouldn't happen; drop overlapping head
      const cut = -i0;
      if (cut >= len) return;
      cols = cols.map(c => c.slice(cut)); i0 = 0;
    }
    const end = i0 + len;
    if (end > this.cap) this._grow(end);
    if (i0 > this.n) { for (const a of this.data) a.fill(NaN, this.n, i0); }   // gap
    for (let j = 0; j < this.data.length; j++) {
      const src = cols[j];
      if (!src) { this.data[j].fill(NaN, i0, end); continue; }
      this.data[j].set(src, i0);
    }
    this.n = Math.max(this.n, end);
    // trim old history
    if (this.n > this.maxN) {
      const drop = this.n - Math.round(this.maxN * 0.8);
      for (const a of this.data) a.copyWithin(0, drop, this.n);
      this.n -= drop; this.t0 += drop / this.fs;
    }
  }
  /** Set the whole buffer at once (playback). */
  setAll(t0, cols) {
    this.t0 = t0; this.n = cols[0].length; this.cap = this.n;
    this.data = cols.map(c => Float32Array.from(c));
  }
  leadIndex(name) { return this.leads.indexOf(name); }
}

const COLORS = { trace: '#39d0d8', grid: '#243038', gridMinor: '#182028', text: '#5a6673', now: '#e8edf2',
                 p: '#f6a609', qrs: '#39d0d8', t: '#b48cff', r: '#f04747', v: '#ff6a3d', s: '#f6a609' };

export class ECGScope {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.leadsShown = ['I', 'II'];
    this.back = 5.0;          // seconds shown before the cursor
    this.fwd = 0.15;          // seconds shown after the cursor
    this.scale = {};          // lead -> {lo,hi} smoothed
    this.dpr = window.devicePixelRatio || 1;
    this.stripH = 14;
  }
  setLeads(names) { this.leadsShown = names.slice(); }
  /** Episodes to shade: [{t0, t1, color, label}] on the store's time base. */
  setEpisodes(list) { this.episodes = list || []; }
  setWindow(back, fwd) { this.back = back; this.fwd = fwd; }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * this.dpr)), h = Math.max(1, Math.round(r.height * this.dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.W = r.width; this.H = r.height;
  }
  clear() {
    this.resize();
    const c = this.ctx; c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.W, this.H);
  }

  /**
   * store: SampleStore, tNow: cursor time, cycle: CardiacCycle (for the phase strip),
   * primary: lead name to draw R markers on.
   */
  draw(store, tNow, cycle, primary) {
    this.resize();
    const c = this.ctx, W = this.W, H = this.H;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const leftPad = 34, rightPad = 6, topPad = 6, botPad = this.stripH + 16;
    const plotW = W - leftPad - rightPad, plotH = H - topPad - botPad;
    const t0 = tNow - this.back, t1 = tNow + this.fwd, span = t1 - t0;
    const xOf = t => leftPad + (t - t0) / span * plotW;
    const leads = this.leadsShown;
    const laneH = plotH / leads.length;

    // ---- grid (25 mm/s: 0.04 s minor, 0.2 s major)
    const pxPerSec = plotW / span;
    c.lineWidth = 1;
    const majorEvery = 0.2, minorEvery = 0.04;
    const gStart = Math.ceil(t0 / minorEvery) * minorEvery;
    if (pxPerSec * minorEvery >= 4) {
      c.strokeStyle = COLORS.gridMinor; c.beginPath();
      for (let t = gStart; t <= t1; t += minorEvery) {
        const k = Math.round(t / minorEvery); if (k % 5 === 0) continue;
        const x = Math.round(xOf(t)) + 0.5; c.moveTo(x, topPad); c.lineTo(x, topPad + plotH);
      }
      c.stroke();
    }
    c.strokeStyle = COLORS.grid; c.beginPath();
    for (let t = Math.ceil(t0 / majorEvery) * majorEvery; t <= t1; t += majorEvery) {
      const x = Math.round(xOf(t)) + 0.5; c.moveTo(x, topPad); c.lineTo(x, topPad + plotH);
    }
    c.stroke();
    // time labels every 1 s
    c.fillStyle = COLORS.text; c.font = '10px ui-monospace, Menlo, monospace'; c.textAlign = 'center';
    for (let t = Math.ceil(t0); t <= t1; t += 1) {
      const x = xOf(t); if (x < leftPad + 8 || x > W - 10) continue;
      c.fillText(fmtT(t), x, H - 3);
    }

    // ---- arrhythmia episodes (shaded spans behind the traces)
    const eps = this.episodes || [];
    if (eps.length) {
      c.save(); c.font = 'bold 9px -apple-system, Helvetica, sans-serif'; c.textAlign = 'left';
      let lastLabelX = -1e9;
      for (const e of eps) {
        if (e.t1 < t0 || e.t0 > t1) continue;
        let xa = Math.max(leftPad, xOf(e.t0)), xb = Math.min(W - rightPad, xOf(e.t1));
        if (xb - xa < 4) { const m = (xa + xb) / 2; xa = m - 2; xb = m + 2; }
        c.fillStyle = e.color; c.globalAlpha = 0.13; c.fillRect(xa, topPad, xb - xa, plotH);
        c.globalAlpha = 0.9; c.fillRect(xa, topPad, xb - xa, 2);
        if (e.label && xa - lastLabelX > 30) { c.fillStyle = e.color; c.fillText(e.label, xa + 2, topPad + 11); lastLabelX = xa; }
        c.globalAlpha = 1;
      }
      c.restore();
    }

    // ---- lanes
    const i0 = store && store.t0 !== null ? Math.max(0, Math.floor((t0 - store.t0) * store.fs)) : 0;
    const i1 = store && store.t0 !== null ? Math.min(store.n, Math.ceil((t1 - store.t0) * store.fs) + 1) : 0;
    leads.forEach((lead, li) => {
      const yTop = topPad + li * laneH, yMid = yTop + laneH / 2;
      // lane separator
      if (li > 0) { c.strokeStyle = COLORS.grid; c.beginPath(); c.moveTo(leftPad, Math.round(yTop) + 0.5); c.lineTo(W - rightPad, Math.round(yTop) + 0.5); c.stroke(); }
      c.fillStyle = '#9fb0bf'; c.font = 'bold 11px -apple-system, Helvetica, sans-serif'; c.textAlign = 'left';
      c.fillText(lead, 6, yTop + 13);
      const col = store ? store.leadIndex(lead) : -1;
      if (col < 0 || i1 <= i0) {
        c.strokeStyle = COLORS.grid; c.beginPath(); c.moveTo(leftPad, yMid + 0.5); c.lineTo(W - rightPad, yMid + 0.5); c.stroke();
        if (col < 0 && store) { c.fillStyle = COLORS.text; c.font = '10px -apple-system, Helvetica, sans-serif'; c.fillText('not in this recording', 6, yTop + 26); }
        return;
      }
      const arr = store.data[col];
      // autoscale on the visible window: robust percentiles on a decimated subset
      const sc = this._scale(lead, arr, i0, i1);
      const mvPerPx = (sc.hi - sc.lo) / (laneH * 0.82);
      const yOf = v => yMid - (v - (sc.hi + sc.lo) / 2) / mvPerPx;
      // 0.5 mV horizontal grid lines when legible
      const stepPx = 0.5 / mvPerPx;
      if (stepPx > 9) {
        c.strokeStyle = COLORS.gridMinor; c.beginPath();
        for (let k = -6; k <= 6; k++) { const y = Math.round(yOf(k * 0.5)) + 0.5; if (y > yTop + 2 && y < yTop + laneH - 2) { c.moveTo(leftPad, y); c.lineTo(W - rightPad, y); } }
        c.stroke();
      }
      // trace
      c.save(); c.beginPath(); c.rect(leftPad, yTop, plotW, laneH); c.clip();
      c.strokeStyle = COLORS.trace; c.lineWidth = 1.4; c.lineJoin = 'round'; c.beginPath();
      let pen = false;
      const step = Math.max(1, Math.floor((i1 - i0) / (plotW * 2)));
      for (let i = i0; i < i1; i += step) {
        const v = arr[i];
        if (Number.isNaN(v)) { pen = false; continue; }
        const x = xOf(store.t0 + i / store.fs), y = yOf(v);
        if (!pen) { c.moveTo(x, y); pen = true; } else c.lineTo(x, y);
      }
      c.stroke();
      // R markers on the primary lead
      if (cycle && lead === primary) {
        c.font = 'bold 10px -apple-system, Helvetica, sans-serif'; c.textAlign = 'center';
        for (let k = 0; k < cycle.beats.length; k++) {
          const R = cycle.beats[k];
          if (R < t0 || R > t1) continue;
          const lab = cycle.labels && cycle.labels[k] || 'N';
          const x = xOf(R);
          if (lab === 'V' || lab === 'S' || lab === '?') {
            c.fillStyle = lab === 'V' ? COLORS.v : (lab === 'S' ? COLORS.s : COLORS.text);
            c.fillText(lab, x, yTop + laneH - 3);
          } else {
            c.fillStyle = COLORS.r;
            c.beginPath(); c.moveTo(x - 3, yTop + laneH - 2); c.lineTo(x + 3, yTop + laneH - 2); c.lineTo(x, yTop + laneH - 7); c.fill();
          }
        }
      }
      c.restore();
    });

    // ---- phase strip: P / QRS / T bands for every beat in view
    const sy = topPad + plotH + 3, sh = this.stripH - 4;
    c.fillStyle = '#0f151b'; c.fillRect(leftPad, sy, plotW, sh);
    if (cycle && cycle.beats.length) {
      const F = cycle.fid;
      const bands = [['p_on', 'p_off', COLORS.p], ['q_on', 'q_off', COLORS.qrs], ['t_on', 't_off', COLORS.t]];
      for (let k = 0; k < cycle.beats.length; k++) {
        const R = cycle.beats[k];
        if (R + F.t_off < t0 || R + F.p_on > t1) continue;
        const isV = cycle.labels && cycle.labels[k] === 'V';
        for (const [a, b, colr0] of bands) {
          if (isV && a === 'p_on') continue;                    // ectopic ventricular beat: no P wave
          const colr = (isV && a === 'q_on') ? COLORS.v : colr0;
          const xa = Math.max(leftPad, xOf(R + F[a])), xb = Math.min(W - rightPad, xOf(R + F[b]));
          if (xb > xa) { c.fillStyle = colr; c.globalAlpha = 0.85; c.fillRect(xa, sy, xb - xa, sh); c.globalAlpha = 1; }
        }
      }
    }
    // ---- cursor
    const xn = xOf(tNow);
    c.strokeStyle = COLORS.now; c.globalAlpha = 0.9; c.lineWidth = 1.2; c.beginPath();
    c.moveTo(Math.round(xn) + 0.5, topPad); c.lineTo(Math.round(xn) + 0.5, topPad + plotH + this.stripH); c.stroke();
    c.globalAlpha = 1;
    // future region tint (playback)
    if (this.fwd > 0.2) { c.fillStyle = 'rgba(255,255,255,0.03)'; c.fillRect(xn, topPad, W - rightPad - xn, plotH); }
  }

  _scale(lead, arr, i0, i1) {
    const n = i1 - i0;
    let lo = Infinity, hi = -Infinity;
    if (n > 4) {
      const step = Math.max(1, Math.floor(n / 400));
      const vals = [];
      for (let i = i0; i < i1; i += step) { const v = arr[i]; if (!Number.isNaN(v)) vals.push(v); }
      if (vals.length > 4) {
        vals.sort((a, b) => a - b);
        lo = vals[Math.floor(vals.length * 0.01)]; hi = vals[Math.floor(vals.length * 0.99)];
      }
    }
    if (!isFinite(lo) || !isFinite(hi)) { lo = -0.6; hi = 0.6; }
    let span = Math.max(0.6, hi - lo);
    const mid = (hi + lo) / 2, pad = 0.25 * span;
    const target = { lo: mid - span / 2 - pad, hi: mid + span / 2 + pad };
    const prev = this.scale[lead];
    if (!prev) { this.scale[lead] = target; return target; }
    const k = 0.15;   // smooth to avoid jitter
    prev.lo += (target.lo - prev.lo) * k; prev.hi += (target.hi - prev.hi) * k;
    return prev;
  }
}

function fmtT(t) {
  const neg = t < 0; t = Math.abs(t);
  const m = Math.floor(t / 60), s = t - m * 60;
  return (neg ? '-' : '') + (m ? `${m}:${s < 10 ? '0' : ''}${s.toFixed(0)}` : `${s.toFixed(0)}s`);
}
