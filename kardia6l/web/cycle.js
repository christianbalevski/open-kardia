// cycle.js — maps ECG timing (R-peaks + P/QRS/T fiducials) onto the mechanical and
// electrical state of the heart at any instant.  Pure math, no rendering.
//
// Physiology encoded here (adult resting heart, textbook electromechanical delays):
//   P wave        atrial depolarization: a wavefront leaves the SA node (high right atrium)
//                 and sweeps across RA then LA over the P duration. Atrial contraction
//                 (the "atrial kick") starts ~30 ms after P onset and peaks just after the
//                 end of P, i.e. during the PR segment.
//   PR segment    the impulse is delayed in the AV node, then races down the His bundle.
//   QRS           ventricular depolarization: septum first, then apex, then the free walls
//                 base-ward. Isovolumetric contraction begins ~40 ms after QRS onset (≈ the
//                 R peak; S1). Ejection begins ~60 ms later when the aortic/pulmonary
//                 valves open.
//   ST segment    ventricles fully depolarized, contracting, ejecting.
//   T wave        ventricular repolarization; ejection tails off and ends ≈ end of T
//                 (aortic valve closes; S2). Then isovolumetric relaxation (~70 ms), rapid
//                 filling (~200 ms), diastasis, and finally the next atrial kick tops the
//                 ventricles up (last ~10 % of filling) just before the next QRS.
//
// All times are seconds on the signal's own time base.  Fiducials are RELATIVE TO R.

export const DEFAULT_FID = { p_on: -0.20, p_off: -0.10, q_on: -0.04, q_off: 0.05, t_on: 0.20, t_off: 0.38 };

const EM_DELAY_A = 0.03;    // atrial electromechanical delay
const EM_DELAY_V = 0.04;    // ventricular electromechanical delay (QRS onset -> tension)
const PRE_EJECT  = 0.06;    // isovolumetric contraction time (tension -> valve open)
const IVRT       = 0.07;    // isovolumetric relaxation time
const RAPID_FILL = 0.22;    // rapid (E-wave) filling duration
const A_KICK     = 0.07;    // fraction of ventricular volume from the atrial kick

export const PHASES = {
  P:   { key: 'P',   name: 'P wave',       desc: 'atria depolarize → atria contract',              color: '#f6a609' },
  PR:  { key: 'PR',  name: 'PR segment',   desc: 'AV-node delay · atrial kick fills the ventricles', color: '#f6a609' },
  QRS: { key: 'QRS', name: 'QRS complex',  desc: 'ventricles depolarize → contraction begins (S1)', color: '#39d0d8' },
  ST:  { key: 'ST',  name: 'ST segment',   desc: 'ventricles contract · blood ejected into aorta & pulmonary artery', color: '#39d0d8' },
  T:   { key: 'T',   name: 'T wave',       desc: 'ventricles repolarize · ejection ends (S2)',      color: '#b48cff' },
  DIA: { key: 'DIA', name: 'Diastole',     desc: 'ventricles relax and fill',                       color: '#5a6673' },
  PVC: { key: 'PVC', name: 'Ventricular ectopic beat', desc: 'premature wide QRS from the ventricle — no P wave, no atrial kick', color: '#f04747' },
};

export function defaultFidForRR(rr) {
  const r = (rr && isFinite(rr) && rr > 0.3 && rr < 2.5) ? rr : 0.85;
  const qt = 0.40 * Math.sqrt(r);
  const q_on = -0.04;
  return { p_on: q_on - 0.16, p_off: q_on - 0.06, q_on, q_off: q_on + 0.09, t_on: q_on + qt - 0.18, t_off: q_on + qt };
}

export function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
export function smoothstep(a, b, x) {
  if (b <= a) return x >= b ? 1 : 0;
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
}
function gauss(x, mu, sigma) { const d = (x - mu) / sigma; return Math.exp(-0.5 * d * d); }

export class CardiacCycle {
  constructor() {
    this.beats = [];              // sorted R times (s)
    this.labels = [];             // per-beat 'N' | 'S' | 'V' | '?'
    this.fid = { ...DEFAULT_FID };
    this.fidSource = 'default';
    this.hr = NaN;
  }

  setFid(fid, source) {
    if (fid && ['p_on', 'p_off', 'q_on', 'q_off', 't_on', 't_off'].every(k => typeof fid[k] === 'number' && isFinite(fid[k]))) {
      this.fid = { ...fid };
      this.fidSource = source || 'delineated';
    }
  }

  setBeats(times, labels) {
    const pairs = Array.from(times || []).map((t, i) => [t, (labels && labels[i]) || 'N']).filter(p => Number.isFinite(p[0]));
    pairs.sort((a, b) => a[0] - b[0]);
    this.beats = pairs.map(p => p[0]);
    this.labels = pairs.map(p => p[1]);
    this._updateHR();
  }

  /** Beat label ('N' | 'S' | 'V' | '?') for the beat at time R (exact match), default 'N'. */
  labelOf(R) {
    if (!this.labels || !this.labels.length) return 'N';
    const b = this.beats;
    let lo = 0, hi = b.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (b[m] < R) lo = m + 1; else hi = m; }
    return (lo < b.length && b[lo] === R) ? this.labels[lo] : 'N';
  }

  /** Live use: merge a fresh window of detections into the running list (dedupe 120 ms). */
  mergeBeats(times, keepFrom, labels) {
    const inc = Array.from(times || []).map((t, i) => [t, (labels && labels[i]) || 'N']).filter(p => Number.isFinite(p[0]));
    inc.sort((a, b) => a[0] - b[0]);
    if (!inc.length) return;
    const lo = inc[0][0] - 0.12;
    // keep our beats before the incoming window's start, then trust the incoming window
    // (the detector re-evaluates its whole window each time, so it may drop/shift late ones)
    const keptT = [], keptL = [];
    for (let i = 0; i < this.beats.length; i++) if (this.beats[i] < lo && (keepFrom === undefined || this.beats[i] >= keepFrom)) { keptT.push(this.beats[i]); keptL.push(this.labels ? this.labels[i] : 'N'); }
    this.beats = keptT.concat(inc.map(p => p[0]));
    this.labels = keptL.concat(inc.map(p => p[1]));
    this._updateHR();
  }

  clear() { this.beats = []; this.labels = []; this.hr = NaN; }

  _updateHR() {
    const b = this.beats;
    if (b.length < 2) { this.hr = NaN; return; }
    const rr = [];
    for (let i = Math.max(1, b.length - 8); i < b.length; i++) {
      const d = b[i] - b[i - 1];
      if (d > 0.25 && d < 3) rr.push(d);
    }
    if (!rr.length) { this.hr = NaN; return; }
    rr.sort((a, b) => a - b);
    this.hr = 60 / rr[Math.floor(rr.length / 2)];
  }

  /** Local heart rate at time t: median of the last few RR intervals ending before t. */
  hrAt(t) {
    const b = this.beats; if (b.length < 2) return NaN;
    let lo = 0, hi = b.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (b[m] <= t) lo = m + 1; else hi = m; }
    const rr = [];
    for (let i = lo - 1; i >= 1 && rr.length < 5; i--) { const d = b[i] - b[i - 1]; if (d > 0.25 && d < 3) rr.push(d); }
    if (!rr.length) return this.hr;
    rr.sort((a, b) => a - b); return 60 / rr[Math.floor(rr.length / 2)];
  }

  medianRR() { return isFinite(this.hr) && this.hr > 0 ? 60 / this.hr : NaN; }

  /** Beats whose influence window contains t: R in [t-1.6, t+0.7]. */
  _nearby(t) {
    const b = this.beats, out = [];
    // binary search for first R >= t - 1.6
    let lo = 0, hi = b.length;
    const from = t - 1.6;
    while (lo < hi) { const m = (lo + hi) >> 1; if (b[m] < from) lo = m + 1; else hi = m; }
    for (let i = lo; i < b.length && b[i] <= t + 0.7; i++) out.push(b[i]);
    return out;
  }

  /**
   * Full state at time t. Returns:
   *  a       0..1 atrial contraction
   *  v       0..1 ventricular contraction (0 = end-diastolic volume, 1 = end-systolic)
   *  eject   0..1 outflow (aortic/pulmonary flow)
   *  pulse   0..1 arterial pressure pulse (for vessel distension)
   *  atria   {front:0..1, plateau:0..1}    depolarization sweep across the atria
   *  vent    {front:0..1, plateau:0..1}    depolarization sweep across the ventricles
   *  sa, av, his  0..1 glows for the conduction system
   *  phase   PHASES entry, beat: R time of the governing beat (or null), sinceR
   */
  stateAt(t) {
    const F = this.fid;
    const s = { a: 0, v: 0, eject: 0, pulse: 0, atria: { front: 0, plateau: 0 }, vent: { front: 0, plateau: 0 },
                sa: 0, av: 0, his: 0, phase: PHASES.DIA, beat: null, sinceR: NaN, inCycle: false };
    const nb = this._nearby(t);
    if (!nb.length) return s;

    let phaseBeat = null, phaseKey = null;
    let aNext = 0;               // atrial kick that belongs to the *next* beat (tops up filling)
    for (const R of nb) {
      const lab = this.labelOf(R);
      const isV = lab === 'V';
      // A ventricular ectopic beat has no preceding P wave / atrial kick, a wide QRS and a
      // wider T; a supraventricular premature beat has a (premature) P and a normal QRS.
      const qOn = R + F.q_on - (isV ? 0.02 : 0), qOff = R + F.q_off + (isV ? 0.05 : 0);
      const tOn = R + F.t_on + (isV ? 0.03 : 0), tOff = R + F.t_off + (isV ? 0.06 : 0);
      const pOn = isV ? NaN : R + F.p_on, pOff = isV ? NaN : R + F.p_off;

      // ---------------- atria: contraction envelope
      const aStart = pOn + EM_DELAY_A, aPeak = pOff + 0.05, aEnd = qOn + 0.12;
      const a = isV ? 0 : smoothstep(aStart, aPeak, t) * (1 - smoothstep(aPeak, aEnd, t));
      if (a > s.a) s.a = a;
      // reservoir/filling of atria while ventricles contract is handled by the renderer via v
      if (t < R) aNext = Math.max(aNext, a);

      // ---------------- ventricles: contraction envelope
      const onset = qOn + EM_DELAY_V;                 // isovolumetric contraction (≈ R, S1)
      const ejStart = onset + PRE_EJECT;              // valves open
      const es = tOff + 0.03;                         // end-systole ≈ end of T (S2)
      let v = 0;
      if (t >= onset && t < es) {
        const u = clamp01((t - onset) / (es - onset));
        v = Math.pow(Math.sin(u * Math.PI / 2), 1.15);         // fast early, saturating
      } else if (t >= es) {
        const d = t - es;
        const f = 0.15 * smoothstep(0, IVRT, d)                        // isovolumetric relaxation
                + 0.72 * smoothstep(IVRT, IVRT + RAPID_FILL, d)         // rapid filling (E)
                + 0.08 * smoothstep(IVRT + RAPID_FILL, IVRT + RAPID_FILL + 0.35, d);   // diastasis
        v = Math.max(0.05, 1 - f);                                     // ~5 % left for the atrial kick
        if (d > 1.4) v = 0.05;
      }
      if (v > s.v) s.v = v;

      // ---------------- ejection & arterial pulse
      let e = 0;
      if (t >= ejStart && t < es) {
        const u = clamp01((t - ejStart) / (es - ejStart));
        e = Math.sin(Math.PI * Math.pow(u, 0.7));
      }
      if (e > s.eject) s.eject = e;
      let p = 0;
      if (t >= ejStart) {
        if (t < es) p = smoothstep(ejStart, ejStart + 0.10, t);
        else p = Math.exp(-(t - es) / 0.28) * (1 + 0.12 * gauss(t - es, 0.05, 0.03));  // dicrotic bump
      }
      if (p > s.pulse) s.pulse = p;

      // ---------------- electrical: atria (SA node -> RA -> LA over the P wave)
      const pDur = Math.max(0.04, pOff - pOn);
      if (!isV && t >= pOn - 0.01) {
        const front = clamp01((t - pOn) / pDur);
        // depolarized tissue stays 'lit' until atrial repolarization (hidden in the QRS)
        const plateau = (t < qOn) ? 1 : (1 - smoothstep(qOn, qOff + 0.06, t));
        if (t <= qOff + 0.06 && (front > s.atria.front || plateau > s.atria.plateau)) {
          s.atria = { front: front, plateau: plateau * (front > 0 ? 1 : 0) };
        }
      }
      // ---------------- electrical: ventricles (septum/apex -> base over the QRS)
      const qDur = Math.max(0.03, qOff - qOn);
      if (t >= qOn - 0.005) {
        const front = clamp01((t - qOn) / qDur);
        const plateau = (t < tOn) ? 1 : (1 - smoothstep(tOn, tOff, t));
        if (t <= tOff && (front > s.vent.front || plateau > s.vent.plateau)) s.vent = { front, plateau };
      }
      // ---------------- conduction system glows
      if (!isV) {
        s.sa  = Math.max(s.sa,  gauss(t, pOn - 0.01, 0.03));
        s.av  = Math.max(s.av,  smoothstep(pOn + 0.4 * pDur, pOff, t) * (1 - smoothstep(qOn - 0.02, qOn + 0.02, t)));
        s.his = Math.max(s.his, gauss(t, qOn + 0.01, 0.025));
      }

      // ---------------- phase label (later beat wins)
      let key = null;
      if (!isV && t >= pOn && t < pOff) key = 'P';
      else if (!isV && t >= pOff && t < qOn) key = 'PR';
      else if (t >= qOn && t < qOff) key = isV ? 'PVC' : 'QRS';
      else if (t >= qOff && t < tOn) key = 'ST';
      else if (t >= tOn && t < tOff) key = 'T';
      if (key) { phaseKey = key; phaseBeat = R; }
      if (t >= R && (s.beat === null || R > s.beat)) { s.beat = R; s.sinceR = t - R; }
    }
    // atrial kick tops up the ventricles at the very end of diastole
    s.v = Math.max(0, s.v - A_KICK * aNext);
    if (phaseKey) { s.phase = PHASES[phaseKey]; s.inCycle = true; if (phaseBeat !== null) { s.beat = phaseBeat; s.sinceR = t - phaseBeat; } }
    return s;
  }
}
