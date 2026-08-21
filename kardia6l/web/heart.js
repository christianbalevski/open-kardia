// heart.js — a clean, stylized-anatomical 3D heart (three.js) whose parts move on the
// timeline produced by cycle.js:
//   * atria contract after the P wave, ventricles after the QRS, relax after the T wave
//   * an emissive depolarization wavefront sweeps the atria (SA node → LA) during P and
//     the ventricles (apex/septum → base) during QRS; the "lit" (depolarized) tissue fades
//     during the T wave (repolarization)
//   * ejection distends the aorta / pulmonary trunk; the conduction system (SA/AV nodes,
//     His bundle, bundle branches, Purkinje) glows in "conduction" (x-ray) mode
//
// Anatomy is procedural (no downloaded model): everything is authored in ONE local frame
// "L" — origin at the LV apex, +y along the long axis toward the base, +x toward the
// patient's LEFT (viewer's right), +z anterior — and then tilted into the world so the
// apex points down/left/forward like a real heart in the chest.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const V3 = (x, y, z) => new THREE.Vector3(x, y, z);

// -------------------------------------------------------------------- materials
function muscleMaterial(color, glowColor, opts = {}) {
  const m = new THREE.MeshPhysicalMaterial({
    color, roughness: opts.roughness ?? 0.52, metalness: 0.0,
    clearcoat: opts.clearcoat ?? 0.35, clearcoatRoughness: 0.4,
    sheen: 0.35, sheenRoughness: 0.6, sheenColor: new THREE.Color(0xff9a8a),
    emissive: new THREE.Color(0x000000),
  });
  const u = {
    uOrigin: { value: new THREE.Vector3() }, uDir: { value: new THREE.Vector3(0, 1, 0) },
    uFrontPos: { value: -1e3 }, uFrontWidth: { value: 1.0 },
    uPlateau: { value: 0 }, uFrontGlow: { value: 0 }, uGlowColor: { value: new THREE.Color(glowColor) },
  };
  m.userData.u = u;
  m.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, u);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWPos;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\n vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec3 vWPos; uniform vec3 uOrigin; uniform vec3 uDir; uniform float uFrontPos; uniform float uFrontWidth;
        uniform float uPlateau; uniform float uFrontGlow; uniform vec3 uGlowColor;`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        { float d = dot(vWPos - uOrigin, uDir);
          float act = 1.0 - smoothstep(uFrontPos - uFrontWidth, uFrontPos + uFrontWidth, d);
          // depolarized tissue: tint the muscle toward the wave colour (not additive white)
          diffuseColor.rgb = mix(diffuseColor.rgb, uGlowColor, clamp(uPlateau * act * 0.62, 0.0, 1.0)); }`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        { float d = dot(vWPos - uOrigin, uDir);
          float ring = exp(-pow((d - uFrontPos) / (uFrontWidth * 0.7), 2.0));
          totalEmissiveRadiance += uGlowColor * (uFrontGlow * ring * 0.55); }`);
  };
  m.customProgramCacheKey = () => 'kardia-wavefront';
  return m;
}
function plainMaterial(color, opts = {}) {
  return new THREE.MeshPhysicalMaterial({
    color, roughness: opts.roughness ?? 0.5, metalness: 0, clearcoat: opts.clearcoat ?? 0.3, clearcoatRoughness: 0.45,
    emissive: new THREE.Color(opts.emissive ?? 0x000000), emissiveIntensity: 0,
  });
}
function glowMaterial(color) {
  return new THREE.MeshStandardMaterial({ color, emissive: new THREE.Color(color), emissiveIntensity: 0.15, roughness: 0.6, metalness: 0 });
}

// -------------------------------------------------------------------- geometry helpers
function ellipsoid(center, radii, material, seg = 40) {
  const g = new THREE.SphereGeometry(1, seg, Math.round(seg * 0.75));
  const m = new THREE.Mesh(g, material);
  m.position.copy(center); m.scale.copy(radii);
  m.userData.center = center.clone(); m.userData.radii = radii.clone();
  return m;
}
/** Bullet-shaped ventricle: lathe profile from the apex (y=0) to the base (y=H) + a shallow cap. */
function ventricleGeometry(R, H, cap = 0.7, seg = 56) {
  const pts = [];
  const N = 40;
  for (let i = 0; i <= N; i++) {
    const s = i / N;
    const r = R * Math.sqrt(1 - Math.pow(1 - s, 2.4)) * (1 - 0.10 * Math.pow(s, 3));
    pts.push(new THREE.Vector2(i === 0 ? 0.0001 : r, s * H));
  }
  const rTop = pts[pts.length - 1].x;
  for (let i = 1; i <= 8; i++) {
    const u = i / 8;
    pts.push(new THREE.Vector2(Math.max(0.0001, rTop * Math.cos(u * Math.PI / 2)), H + cap * Math.sin(u * Math.PI / 2)));
  }
  const g = new THREE.LatheGeometry(pts, seg);
  g.computeVertexNormals();
  return g;
}
function lvRadiusAt(y, R, H) {
  const s = Math.max(0, Math.min(1, y / H));
  return R * Math.sqrt(1 - Math.pow(1 - s, 2.4)) * (1 - 0.10 * Math.pow(s, 3));
}
function tube(points, radius, material, opts = {}) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => p.clone()), false, 'catmullrom', opts.tension ?? 0.5);
  const g = new THREE.TubeGeometry(curve, opts.segments ?? 48, radius, opts.radial ?? 20, false);
  const m = new THREE.Mesh(g, material);
  if (opts.centered) {
    g.computeBoundingBox();
    const c = g.boundingBox.getCenter(new THREE.Vector3());
    g.translate(-c.x, -c.y, -c.z); m.position.copy(c);
  }
  return m;
}
function taperedTube(points, r0, r1, material, segments = 40, radial = 20) {
  // TubeGeometry has one radius; emulate a taper by scaling ring vertices along the curve
  const curve = new THREE.CatmullRomCurve3(points.map(p => p.clone()));
  const g = new THREE.TubeGeometry(curve, segments, 1, radial, false);
  const pos = g.attributes.position;
  const ringN = radial + 1;
  for (let i = 0; i < pos.count; i++) {
    const seg = Math.floor(i / ringN);
    const u = seg / segments;
    const r = r0 + (r1 - r0) * u;
    const c = curve.getPointAt(u);
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    pos.setXYZ(i, c.x + (x - c.x) * r, c.y + (y - c.y) * r, c.z + (z - c.z) * r);
  }
  pos.needsUpdate = true; g.computeVertexNormals();
  return new THREE.Mesh(g, material);
}
function projectToEllipsoid(p, e, lift = 1.03) {
  const c = e.userData.center, r = e.userData.radii;
  const d = p.clone().sub(c);
  const f = Math.sqrt((d.x / r.x) ** 2 + (d.y / r.y) ** 2 + (d.z / r.z) ** 2) || 1;
  return c.clone().add(d.multiplyScalar(lift / f));
}
function insideEllipsoid(p, e) {
  const c = e.userData.center, r = e.userData.radii;
  return ((p.x - c.x) / r.x) ** 2 + ((p.y - c.y) / r.y) ** 2 + ((p.z - c.z) / r.z) ** 2 < 1;
}

// -------------------------------------------------------------------- the view
export class HeartView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.5;
    pmrem.dispose();

    this.camera = new THREE.PerspectiveCamera(34, 1, 0.1, 500);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 10; this.controls.maxDistance = 70;
    this.controls.autoRotateSpeed = 0.7;

    this._lights();
    this._build();
    this.setXray(false);
    this.resetView();
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this.state = null;
    this._tmpV = new THREE.Vector3(); this._tmpV2 = new THREE.Vector3(); this._tmpQ = new THREE.Quaternion();
    this._cyan = new THREE.Color(0x2ab8c2); this._violet = new THREE.Color(0x9d7ce8); this._amber = new THREE.Color(0xe89a1a);
  }

  _lights() {
    const key = new THREE.DirectionalLight(0xfff1e0, 2.4); key.position.set(7, 12, 9); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xa9c8ff, 0.9); fill.position.set(-9, 3, 7); this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0x8fe0ff, 1.3); rim.position.set(-4, 8, -10); this.scene.add(rim);
    this.scene.add(new THREE.HemisphereLight(0x3a4757, 0x140b0b, 0.7));
  }

  _build() {
    // ---- the anatomical frame L -> world (apex down/left/forward, base up/right/back)
    const apexW = V3(2.9, -2.0, 1.0), baseW = V3(0.2, 3.2, -0.3);
    const yAx = baseW.clone().sub(apexW).normalize();
    const xAx = new THREE.Vector3().crossVectors(yAx, V3(0, 0, 1)).normalize();   // patient's left
    const zAx = new THREE.Vector3().crossVectors(xAx, yAx).normalize();            // anterior
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(xAx, yAx, zAx));

    this.root = new THREE.Group();          // whole heart, world placement
    this.root.position.copy(apexW); this.root.quaternion.copy(q);
    this.scene.add(this.root);
    this.baseFrame = new THREE.Group(); this.root.add(this.baseFrame);        // static (vessels)
    this.vent = new THREE.Group(); this.root.add(this.vent);                  // scales about the apex
    this.atriaPivot = V3(-0.9, 8.6, -1.0);                                    // atria scale about their roof
    this.atria = new THREE.Group(); this.atria.position.copy(this.atriaPivot); this.root.add(this.atria);
    const A = (obj) => { obj.position.sub(this.atriaPivot); this.atria.add(obj); return obj; };

    // ---- materials
    this.matVent = muscleMaterial(0xa9393b, 0x2ab8c2);
    this.matAtria = muscleMaterial(0xb8474b, 0xe89a1a, { roughness: 0.55 });
    this.matAorta = plainMaterial(0xc9524e, { emissive: 0xff5a4a });
    this.matPA = plainMaterial(0x7d6db8, { emissive: 0x9a8cff });
    this.matVein = plainMaterial(0x5b6cb4);
    this.matPV = plainMaterial(0xc25a5c);
    this.matCoronary = plainMaterial(0xd94b4b, { roughness: 0.4 });
    this.matCond = glowMaterial(0xffd75e);
    this.matSA = glowMaterial(0xfff2a8);
    this.matAV = glowMaterial(0xfff2a8);
    this.matHis = glowMaterial(0xffe27a);
    this.opaqueMats = [this.matVent, this.matAtria, this.matAorta, this.matPA, this.matVein, this.matPV, this.matCoronary];

    // ---- ventricles (frame L, apex at origin)
    const R = 2.9, H = 6.2;
    this.lv = new THREE.Mesh(ventricleGeometry(R, H, 0.7), this.matVent); this.vent.add(this.lv);
    this.rv = ellipsoid(V3(-1.8, 3.7, 1.15), V3(2.3, 2.9, 1.6), this.matVent, 48); this.vent.add(this.rv);
    this.rvot = taperedTube([V3(-1.9, 5.4, 1.45), V3(-1.6, 6.6, 1.55), V3(-1.4, 7.5, 1.7)], 1.3, 0.95, this.matVent, 24, 24);
    this.vent.add(this.rvot);
    this.ventLen = H + 0.7;

    // coronary arteries: LAD in the anterior interventricular groove (computed on the surface), RCA in the right AV groove
    const lad = [];
    for (let y = 6.3; y >= 0.6; y -= 0.5) {
      // find azimuth (from +z toward -x) where the LV surface meets the RV ellipsoid, on the front
      const r = lvRadiusAt(y, R, H) * 1.045;
      let best = null;
      for (let phi = -1.6; phi <= 1.3; phi += 0.02) {
        const p = V3(-Math.sin(phi) * r, y, Math.cos(phi) * r);   // phi=0 -> +z (front); phi>0 -> -x
        if (insideEllipsoid(p, this.rv)) { best = p; break; }     // first point inside the RV = anterior LV/RV boundary
      }
      if (best) lad.push(best.clone().multiplyScalar(1.0));
    }
    lad.push(V3(0.35, 0.25, 0.85));
    if (lad.length > 3) this.vent.add(tube(lad, 0.15, this.matCoronary, { segments: 60, radial: 10 }));
    const rcaPts = [V3(-1.0, 6.4, 1.9), V3(-2.4, 6.3, 2.05), V3(-3.6, 6.1, 1.0), V3(-4.1, 5.9, -0.5), V3(-3.4, 5.6, -1.9)]
      .map(p => projectToEllipsoid(p, this.rv, 1.02));
    this.vent.add(tube(rcaPts, 0.15, this.matCoronary, { segments: 40, radial: 10 }));

    // ---- atria (frame L, grouped so they scale about their roof)
    this.ra = A(ellipsoid(V3(-2.6, 7.6, 0.3), V3(2.1, 1.9, 2.0), this.matAtria, 44));
    this.raa = A(ellipsoid(V3(-2.9, 8.5, 2.0), V3(1.05, 0.62, 0.8), this.matAtria, 28)); this.raa.rotation.z = 0.35; this.raa.rotation.y = -0.4;
    this.la = A(ellipsoid(V3(0.9, 7.9, -2.2), V3(2.4, 1.8, 2.0), this.matAtria, 44));
    this.laa = A(ellipsoid(V3(2.1, 8.0, 0.8), V3(0.95, 0.55, 0.7), this.matAtria, 28)); this.laa.rotation.z = -0.4; this.laa.rotation.y = 0.5;
    // veins (attached to the atria so they move with them)
    A(taperedTube([V3(-2.6, 8.4, -0.3), V3(-2.4, 11.0, -0.6), V3(-2.2, 13.8, -0.7)], 0.95, 0.85, this.matVein, 20));  // SVC
    A(taperedTube([V3(-2.7, 6.6, -0.6), V3(-3.0, 4.6, -1.9), V3(-3.1, 2.4, -3.0)], 0.9, 0.85, this.matVein, 20));      // IVC
    A(tube([V3(-1.0, 8.5, -3.2), V3(-2.4, 9.1, -4.2), V3(-3.6, 9.5, -4.8)], 0.5, this.matPV, { segments: 16 }));       // RSPV
    A(tube([V3(-0.9, 7.3, -3.4), V3(-2.3, 6.9, -4.4), V3(-3.6, 6.6, -5.0)], 0.5, this.matPV, { segments: 16 }));       // RIPV
    A(tube([V3(2.6, 8.5, -3.1), V3(3.8, 9.0, -4.0), V3(4.9, 9.4, -4.6)], 0.5, this.matPV, { segments: 16 }));           // LSPV
    A(tube([V3(2.7, 7.3, -3.3), V3(3.9, 6.9, -4.2), V3(4.9, 6.5, -4.8)], 0.5, this.matPV, { segments: 16 }));           // LIPV
    this.saNodeL = V3(-3.0, 8.9, -0.6);   // SA node: roof of RA at the SVC junction
    this.laFarL = V3(3.2, 7.9, -2.4);     // far end of the LA (end of atrial depolarization)

    // ---- great vessels (static base frame)
    this.aortaRoot = ellipsoid(V3(-0.1, 6.5, -0.1), V3(1.35, 1.2, 1.35), this.matAorta, 28); this.baseFrame.add(this.aortaRoot);
    this.aorta = tube([V3(-0.1, 6.6, -0.1), V3(0.2, 8.6, 0.25), V3(0.6, 10.5, -0.1), V3(1.4, 12.0, -1.2), V3(2.6, 11.2, -2.7),
                       V3(2.95, 9.2, -3.9), V3(2.9, 6.2, -4.3), V3(2.4, 2.4, -4.4)], 1.15, this.matAorta,
                      { segments: 90, radial: 24, centered: true });
    this.baseFrame.add(this.aorta);
    for (const [a, b, r] of [[V3(0.85, 11.85, -0.55), V3(0.35, 14.3, 0.1), 0.42], [V3(1.5, 12.15, -1.15), V3(1.6, 14.4, -1.15), 0.36],
                             [V3(2.15, 12.0, -1.8), V3(2.7, 14.1, -2.3), 0.4]]) {
      this.baseFrame.add(tube([a, a.clone().lerp(b, 0.5), b], r, this.matAorta, { segments: 12 }));
    }
    this.paRoot = ellipsoid(V3(-1.4, 7.7, 1.75), V3(1.05, 0.9, 1.05), this.matPA, 24); this.baseFrame.add(this.paRoot);
    this.pa = tube([V3(-1.4, 7.7, 1.75), V3(-0.9, 9.4, 2.2), V3(0.4, 10.5, 1.9), V3(0.9, 10.7, 1.4)], 0.95, this.matPA,
                   { segments: 40, radial: 22, centered: true });
    this.baseFrame.add(this.pa);
    this.baseFrame.add(tube([V3(0.9, 10.7, 1.4), V3(-0.4, 10.2, -0.9), V3(-2.0, 10.1, -1.5), V3(-3.8, 9.9, -1.7)], 0.66, this.matPA, { segments: 30 })); // RPA
    this.baseFrame.add(tube([V3(0.9, 10.7, 1.4), V3(2.5, 10.7, 0.5), V3(3.6, 10.4, -0.5), V3(4.6, 9.9, -1.3)], 0.66, this.matPA, { segments: 30 }));   // LPA

    // ---- conduction system (frame L; ventricular part in the vent group so it contracts too)
    this.cond = new THREE.Group(); this.root.add(this.cond);
    this.condVent = new THREE.Group(); this.vent.add(this.condVent);
    this.saNode = new THREE.Mesh(new THREE.SphereGeometry(0.3, 20, 16), this.matSA); this.saNode.position.copy(this.saNodeL); this.cond.add(this.saNode);
    this.avNode = new THREE.Mesh(new THREE.SphereGeometry(0.28, 20, 16), this.matAV); this.avNode.position.set(-0.9, 6.35, 0.1); this.cond.add(this.avNode);
    // internodal / Bachmann's bundle
    this.cond.add(tube([this.saNodeL, V3(-2.2, 8.4, -0.2), V3(-1.4, 7.4, 0.0), V3(-0.9, 6.5, 0.1)], 0.07, this.matCond, { segments: 20, radial: 8 }));
    this.cond.add(tube([this.saNodeL, V3(-1.2, 9.1, -1.4), V3(0.9, 8.9, -1.9)], 0.07, this.matCond, { segments: 20, radial: 8 }));
    // His bundle + branches + Purkinje (in vent group coordinates = frame L)
    const his = tube([V3(-0.9, 6.3, 0.1), V3(-0.75, 5.6, 0.2)], 0.11, this.matHis, { segments: 6, radial: 8 }); this.condVent.add(his);
    const lbb = [V3(-0.75, 5.6, 0.2), V3(-0.4, 3.4, 0.4), V3(0.1, 1.2, 0.5), V3(0.35, 0.5, 0.4)];
    const rbb = [V3(-0.75, 5.6, 0.2), V3(-1.1, 3.4, 0.9), V3(-1.4, 1.5, 1.2), V3(-1.5, 0.9, 1.25)];
    this.condVent.add(tube(lbb, 0.09, this.matCond, { segments: 24, radial: 8 }));
    this.condVent.add(tube(rbb, 0.09, this.matCond, { segments: 24, radial: 8 }));
    const purk = [
      [V3(0.1, 1.2, 0.5), V3(1.5, 1.6, 0.6), V3(2.4, 3.2, 0.2), V3(2.6, 4.8, -0.3)],
      [V3(0.1, 1.2, 0.5), V3(0.6, 1.4, -1.0), V3(1.4, 3.0, -1.9), V3(1.6, 4.8, -2.0)],
      [V3(0.35, 0.5, 0.4), V3(-0.4, 1.4, -1.6), V3(-1.0, 3.2, -2.1), V3(-1.2, 4.9, -1.9)],
      [V3(-1.4, 1.5, 1.2), V3(-2.6, 1.9, 1.7), V3(-3.6, 3.4, 1.4), V3(-3.8, 4.8, 0.9)],
      [V3(-1.4, 1.5, 1.2), V3(-2.4, 2.4, 2.4), V3(-3.0, 4.2, 2.4), V3(-2.6, 5.4, 2.3)],
    ];
    for (const p of purk) this.condVent.add(tube(p, 0.06, this.matCond, { segments: 20, radial: 6 }));

    // remember the whole-heart bounds for camera fitting
    this.root.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(this.root);
    this.center = box.getCenter(new THREE.Vector3());
    this.radius = box.getBoundingSphere(new THREE.Sphere()).radius;
  }

  resetView() {
    const dist = this.radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.12;
    this.controls.target.copy(this.center);
    this.camera.position.set(this.center.x + dist * 0.10, this.center.y + dist * 0.05, this.center.z + dist);
    this.camera.lookAt(this.center);
    this.controls.update();
  }
  setAutoRotate(on) { this.controls.autoRotate = !!on; }
  setXray(on) {
    this.xray = !!on;
    for (const m of this.opaqueMats) {
      m.transparent = on; m.opacity = on ? (m === this.matVent || m === this.matAtria ? 0.28 : 0.16) : 1.0;
      m.depthWrite = !on; m.needsUpdate = true;
    }
    this.cond.visible = on; this.condVent.visible = on;
  }
  _resize() {
    const el = this.canvas.parentElement; if (!el) return;
    const w = el.clientWidth, h = el.clientHeight; if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  /** Apply a cycle state (from CardiacCycle.stateAt) to the model. */
  setState(s) { this.state = s; }

  render() {
    const s = this.state || { a: 0, v: 0, eject: 0, pulse: 0, atria: { front: 0, plateau: 0 }, vent: { front: 0, plateau: 0 }, sa: 0, av: 0, his: 0 };
    // ---- ventricles: radial + longitudinal shortening about the apex, slight twist
    const v = s.v;
    this.vent.scale.set(1 - 0.13 * v, 1 - 0.085 * v, 1 - 0.13 * v);
    this.vent.rotation.y = 0.10 * v;
    this.rv.scale.set(2.3 * (1 - 0.05 * v), 2.9, 1.6 * (1 - 0.06 * v));
    // ---- atria: contract with the atrial kick, fill (reservoir) while the ventricles contract; the AV plane descends
    const aS = 1 - 0.13 * s.a + 0.05 * v;
    this.atria.scale.set(aS, 1 - 0.10 * s.a + 0.03 * v, aS);
    this.atria.position.set(this.atriaPivot.x, this.atriaPivot.y - 0.35 * v, this.atriaPivot.z);
    // ---- arterial pulse
    const pulse = s.pulse;
    this.aorta.scale.setScalar(1 + 0.045 * pulse); this.aortaRoot.scale.set(1.35, 1.2, 1.35).multiplyScalar(1 + 0.04 * pulse);
    this.pa.scale.setScalar(1 + 0.05 * pulse); this.paRoot.scale.set(1.05, 0.9, 1.05).multiplyScalar(1 + 0.05 * pulse);
    this.matAorta.emissiveIntensity = 0.22 * s.eject; this.matPA.emissiveIntensity = 0.18 * s.eject;

    // ---- depolarization wavefronts (world space)
    this.root.updateWorldMatrix(true, false);
    // ventricles: apex -> base
    const uV = this.matVent.userData.u;
    this.vent.getWorldPosition(uV.uOrigin.value);
    this.vent.getWorldQuaternion(this._tmpQ);
    uV.uDir.value.set(0, 1, 0).applyQuaternion(this._tmpQ).normalize();
    const lenV = this.ventLen * this.vent.scale.y;
    const fv = s.vent.front, pv = s.vent.plateau;
    uV.uFrontPos.value = pv > 0 ? (-0.3 + fv * (lenV + 0.6)) : -1e3;
    uV.uFrontWidth.value = 0.16 * lenV;
    uV.uPlateau.value = pv;
    uV.uFrontGlow.value = (fv > 0 && fv < 1) ? pv : 0;
    // during the T wave the plateau colour shifts cyan -> violet (repolarization)
    const repol = pv < 1 ? (1 - pv) : 0;
    uV.uGlowColor.value.copy(this._cyan).lerp(this._violet, Math.min(1, repol * 1.4));
    // atria: SA node -> far LA
    const uA = this.matAtria.userData.u;
    this.atria.updateWorldMatrix(true, false);
    const o = this._tmpV.copy(this.saNodeL).sub(this.atriaPivot); this.atria.localToWorld(o);
    const far = this._tmpV2.copy(this.laFarL).sub(this.atriaPivot); this.atria.localToWorld(far);
    uA.uOrigin.value.copy(o);
    const lenA = far.sub(o).length(); uA.uDir.value.copy(far).normalize();
    const fa = s.atria.front, pa = s.atria.plateau;
    uA.uFrontPos.value = pa > 0 ? (-0.4 + fa * (lenA + 0.8)) : -1e3;
    uA.uFrontWidth.value = 0.20 * lenA;
    uA.uPlateau.value = pa;
    uA.uFrontGlow.value = (fa > 0 && fa < 1) ? pa : 0;
    uA.uGlowColor.value.copy(this._amber);
    // conduction system glows
    this.matSA.emissiveIntensity = 0.15 + 3.0 * s.sa;
    this.matAV.emissiveIntensity = 0.15 + 2.5 * s.av;
    this.matHis.emissiveIntensity = 0.15 + 3.0 * s.his;
    this.matCond.emissiveIntensity = 0.12 + 2.2 * Math.max(s.his, (fv > 0 && fv < 0.6) ? pv : 0) + 0.8 * s.av * 0.3;

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
