/* Browser viewer for the drone simulation. The server streams poses (SSE, ~30 Hz); this draws them.
 * Sim coordinates are z-up; everything lives in `world`, rotated so three.js (y-up) shows it upright. */
(() => {
const $ = id => document.getElementById(id);
const RUN = new URLSearchParams(location.search).get('run');          // set: watching a training worker, not a demo
const EVENTS = RUN ? `/train/${RUN}/events` : '/events';
const WORLD = RUN ? `/train/${RUN}/world` : '/api/world';
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.78;

const scene = new THREE.Scene();
const SKY_TOP = 0x5d9be0, SKY_HORIZON = 0xcfe4f5;
scene.fog = new THREE.Fog(0xbcd3e8, 45, 190);
(() => {                                    // sky gradient as the background
  const c = document.createElement('canvas'); c.width = 2; c.height = 256;
  const g = c.getContext('2d'), grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, '#4f8fdc'); grad.addColorStop(.55, '#a9cdee'); grad.addColorStop(1, '#e3eef8');
  g.fillStyle = grad; g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c); t.encoding = THREE.sRGBEncoding; scene.background = t;
})();

const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 400);
camera.position.set(5, 4.5, 5);
const controls = new THREE.OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495; controls.minDistance = 1.2; controls.maxDistance = 60;
controls.target.set(0, 1, 0);

scene.add(new THREE.HemisphereLight(0xcfe0f5, 0x4a5a3a, 0.55));
const sun = new THREE.DirectionalLight(0xfff0d6, 1.55);
sun.position.set(14, 22, 9); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, {left: -16, right: 16, top: 16, bottom: -16, near: 1, far: 70});
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03;
scene.add(sun, sun.target);

const world = new THREE.Group(); world.rotation.x = -Math.PI / 2; scene.add(world);   // sim z-up -> y-up
const scenery = new THREE.Group(); world.add(scenery);          // the flat-level backdrop (hidden when terrain is shown)
const terrainGroup = new THREE.Group(); world.add(terrainGroup);   // hills, trees and rocks of levels 4-5

/* ---------- deterministic random so the scenery is identical on every load ---------- */
function rng(seed) { let s = seed >>> 0; return () => ((s = Math.imul(s ^ (s >>> 15), 2246822507) + 0x9e3779b9 >>> 0) / 4294967296); }
const R = rng(7);

/* ---------- ground ---------- */
function grassTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d'), r = rng(3);
  g.fillStyle = '#4f7f35'; g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 9000; i++) {
    const x = r() * 512, y = r() * 512, l = 40 + r() * 40;
    g.fillStyle = `hsl(${84 + r() * 26},${38 + r() * 22}%,${22 + r() * 20}%)`;
    g.globalAlpha = .35; g.fillRect(x, y, 1 + r() * 2, 1 + r() * 2 + l * .02);
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(60, 60);
  t.anisotropy = 8; t.encoding = THREE.sRGBEncoding; return t;
}
const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600),
  new THREE.MeshStandardMaterial({map: grassTexture(), roughness: 1}));
ground.receiveShadow = true; scenery.add(ground);

/* a lighter mowed arena, with corner posts, so the flying area reads clearly */
const arena = new THREE.Mesh(new THREE.PlaneGeometry(16, 16),
  new THREE.MeshStandardMaterial({color: 0x6e9a48, roughness: 1, transparent: true, opacity: .5}));
arena.position.z = 0.01; arena.receiveShadow = true; scenery.add(arena);
for (const [x, y] of [[-8, -8], [-8, 8], [8, -8], [8, 8]]) {
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 2, 10),
    new THREE.MeshStandardMaterial({color: 0xf2b705, roughness: .5}));
  post.rotation.x = Math.PI / 2; post.position.set(x, y, 1); post.castShadow = true; scenery.add(post);
}

/* ---------- scenery: forest, rocks, hills, buildings (visual only) ---------- */
const mats = {
  trunk: new THREE.MeshStandardMaterial({color: 0x5a3d26, roughness: 1}),
  pine: [0x1a4e2d, 0x1e5833, 0x246238].map(c => new THREE.MeshStandardMaterial({color: c, roughness: .95, flatShading: true})),
  leaf: [0x2f7430, 0x3a8434, 0x479038, 0x52963f].map(c => new THREE.MeshStandardMaterial({color: c, roughness: .95, flatShading: true})),
  rock: new THREE.MeshStandardMaterial({color: 0x77736c, roughness: .95, flatShading: true}),
  wall: [0xd8d2c6, 0xc9c4ba, 0xbdb7ab, 0xd9cdb8].map(c => new THREE.MeshStandardMaterial({color: c, roughness: .9})),
  roof: new THREE.MeshStandardMaterial({color: 0x8a4b3a, roughness: .9}),
  glass: new THREE.MeshStandardMaterial({color: 0x2c3a4a, roughness: .25, metalness: .4}),
};
function cylZ(rt, rb, h, seg, mat) {          // cylinder standing along sim +z
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, seg), mat);
  m.rotation.x = Math.PI / 2; m.castShadow = m.receiveShadow = true; return m;
}
function coneZ(r, h, seg, mat) { return cylZ(0, r, h, seg, mat); }
function tree(x, y, kind, s, z0 = 0, target = scenery) {
  const g = new THREE.Group();
  if (kind === 'pine') {
    const th = 1.0 * s; const t = cylZ(.13 * s, .18 * s, th, 8, mats.trunk); t.position.z = th / 2; g.add(t);
    for (let i = 0; i < 4; i++) {
      const c = coneZ((1.7 - i * .33) * s, 1.9 * s, 9, mats.pine[i % 3]); c.position.z = (1.2 + i * 1.0) * s; g.add(c);
    }
  } else {
    const th = 2.2 * s; const t = cylZ(.14 * s, .22 * s, th, 8, mats.trunk); t.position.z = th / 2; g.add(t);
    for (let i = 0; i < 5; i++) {
      const b = new THREE.Mesh(new THREE.IcosahedronGeometry((1.1 + R() * .5) * s, 1), mats.leaf[i % 4]);
      b.position.set((R() - .5) * 1.6 * s, (R() - .5) * 1.6 * s, th + (0.2 + R() * 1.1) * s);
      b.castShadow = b.receiveShadow = true; g.add(b);
    }
  }
  g.position.set(x, y, 0); g.rotation.y = R() * 6.28; target.add(g);
}
function rock(x, y, r, z0 = r * .3, target = scenery) {
  const geo = new THREE.IcosahedronGeometry(r, 1), pos = geo.attributes.position, rr = rng((x * 73 + y * 31) | 0);
  for (let i = 0; i < pos.count; i++) { const k = .78 + rr() * .4; pos.setXYZ(i, pos.getX(i) * k, pos.getY(i) * k, pos.getZ(i) * k * .75); }
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, mats.rock); m.position.set(x, y, z0); m.castShadow = m.receiveShadow = true; target.add(m);
}
function building(x, y, w, d, h) { const target = scenery;
  const g = new THREE.Group(), body = new THREE.Mesh(new THREE.BoxGeometry(w, d, h), mats.wall[(R() * 4) | 0]);
  body.position.z = h / 2; body.castShadow = body.receiveShadow = true; g.add(body);
  const roof = new THREE.Mesh(new THREE.BoxGeometry(w + .4, d + .4, .3), mats.roof); roof.position.z = h + .15; roof.castShadow = true; g.add(roof);
  for (let f = 1; f * 2.2 < h; f++) for (let i = 0; i < Math.floor(w / 1.6); i++) {
    const wd = new THREE.Mesh(new THREE.BoxGeometry(.7, .05, .9), mats.glass);
    wd.position.set(-w / 2 + .9 + i * 1.6, -d / 2 - .02, f * 2.2 - .3); g.add(wd);
  }
  g.position.set(x, y, 0); g.rotation.z = R() * 6.28; target.add(g);
}
function hill(x, y, r, h, col) { const target = scenery;
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshStandardMaterial({color: col, roughness: 1, flatShading: true}));
  m.rotation.x = Math.PI / 2; m.scale.set(1, h / r, 1); m.position.set(x, y, 0); m.receiveShadow = true; target.add(m);
}
(() => {
  for (let k = 0; k < 70; k++) {              // forest ring outside the arena
    const a = R() * 6.283, d = 11 + R() * 40, x = Math.cos(a) * d, y = Math.sin(a) * d;
    if (Math.abs(x) < 9.5 && Math.abs(y) < 9.5) continue;
    tree(x, y, R() < .55 ? 'pine' : 'oak', .8 + R() * .9);
  }
  for (let k = 0; k < 26; k++) {
    const a = R() * 6.283, d = 10 + R() * 22, x = Math.cos(a) * d, y = Math.sin(a) * d;
    if (Math.abs(x) > 9.5 || Math.abs(y) > 9.5) rock(x, y, .3 + R() * .7);
  }
  for (let k = 0; k < 5; k++) { const a = 6.283 * (k + .5) / 5 + .3; building(Math.cos(a) * 30, Math.sin(a) * 30, 3.5 + R() * 3, 3.5 + R() * 3, 4 + R() * 7); }
  hill(-70, 40, 60, 22, 0x4f7a3f); hill(60, 80, 70, 30, 0x58844a); hill(90, -50, 55, 20, 0x466f38);
  hill(-40, -110, 90, 42, 0x6f7f78); hill(20, 140, 100, 50, 0x7b8a86);
})();

/* ---------- the F450-style quadcopter ---------- */
function makeDrone(opts = {}) {
  const g = new THREE.Group(), props = [];
  const dark = new THREE.MeshStandardMaterial({color: 0x23262b, roughness: .55, metalness: .3});
  const plate = new THREE.MeshStandardMaterial({color: 0x15171a, roughness: .5, metalness: .2});
  const front = new THREE.MeshStandardMaterial({color: opts.front ?? 0xd8342c, roughness: .5});
  const back = new THREE.MeshStandardMaterial({color: opts.back ?? 0xeeeeea, roughness: .5});
  const metal = new THREE.MeshStandardMaterial({color: 0x9aa0a8, roughness: .35, metalness: .8});
  const bat = new THREE.MeshStandardMaterial({color: opts.battery ?? 0x2455b8, roughness: .6});
  const add = (geo, mat, x, y, z) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = true; g.add(m); return m; };
  add(new THREE.BoxGeometry(.115, .115, .012), plate, 0, 0, .0);
  add(new THREE.BoxGeometry(.1, .1, .01), plate, 0, 0, .05);
  for (const [x, y] of [[.04, .04], [-.04, .04], [.04, -.04], [-.04, -.04]]) add(new THREE.CylinderGeometry(.006, .006, .05, 8).rotateX(Math.PI / 2), metal, x, y, .025);
  add(new THREE.BoxGeometry(.13, .042, .03), bat, -.01, 0, -.03);
  add(new THREE.BoxGeometry(.05, .05, .02), new THREE.MeshStandardMaterial({color: 0x1d6b3a, roughness: .6}), .0, 0, .065);   // flight controller
  add(new THREE.CylinderGeometry(.012, .012, .008, 12).rotateX(Math.PI / 2), metal, -.03, 0, .076);              // GPS puck
  const cam = add(new THREE.BoxGeometry(.04, .05, .035), dark, .075, 0, -.012);                                 // depth camera
  add(new THREE.CylinderGeometry(.009, .009, .01, 12).rotateZ(Math.PI / 2), new THREE.MeshStandardMaterial({color: 0x0a0a14, roughness: .1, metalness: .6}), .097, .0, -.012);
  const L = .225;                                                                                              // motor-to-centre distance
  [[1, 1], [-1, 1], [-1, -1], [1, -1]].forEach(([sx, sy], i) => {
    const a = Math.atan2(sy, sx), isFront = sx > 0;
    const arm = new THREE.Mesh(new THREE.BoxGeometry(L, .03, .014), isFront ? front : back);
    arm.position.set(sx * L / 2 * .7071, sy * L / 2 * .7071, 0); arm.rotation.z = a; arm.castShadow = true; g.add(arm);
    const mx = sx * L * .7071, my = sy * L * .7071;
    add(new THREE.CylinderGeometry(.018, .018, .03, 16).rotateX(Math.PI / 2), dark, mx, my, .018);
    add(new THREE.CylinderGeometry(.004, .004, .02, 8).rotateX(Math.PI / 2), metal, mx, my, .036);
    const prop = new THREE.Group(); prop.position.set(mx, my, .046);
    const bm = new THREE.MeshStandardMaterial({color: isFront ? 0xf2f2f2 : 0x2a2a2e, roughness: .4});
    for (const s of [1, -1]) { const b = new THREE.Mesh(new THREE.BoxGeometry(.12, .014, .002), bm); b.position.x = s * .06; b.rotation.x = s * .12; b.castShadow = true; prop.add(b); }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(.127, 28), new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide}));
    prop.add(disc); prop.userData = {dir: i % 2 ? 1 : -1, disc}; g.add(prop); props.push(prop);
    const leg = add(new THREE.CylinderGeometry(.006, .006, .06, 6).rotateX(Math.PI / 2), dark, mx * .55, my * .55, -.02);
    add(new THREE.BoxGeometry(.05, .008, .008), dark, mx * .55, my * .55, -.052).rotation.z = a + Math.PI / 2;
    const led = new THREE.Mesh(new THREE.SphereGeometry(.009, 8, 6), new THREE.MeshBasicMaterial({color: isFront ? 0xff3030 : 0x40ff70}));
    led.position.set(mx, my, .0); g.add(led);
  });
  g.userData.props = props; g.scale.setScalar(opts.scale ?? 1);
  return g;
}

/* ---------- people ---------- */
function makeWalker(shirt, skin = 0xe0b48a, pants = 0x2f3b52) {
  const g = new THREE.Group();
  const M = c => new THREE.MeshStandardMaterial({color: c, roughness: .8});
  const part = (geo, mat, x, y, z, parent = g) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = true; parent.add(m); return m; };
  part(new THREE.CapsuleGeometry(.17, .34, 4, 10).rotateX(Math.PI / 2), M(shirt), 0, 0, 1.15);
  part(new THREE.SphereGeometry(.11, 14, 12), M(skin), 0, 0, 1.58);
  part(new THREE.SphereGeometry(.115, 14, 12, 0, 6.283, 0, 1.4), M(0x2b2118), 0, 0, 1.6);
  const limb = (x, z0, len, r, color) => {
    const p = new THREE.Group(); p.position.set(0, x, z0);
    const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 4, 8).rotateX(Math.PI / 2), M(color));
    m.position.z = -len / 2 - r * .5; m.castShadow = true; p.add(m); g.add(p); return p;
  };
  g.userData.legs = [limb(-.09, .8, .5, .075, pants), limb(.09, .8, .5, .075, pants)];
  g.userData.arms = [limb(-.23, 1.4, .42, .055, shirt), limb(.23, 1.4, .42, .055, shirt)];
  g.userData.phase = Math.random() * 6; g.userData.yaw = 0;
  return g;
}
function animateWalker(w, vx, vy, dt) {
  const sp = Math.hypot(vx, vy), u = w.userData;
  if (sp > .08) { const want = Math.atan2(vy, vx); let d = want - u.yaw; d = Math.atan2(Math.sin(d), Math.cos(d)); u.yaw += d * Math.min(1, dt * 8); }
  w.rotation.z = u.yaw;
  u.phase += dt * (3 + sp * 4.5);
  const sw = Math.min(sp, 1.4) * .55 * Math.sin(u.phase), still = sp < .08 ? 0 : 1;
  u.legs[0].rotation.y = sw * still; u.legs[1].rotation.y = -sw * still;
  u.arms[0].rotation.y = -sw * .8 * still; u.arms[1].rotation.y = sw * .8 * still;
}

function makeRock(id, r) {
  const geo = new THREE.IcosahedronGeometry(r, 1), pos = geo.attributes.position, rr = rng(id * 977 + 11);
  for (let i = 0; i < pos.count; i++) { const k = .8 + rr() * .3; pos.setXYZ(i, pos.getX(i) * k, pos.getY(i) * k, pos.getZ(i) * k); }
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color: 0x6f695f, roughness: .95, flatShading: true}));
  m.castShadow = m.receiveShadow = true; m.userData.spin = new THREE.Vector3(rr() - .5, rr() - .5, rr() - .5).multiplyScalar(1.4); return m;
}

/* ---------- live objects ---------- */
const dyn = new THREE.Group(); world.add(dyn);
const drone = makeDrone({scale: 1.6}); world.add(drone);
const personFigure = makeWalker(0x2a9d5c); world.add(personFigure);
const ring = new THREE.Mesh(new THREE.RingGeometry(.45, .55, 40), new THREE.MeshBasicMaterial({color: 0x38c172, transparent: true, opacity: .7, side: THREE.DoubleSide}));
ring.position.z = .03; world.add(ring);
const droneShadowBlob = new THREE.Mesh(new THREE.CircleGeometry(.3, 24), new THREE.MeshBasicMaterial({color: 0, transparent: true, opacity: .18, depthWrite: false}));
droneShadowBlob.position.z = .02; world.add(droneShadowBlob);
const SHIRTS = [0xe8832a, 0xb04a9a, 0xe6d23a, 0x3a96c8];
const objs = new Map();                       // id/group -> three object
let trailPts = [], trailOn = true;
const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: 0x4c8dff, transparent: true, opacity: .8}));
trail.frustumCulled = false; world.add(trail);

function clearDynamic() { for (const o of objs.values()) dyn.remove(o); objs.clear(); trailPts = []; trail.geometry.setFromPoints([]); }

/* ---------- terrain, trees and rocks of levels 4-5 (fetched once per episode) ---------- */
function buildWorld(w) {
  while (terrainGroup.children.length) terrainGroup.remove(terrainGroup.children[0]);
  scenery.visible = !w.terrain;
  scene.fog.far = w.terrain ? 260 : 190;
  if (!w.terrain) return;
  const {half, n, z} = w.terrain, geo = new THREE.PlaneGeometry(2 * half, 2 * half, n - 1, n - 1), pos = geo.attributes.position;
  for (let iy = 0; iy < n; iy++) for (let ix = 0; ix < n; ix++) pos.setZ(iy * n + ix, z[(n - 1 - iy) * n + ix]);
  geo.computeVertexNormals();
  const nrm = geo.attributes.normal, col = new Float32Array(pos.count * 3), c = new THREE.Color(), r = rng(5);
  for (let i = 0; i < pos.count; i++) {
    const h = pos.getZ(i), slope = 1 - nrm.getZ(i), jit = (r() - .5) * .05;
    c.setRGB(.27 + jit, .46 + jit, .2 + jit);                                        // grass
    if (slope > .12) c.lerp(new THREE.Color(.42, .36, .27), Math.min((slope - .12) * 4, 1));   // dirt on slopes
    if (h > 9) c.lerp(new THREE.Color(.46, .46, .45), Math.min((h - 9) / 6, 1));               // rock
    if (h > 22) c.lerp(new THREE.Color(.95, .96, .98), Math.min((h - 22) / 6, 1));            // snow
    col.set([c.r, c.g, c.b], i * 3);
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({vertexColors: true, roughness: 1}));
  mesh.receiveShadow = true; terrainGroup.add(mesh);
  const groups = new Map();
  for (const o of w.statics) { if (!groups.has(o.g)) groups.set(o.g, []); groups.get(o.g).push(o); }
  for (const [g, list] of groups) {
    if (list[0].k === 'rock') { const o = list[0]; rock(o.p[0], o.p[1], o.r, o.p[2], terrainGroup); }
    else { const base = list[0]; tree(base.p[0], base.p[1], g % 3 ? 'pine' : 'oak', g % 3 ? .75 : 1.0, base.p[2] - .45, terrainGroup); }
  }
}
function loadWorld() { fetch(WORLD).then(r => r.json()).then(buildWorld).catch(() => {}); }

/* ---------- state buffer + interpolation ---------- */
const buf = [];
let episode = -1, lastFrameT = performance.now();
const INTERP_MS = 90;
const lerp = (a, b, t) => a + (b - a) * t;
const v3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

function pick(now) {
  const t = now - INTERP_MS;
  if (!buf.length) return null;
  if (buf.length === 1 || t <= buf[0].rt) return {a: buf[0], b: buf[0], k: 0};
  for (let i = buf.length - 1; i > 0; i--) if (buf[i - 1].rt <= t) {
    const a = buf[i - 1], b = buf[i];
    return {a, b, k: t >= b.rt ? 1 : (t - a.rt) / (b.rt - a.rt)};
  }
  return {a: buf[0], b: buf[0], k: 0};
}

function applyState(a, b, k, dt) {
  const qa = new THREE.Quaternion(...a.drone.q), qb = new THREE.Quaternion(...b.drone.q);
  const dp = v3(a.drone.p, b.drone.p, k);
  drone.position.set(...dp); drone.quaternion.copy(qa.slerp(qb, k));
  const speed = Math.hypot(...a.drone.v), rpm = 50 + Math.min(speed, 2.5) * 12;
  for (const p of drone.userData.props) { p.rotation.z += p.userData.dir * rpm * dt; p.userData.disc.material.opacity = .13; }
  const pp = [lerp(a.person.p[0], b.person.p[0], k), lerp(a.person.p[1], b.person.p[1], k)];
  const pg = lerp(a.person.gz || 0, b.person.gz || 0, k), dg = lerp(a.drone.gz || 0, b.drone.gz || 0, k);
  personFigure.position.set(pp[0], pp[1], pg);
  animateWalker(personFigure, a.person.v[0], a.person.v[1], dt);
  ring.position.set(pp[0], pp[1], pg + .03);
  droneShadowBlob.position.set(dp[0], dp[1], dg + .02); const h = Math.max(dp[2] - dg, .05);
  droneShadowBlob.scale.setScalar(1 + h * .25); droneShadowBlob.material.opacity = Math.max(.04, .2 - h * .05);

  const seen = new Set(), bmap = new Map(b.obs.map(o => [o.id, o]));
  for (const oa of a.obs) {
    const ob = bmap.get(oa.id) || oa;
    if (!oa.a && !ob.a) continue;
    const pos = v3(oa.p, ob.p, k);
    if (oa.k === 'pedestrian') {
      if (Math.abs(oa.p[2] - 0.3) > 1e-3) continue;               // one figure per group, driven by its lowest sphere
      const key = 'ped' + oa.g; seen.add(key);
      let w = objs.get(key); if (!w) { w = makeWalker(SHIRTS[oa.g % 4]); dyn.add(w); objs.set(key, w); }
      w.position.set(pos[0], pos[1], pos[2] - 0.3); animateWalker(w, oa.v[0], oa.v[1], dt); continue;
    }
    const key = 'o' + oa.id; seen.add(key);
    let m = objs.get(key);
    if (!m) {
      m = oa.k === 'rival' ? makeDrone({front: 0xff5a1f, back: 0x2b2b2f, battery: 0xaa2222, scale: 1.4}) : makeRock(oa.id, oa.r);
      dyn.add(m); objs.set(key, m);
    }
    m.position.set(pos[0], pos[1], pos[2]);
    if (oa.k === 'rival') {
      const yaw = Math.atan2(oa.v[1], oa.v[0]); m.rotation.set(0, 0, yaw);
      m.rotateY(Math.min(Math.hypot(...oa.v) * .08, .3));
      for (const p of m.userData.props) { p.rotation.z += p.userData.dir * 70 * dt; p.userData.disc.material.opacity = .13; }
    } else { m.rotation.x += m.userData.spin.x * dt; m.rotation.y += m.userData.spin.y * dt; m.rotation.z += m.userData.spin.z * dt; }
  }
  for (const [key, o] of objs) if (!seen.has(key)) { dyn.remove(o); objs.delete(key); }

  if (trailOn) {
    if (!trailPts.length || trailPts[trailPts.length - 1].distanceTo(drone.position) > .05) trailPts.push(drone.position.clone());
    if (trailPts.length > 160) trailPts.shift();
    trail.geometry.setFromPoints(trailPts);
  } else if (trailPts.length) { trailPts = []; trail.geometry.setFromPoints([]); }
}

/* ---------- camera that follows without taking over ---------- */
let follow = true;
const focus = new THREE.Vector3(0, 1, 0); let focusInit = false;
function followCamera(dt) {
  const d = drone.position, p = personFigure.position;
  const sx = (d.x + p.x) / 2, sy = (d.y + p.y) / 2, sz = (d.z + p.z) / 2 + 0.3;                   // midpoint in sim coords
  const tgt = new THREE.Vector3(sx, sz, -sy);                                               // -> three.js (x, z, -y)
  if (!focusInit) { focus.copy(tgt); focusInit = true; controls.target.copy(tgt); camera.position.copy(tgt).add(new THREE.Vector3(3.6, 2.6, 3.6)); }
  const delta = tgt.clone().sub(focus).multiplyScalar(Math.min(1, dt * 3.2)); focus.add(delta);
  if (follow) { controls.target.add(delta); camera.position.add(delta); }
  sun.position.set(focus.x + 14, 22, focus.z + 9); sun.target.position.copy(focus);
}

/* ---------- UI ---------- */
const COL = {'FOLLOW TARGET': ['var(--ok)', '#38c17233'], AVOIDING: ['var(--warn)', '#ff9f1a33'], COLLISION: ['var(--bad)', '#ef444433']};
let lastStatus = '', windAngle = 0;
function setHud(s) {
  if (s.status !== lastStatus) {
    lastStatus = s.status; const [c, halo] = COL[s.status] || ['var(--mut)', '#fff2'];
    $('dot').style.background = c; $('dot').style.boxShadow = `0 0 0 4px ${halo}`; $('label').textContent = s.status;
    $('flash').style.opacity = s.status === 'COLLISION' ? .55 : 0;
  }
  const w = s.wind || [0, 0, 0], ws = Math.hypot(w[0], w[1]);
  $('wtxt').textContent = ws < .3 ? 'calm' : ws.toFixed(1) + ' m/s';
  windAngle = Math.atan2(w[0], w[1]); $('wind').style.opacity = ws < .3 ? .55 : 1;
  $('meta').textContent = RUN ? `${s.policy} · level ${s.difficulty} · ${s.episodes} episodes, ${s.collisions} crashes`
    : `${s.policy} policy · level ${s.difficulty} · ${s.episodes} episodes · ${s.collisions} collisions`;
  document.querySelectorAll('#pol button').forEach(b => b.classList.toggle('on', b.dataset.pol === s.policy));
  if (document.activeElement !== $('diff')) $('diff').value = s.difficulty;
}
const post = p => fetch('/api/' + p, {method: 'POST'}).catch(() => {});
document.querySelectorAll('[data-act]').forEach(b => b.onclick = () => post(b.dataset.act));
document.querySelectorAll('[data-pol]').forEach(b => b.onclick = () => post('policy/' + b.dataset.pol));
$('diff').onchange = e => post('difficulty/' + e.target.value);
$('follow').onclick = e => { follow = !follow; e.target.classList.toggle('on', follow); };
$('trail').onclick = e => { trailOn = !trailOn; e.target.classList.toggle('on', trailOn); };
addEventListener('keydown', e => {
  if (RUN) return;
  const k = {o: 'debris', r: 'rival', p: 'pedestrian', n: 'new'}[e.key]; if (k && !e.target.matches('select')) post(k);
  if ({1: 'trained', 2: 'reactive', 3: 'naive'}[e.key]) post('policy/' + {1: 'trained', 2: 'reactive', 3: 'naive'}[e.key]);
  if (e.key === 'c') $('follow').click();
});

function connect() {
  const es = new EventSource(EVENTS);
  es.onmessage = ev => {
    const s = JSON.parse(ev.data); s.rt = performance.now();
    if (s.ep !== episode) { episode = s.ep; clearDynamic(); buf.length = 0; loadWorld(); }
    buf.push(s); while (buf.length > 12) buf.shift(); setHud(s);
  };
  es.onerror = () => { $('label').textContent = 'RECONNECTING'; es.close(); setTimeout(connect, 1500); };
}
connect();

if (RUN) {                                                  // training view: a progress bar and a run picker instead of controls
  $('bar').style.display = 'none'; $('watchbar').style.display = 'flex'; $('hint').style.display = 'none';
  fetch('/api/runs').then(r => r.json()).then(runs => {
    $('runsel').innerHTML = runs.map(r => `<option value="${r.id}"${r.id === RUN ? ' selected' : ''}>${r.title}</option>`).join('');
    $('runsel').onchange = e => { location.search = '?run=' + e.target.value; };
  }).catch(() => {});
  const poll = () => {
    fetch(`/train/${RUN}/progress`).then(r => r.json()).then(d => {
      const p = d.progress || {}, lag = p.updated_at ? Date.now() / 1000 - p.updated_at : null;
      $('pbar').firstElementChild.style.width = (p.pct || 0) + '%';
      $('ptxt').textContent = p.total ? `${p.pct.toFixed(1)}% · ${(p.steps / 1e6).toFixed(1)}M / ${(p.total / 1e6).toFixed(0)}M steps` + (lag > 120 ? ' · no update' : '') : 'waiting for first update';
    }).catch(() => {});
    setTimeout(poll, 3000);
  };
  poll();
}

function resize() {
  const w = innerWidth, h = innerHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

function frame(now) {
  const dt = Math.min((now - lastFrameT) / 1000, .1); lastFrameT = now;
  const pk = pick(now);
  if (pk) { applyState(pk.a, pk.b, pk.k, dt); followCamera(dt); }
  const az = Math.atan2(camera.position.x - controls.target.x, camera.position.z - controls.target.z);   // camera heading
  $('warrow').style.transform = `rotate(${((windAngle + az) * 180 / Math.PI).toFixed(0)}deg)`;
  controls.update(); renderer.render(scene, camera); requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
