import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const W = 1080;
const H = 1920;
const DURATION = 10;
const canvas = document.createElement('canvas');
canvas.width = W;
canvas.height = H;
document.body.appendChild(canvas);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  preserveDrawingBuffer: true,
  powerPreference: 'high-performance'
});
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color('#272823');
scene.fog = new THREE.FogExp2('#272823', 0.024);
const camera = new THREE.PerspectiveCamera(44, W / H, 0.08, 45);

const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, v) => {
  const x = clamp((v - a) / (b - a + 1e-8));
  return x * x * (3 - 2 * x);
};
const easeOut = x => 1 - Math.pow(1 - clamp(x), 3);
const easeInOut = x => x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
const backOut = x => {
  const c = 1.70158;
  return 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2);
};
const hash = n => {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
};

function material(color, roughness = 0.72, metalness = 0) {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}
const MAT = {
  skin: material('#dca17d', 0.68),
  skinLight: material('#e8b08a', 0.64),
  skinShadow: material('#bd795e', 0.76),
  lips: material('#9f5c52', 0.7),
  hair: material('#281d19', 0.92),
  hairLight: material('#3b2921', 0.88),
  eye: material('#fff9e9', 0.3),
  iris: material('#6d4c2e', 0.38),
  pupil: material('#171312', 0.25),
  jacket: material('#6b4932', 0.9),
  jacketLight: material('#805b3d', 0.86),
  jacketDark: material('#4b3327', 0.92),
  shirt: material('#f1ead9', 0.8),
  tie: material('#a62f28', 0.65),
  tieLight: material('#c54132', 0.58),
  shoe: material('#211c19', 0.52),
  beige: material('#b9ab8c', 0.72),
  beigeLight: material('#d2c6a9', 0.66),
  beigeDark: material('#766e5c', 0.8),
  black: material('#171918', 0.55),
  darkPlastic: material('#2a2d2a', 0.6),
  red: material('#a63227', 0.58),
  brass: material('#b58a45', 0.32, 0.6),
  paper: material('#ddd3b8', 0.96),
  whitePaper: material('#eee8d8', 0.96),
  wood: material('#62432e', 0.78),
  woodDark: material('#362820', 0.88),
  teddy: material('#9a6a3d', 0.98),
  teddyLight: material('#c0925d', 0.96)
};

const sphereGeo = new THREE.SphereGeometry(1, 28, 20);
const lowSphereGeo = new THREE.SphereGeometry(1, 18, 12);
const cylinderGeo = new THREE.CylinderGeometry(1, 1, 1, 20);
const boxGeo = new THREE.BoxGeometry(1, 1, 1);

function addMesh(geometry, mat, parent, position, scale, rotation) {
  const m = new THREE.Mesh(geometry, mat);
  if (position) m.position.set(...position);
  if (scale) m.scale.set(...scale);
  if (rotation) m.rotation.set(...rotation);
  m.castShadow = true;
  m.receiveShadow = true;
  (parent || scene).add(m);
  return m;
}
function sphere(parent, pos, scale, mat = MAT.skin, low = false) {
  return addMesh(low ? lowSphereGeo : sphereGeo, mat, parent, pos, scale);
}
function box(parent, pos, scale, mat, radius = 0) {
  const g = radius > 0
    ? new RoundedBoxGeometry(scale[0], scale[1], scale[2], 3, radius)
    : boxGeo;
  const m = addMesh(g, mat, parent, pos, radius > 0 ? null : scale);
  return m;
}
function cyl(parent, pos, scale, mat, rotation = null) {
  return addMesh(cylinderGeo, mat, parent, pos, scale, rotation);
}
function setSegment(mesh, a, b, radius) {
  const delta = new THREE.Vector3().subVectors(b, a);
  const len = Math.max(0.001, delta.length());
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.scale.set(radius, len, radius);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
}
function canvasTexture(c) {
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  return t;
}
function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, rr);
}

// ---------- Procedural material and environment textures ----------
function wallTexture() {
  const c = makeCanvas(512, 512);
  const x = c.getContext('2d');
  x.fillStyle = '#77776a';
  x.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 9000; i++) {
    const v = 85 + Math.floor(hash(i) * 70);
    x.fillStyle = `rgba(${v},${v},${v - 8},${0.018 + hash(i + 4) * 0.025})`;
    x.fillRect(hash(i + 1) * 512, hash(i + 2) * 512, 1 + hash(i + 3) * 3, 1 + hash(i + 5) * 3);
  }
  for (let i = 0; i < 20; i++) {
    x.strokeStyle = `rgba(47,43,36,${0.025 + hash(i) * 0.035})`;
    x.lineWidth = 1;
    x.beginPath();
    let px = hash(i + 10) * 512;
    let py = hash(i + 20) * 512;
    x.moveTo(px, py);
    for (let j = 0; j < 5; j++) {
      px += (hash(i * 20 + j) - 0.5) * 45;
      py += hash(i * 30 + j) * 28;
      x.lineTo(px, py);
    }
    x.stroke();
  }
  const t = canvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(3, 2);
  return t;
}
function woodTexture() {
  const c = makeCanvas(768, 384);
  const x = c.getContext('2d');
  const g = x.createLinearGradient(0, 0, 0, 384);
  g.addColorStop(0, '#765238');
  g.addColorStop(0.45, '#5d3d2b');
  g.addColorStop(1, '#452f24');
  x.fillStyle = g;
  x.fillRect(0, 0, 768, 384);
  for (let i = 0; i < 120; i++) {
    const yy = hash(i) * 384;
    x.strokeStyle = `rgba(${35 + hash(i + 3) * 50},24,17,${0.08 + hash(i + 7) * 0.16})`;
    x.lineWidth = 0.5 + hash(i + 9) * 2;
    x.beginPath();
    x.moveTo(0, yy);
    for (let j = 0; j <= 12; j++) {
      x.lineTo(j * 64, yy + Math.sin(j * 0.9 + i) * (2 + hash(i) * 5));
    }
    x.stroke();
  }
  for (let i = 0; i < 22; i++) {
    x.strokeStyle = 'rgba(225,196,145,0.13)';
    x.lineWidth = 1;
    x.beginPath();
    const px = hash(i + 99) * 768;
    const py = hash(i + 199) * 384;
    x.moveTo(px, py);
    x.lineTo(px + 25 + hash(i) * 100, py + (hash(i + 2) - 0.5) * 10);
    x.stroke();
  }
  return canvasTexture(c);
}
function londonTexture() {
  const c = makeCanvas(900, 900);
  const x = c.getContext('2d');
  const sky = x.createLinearGradient(0, 0, 0, 900);
  sky.addColorStop(0, '#73848b');
  sky.addColorStop(0.55, '#9ba5a3');
  sky.addColorStop(1, '#566568');
  x.fillStyle = sky;
  x.fillRect(0, 0, 900, 900);
  x.fillStyle = 'rgba(236,229,201,0.23)';
  x.beginPath();
  x.arc(680, 190, 95, 0, Math.PI * 2);
  x.fill();
  // Distant London roofline.
  for (let i = 0; i < 18; i++) {
    const bw = 45 + hash(i) * 70;
    const bh = 80 + hash(i + 3) * 150;
    const bx = i * 55 - 30;
    x.fillStyle = i % 2 ? '#465558' : '#526064';
    x.fillRect(bx, 610 - bh, bw, bh + 170);
    x.fillStyle = 'rgba(211,198,142,0.36)';
    for (let wy = 630 - bh; wy < 590; wy += 34) {
      for (let wx = bx + 10; wx < bx + bw - 8; wx += 22) x.fillRect(wx, wy, 7, 12);
    }
  }
  // Big Ben / Elizabeth Tower silhouette.
  x.fillStyle = '#36474a';
  x.fillRect(600, 260, 102, 500);
  x.fillRect(585, 340, 132, 22);
  x.beginPath();
  x.moveTo(592, 260);
  x.lineTo(650, 145);
  x.lineTo(710, 260);
  x.closePath();
  x.fill();
  x.fillRect(646, 100, 9, 70);
  x.fillStyle = '#c6b67e';
  x.beginPath();
  x.arc(651, 315, 36, 0, Math.PI * 2);
  x.fill();
  x.strokeStyle = '#354347';
  x.lineWidth = 5;
  x.beginPath();
  x.arc(651, 315, 30, 0, Math.PI * 2);
  x.stroke();
  x.beginPath();
  x.moveTo(651, 315); x.lineTo(651, 291);
  x.moveTo(651, 315); x.lineTo(670, 323);
  x.stroke();
  // A tiny red double-decker bus.
  x.fillStyle = '#a6332c';
  roundRect(x, 120, 665, 265, 126, 16); x.fill();
  x.fillStyle = '#b9ced0';
  for (let i = 0; i < 5; i++) {
    x.fillRect(140 + i * 45, 684, 32, 34);
    x.fillRect(140 + i * 45, 732, 32, 28);
  }
  x.fillStyle = '#29383b';
  x.beginPath(); x.arc(172, 790, 23, 0, Math.PI * 2); x.fill();
  x.beginPath(); x.arc(333, 790, 23, 0, Math.PI * 2); x.fill();
  // Rain on glass.
  for (let i = 0; i < 145; i++) {
    const px = hash(i + 500) * 900;
    const py = hash(i + 800) * 900;
    x.strokeStyle = `rgba(225,238,235,${0.12 + hash(i) * 0.3})`;
    x.lineWidth = 1 + hash(i + 2) * 2;
    x.beginPath();
    x.moveTo(px, py);
    x.lineTo(px - 5, py + 12 + hash(i + 6) * 40);
    x.stroke();
  }
  const glare = x.createLinearGradient(0, 0, 900, 900);
  glare.addColorStop(0, 'rgba(255,255,255,0.18)');
  glare.addColorStop(0.35, 'rgba(255,255,255,0.015)');
  glare.addColorStop(0.7, 'rgba(255,255,255,0.09)');
  glare.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = glare;
  x.fillRect(0, 0, 900, 900);
  return canvasTexture(c);
}
function labelTexture(text, options = {}) {
  const c = makeCanvas(options.w || 512, options.h || 256);
  const x = c.getContext('2d');
  x.fillStyle = options.bg || '#e9e0c8';
  x.fillRect(0, 0, c.width, c.height);
  if (options.border !== false) {
    x.strokeStyle = options.borderColor || '#665d4b';
    x.lineWidth = 10;
    x.strokeRect(5, 5, c.width - 10, c.height - 10);
  }
  x.fillStyle = options.color || '#38342d';
  x.textAlign = 'center';
  x.textBaseline = 'middle';
  x.font = `${options.weight || 800} ${options.fontSize || 54}px Inter, sans-serif`;
  const lines = text.split('\n');
  lines.forEach((line, i) => x.fillText(line, c.width / 2, c.height / 2 + (i - (lines.length - 1) / 2) * (options.fontSize || 54) * 1.2));
  return canvasTexture(c);
}

// ---------- Lighting ----------
scene.add(new THREE.HemisphereLight('#b7c9cf', '#423126', 1.55));
const key = new THREE.DirectionalLight('#ffe0b0', 3.1);
key.position.set(3.8, 6.5, 4.5);
key.target.position.set(-0.25, 1.35, -0.3);
key.castShadow = true;
key.shadow.mapSize.set(1536, 1536);
key.shadow.camera.left = -4.5;
key.shadow.camera.right = 4.5;
key.shadow.camera.top = 5;
key.shadow.camera.bottom = -2;
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 16;
key.shadow.bias = -0.00025;
key.shadow.normalBias = 0.025;
scene.add(key, key.target);
const windowFill = new THREE.DirectionalLight('#9fc9d5', 1.65);
windowFill.position.set(3, 3.8, -1.5);
windowFill.target.position.set(-0.6, 1.7, 0.2);
scene.add(windowFill, windowFill.target);
const faceFill = new THREE.PointLight('#ffd5ae', 10, 7, 2);
faceFill.position.set(-2.8, 3.3, 3.5);
scene.add(faceFill);

// ---------- Small, messy London office ----------
const office = new THREE.Group();
scene.add(office);
const wallMat = material('#ffffff', 0.96);
wallMat.map = wallTexture();
const floorMat = material('#51483c', 0.92);
const floor = addMesh(new THREE.PlaneGeometry(12, 12), floorMat, office, [0, 0, 0], null, [-Math.PI / 2, 0, 0]);
floor.receiveShadow = true;
floor.castShadow = false;
const backWall = box(office, [0, 3.25, -2.25], [8, 6.5, 0.14], wallMat);
backWall.castShadow = false;
const leftWall = box(office, [-3.85, 3.25, 0.5], [0.14, 6.5, 5.5], wallMat);
leftWall.castShadow = false;
box(office, [0, 0.13, -2.12], [7.8, 0.25, 0.12], MAT.woodDark);
// Floorboards and worn rug.
for (let i = 0; i < 12; i++) {
  const seam = box(office, [-3.4 + i * 0.62, 0.006, 0.2], [0.015, 0.012, 7], MAT.woodDark);
  seam.castShadow = false;
}
const rugMat = material('#5a302b', 0.98);
const rug = addMesh(new THREE.PlaneGeometry(3.4, 2.5), rugMat, office, [-0.35, 0.018, 0.5], null, [-Math.PI / 2, 0, -0.05]);
rug.receiveShadow = true;
for (let i = 0; i < 7; i++) {
  const stripe = box(office, [-1.65 + i * 0.43, 0.025, 1.6], [0.08, 0.015, 0.25], material(i % 2 ? '#9b7651' : '#322f2a', 1));
  stripe.castShadow = false;
}

// Rainy window and chunky painted frame.
const windowGroup = new THREE.Group();
windowGroup.position.set(1.6, 2.9, -2.14);
office.add(windowGroup);
const londonMat = new THREE.MeshBasicMaterial({ map: londonTexture() });
addMesh(new THREE.PlaneGeometry(1.85, 1.85), londonMat, windowGroup, [0, 0, 0.085]);
const frameMat = material('#d4cbb3', 0.82);
box(windowGroup, [-1.0, 0, 0.14], [0.13, 2.12, 0.16], frameMat);
box(windowGroup, [1.0, 0, 0.14], [0.13, 2.12, 0.16], frameMat);
box(windowGroup, [0, 1.0, 0.14], [2.12, 0.13, 0.16], frameMat);
box(windowGroup, [0, -1.0, 0.14], [2.12, 0.13, 0.16], frameMat);
box(windowGroup, [0, 0, 0.15], [0.08, 2.0, 0.12], frameMat);
box(windowGroup, [0, 0.05, 0.15], [2.0, 0.08, 0.12], frameMat);
box(windowGroup, [0, -1.12, 0.25], [2.35, 0.14, 0.48], frameMat);
// A few animated rivulets just in front of the glass.
const rainDrops = [];
const rainMat = new THREE.MeshBasicMaterial({ color: '#d7eeeb', transparent: true, opacity: 0.27 });
for (let i = 0; i < 16; i++) {
  const drop = box(windowGroup, [lerp(-0.88, 0.88, hash(i)), lerp(-0.85, 0.9, hash(i + 30)), 0.105], [0.009, 0.08 + hash(i + 10) * 0.13, 0.006], rainMat);
  drop.castShadow = false;
  drop.userData.startY = drop.position.y;
  drop.userData.speed = 0.08 + hash(i + 70) * 0.18;
  rainDrops.push(drop);
}

// Corkboard, crooked notes, and clock.
const cork = material('#8a643e', 0.98);
box(office, [-2.25, 2.25, -2.08], [1.55, 1.25, 0.12], MAT.woodDark);
box(office, [-2.25, 2.25, -1.99], [1.4, 1.1, 0.05], cork);
const noteData = [
  [-2.55, 2.47, -1.94, -0.11, 'BIG\nIDEA!', '#e6d36c'],
  [-1.98, 2.48, -1.94, 0.08, 'GET\nONLINE', '#d7e2ce'],
  [-2.48, 1.96, -1.94, 0.05, 'CALL\nMUM', '#e9d8bd'],
  [-1.98, 1.98, -1.94, -0.08, 'SEO???', '#d6a69a']
];
for (const [nx, ny, nz, nr, txt, col] of noteData) {
  const nm = new THREE.MeshStandardMaterial({ map: labelTexture(txt, { w: 256, h: 256, bg: col, fontSize: 48, border: false }), roughness: 0.95 });
  const note = addMesh(new THREE.PlaneGeometry(0.45, 0.45), nm, office, [nx, ny, nz]);
  note.rotation.z = nr;
  sphere(office, [nx, ny + 0.18, nz + 0.025], [0.025, 0.025, 0.018], MAT.red, true);
}
const clockGroup = new THREE.Group();
clockGroup.position.set(-0.62, 3.65, -2.03);
office.add(clockGroup);
cyl(clockGroup, [0, 0, 0], [0.37, 0.07, 0.37], MAT.woodDark, [Math.PI / 2, 0, 0]);
cyl(clockGroup, [0, 0, 0.075], [0.32, 0.02, 0.32], MAT.shirt, [Math.PI / 2, 0, 0]);
for (let i = 0; i < 12; i++) {
  const a = i / 12 * Math.PI * 2;
  const tick = box(clockGroup, [Math.sin(a) * 0.265, Math.cos(a) * 0.265, 0.105], [0.016, 0.045, 0.012], MAT.black);
  tick.rotation.z = -a;
}
const hourHand = box(clockGroup, [0.065, 0.055, 0.12], [0.022, 0.18, 0.018], MAT.black);
hourHand.rotation.z = -0.82;
const minuteHand = box(clockGroup, [-0.075, 0.075, 0.125], [0.015, 0.24, 0.015], MAT.red);
minuteHand.rotation.z = 0.72;
sphere(clockGroup, [0, 0, 0.14], [0.035, 0.035, 0.018], MAT.brass, true);

// Battered desk.
const desk = new THREE.Group();
scene.add(desk);
const deskMat = material('#ffffff', 0.74);
deskMat.map = woodTexture();
box(desk, [0, 0.94, 0.28], [3.75, 0.15, 1.65], deskMat, 0.035);
for (const dx of [-1.65, 1.65]) {
  for (const dz of [-0.32, 0.85]) box(desk, [dx, 0.46, dz], [0.16, 0.93, 0.16], MAT.woodDark, 0.025);
}
box(desk, [1.35, 0.56, 0.05], [0.65, 0.65, 0.85], MAT.wood, 0.025);
for (let i = 0; i < 2; i++) {
  box(desk, [1.35, 0.39 + i * 0.34, 0.49], [0.56, 0.27, 0.04], deskMat, 0.015);
  box(desk, [1.35, 0.39 + i * 0.34, 0.525], [0.17, 0.035, 0.035], MAT.brass, 0.01);
}
// Desk scratches and ink ring.
const ringMat = new THREE.MeshBasicMaterial({ color: '#2b1c16', transparent: true, opacity: 0.28, side: THREE.DoubleSide });
const mugRing = addMesh(new THREE.RingGeometry(0.13, 0.145, 32), ringMat, desk, [-0.62, 1.021, 0.55], null, [-Math.PI / 2, 0, 0]);
mugRing.castShadow = false;

// Paper clutter.
const paperGroups = [];
for (let i = 0; i < 8; i++) {
  const p = new THREE.Group();
  p.position.set(-1.25 + hash(i) * 2.5, 1.025 + i * 0.001, -0.15 + hash(i + 20) * 1.1);
  p.rotation.y = (hash(i + 40) - 0.5) * 1.1;
  desk.add(p);
  const pm = i % 2 ? MAT.paper : MAT.whitePaper;
  const sheet = box(p, [0, 0, 0], [0.48 + hash(i + 4) * 0.22, 0.009, 0.62], pm);
  sheet.castShadow = false;
  for (let j = 0; j < 4; j++) {
    const line = box(p, [0, 0.007, -0.18 + j * 0.09], [0.27 + hash(j + i) * 0.14, 0.003, 0.012], MAT.beigeDark);
    line.castShadow = false;
  }
  paperGroups.push(p);
}
// Floppy disks with correctly fitted paper labels.
for (let i = 0; i < 3; i++) {
  const floppy = new THREE.Group();
  floppy.position.set(-1.35 + i * 0.16, 1.06 + i * 0.025, 0.67);
  floppy.rotation.y = -0.2 + i * 0.15;
  desk.add(floppy);
  box(floppy, [0, 0, 0], [0.42, 0.025, 0.42], i === 1 ? MAT.red : MAT.darkPlastic, 0.015);
  box(floppy, [0, 0.018, 0.035], [0.25, 0.008, 0.19], MAT.shirt);
  box(floppy, [0, 0.019, -0.145], [0.18, 0.009, 0.07], MAT.beigeDark);
}

// Anglepoise lamp.
const lamp = new THREE.Group();
lamp.position.set(1.58, 1.04, -0.3);
scene.add(lamp);
cyl(lamp, [0, 0, 0], [0.26, 0.055, 0.26], MAT.red);
const lampA = new THREE.Vector3(0, 0.04, 0);
const lampB = new THREE.Vector3(-0.18, 0.64, 0);
const lampC = new THREE.Vector3(-0.48, 1.0, 0.12);
const lampSeg1 = cyl(lamp, [0, 0, 0], [1, 1, 1], MAT.brass);
const lampSeg2 = cyl(lamp, [0, 0, 0], [1, 1, 1], MAT.brass);
setSegment(lampSeg1, lampA, lampB, 0.025);
setSegment(lampSeg2, lampB, lampC, 0.025);
sphere(lamp, lampB.toArray(), [0.065, 0.065, 0.065], MAT.red, true);
const shade = addMesh(new THREE.ConeGeometry(0.25, 0.34, 24, 1, true), MAT.red, lamp, [-0.5, 0.92, 0.15], null, [0.25, 0, -0.55]);
const bulb = new THREE.PointLight('#ffb45f', 7, 3, 2);
bulb.position.set(-0.55, 0.79, 0.22);
lamp.add(bulb);

// Mug with tea and a tiny spoon.
const mug = new THREE.Group();
mug.position.set(-0.7, 1.18, 0.63);
scene.add(mug);
const mugMat = material('#d7d1bb', 0.4);
cyl(mug, [0, 0, 0], [0.15, 0.27, 0.15], mugMat);
cyl(mug, [0, 0.145, 0], [0.127, 0.012, 0.127], material('#543425', 0.28));
const handle = addMesh(new THREE.TorusGeometry(0.13, 0.035, 10, 24, Math.PI * 1.65), mugMat, mug, [0.16, 0.015, 0], null, [Math.PI / 2, 0, Math.PI / 2]);
const spoon = cyl(mug, [0.035, 0.27, 0], [0.012, 0.34, 0.012], MAT.brass);
spoon.rotation.z = -0.18;
const mugLabel = new THREE.MeshBasicMaterial({ map: labelTexture('BOSS', { w: 256, h: 128, bg: '#d7d1bb', color: '#71372d', fontSize: 60, border: false }) });
addMesh(new THREE.PlaneGeometry(0.18, 0.09), mugLabel, mug, [0, 0, 0.151]);

// Teddy, quietly supervising.
const teddy = new THREE.Group();
teddy.position.set(1.47, 1.28, 0.43);
teddy.rotation.y = -0.25;
scene.add(teddy);
sphere(teddy, [0, 0, 0], [0.19, 0.25, 0.15], MAT.teddy);
sphere(teddy, [0, 0.32, 0.02], [0.22, 0.2, 0.17], MAT.teddy);
sphere(teddy, [-0.18, 0.45, 0], [0.09, 0.09, 0.06], MAT.teddy, true);
sphere(teddy, [0.18, 0.45, 0], [0.09, 0.09, 0.06], MAT.teddy, true);
sphere(teddy, [0, 0.26, 0.16], [0.12, 0.085, 0.06], MAT.teddyLight, true);
sphere(teddy, [-0.075, 0.35, 0.16], [0.022, 0.026, 0.015], MAT.pupil, true);
sphere(teddy, [0.075, 0.35, 0.16], [0.022, 0.026, 0.015], MAT.pupil, true);
sphere(teddy, [0, 0.28, 0.215], [0.025, 0.02, 0.018], MAT.pupil, true);
sphere(teddy, [-0.2, 0.02, 0], [0.09, 0.2, 0.09], MAT.teddy, true).rotation.z = -0.45;
sphere(teddy, [0.2, 0.02, 0], [0.09, 0.2, 0.09], MAT.teddy, true).rotation.z = 0.45;
sphere(teddy, [-0.11, -0.22, 0.04], [0.11, 0.13, 0.12], MAT.teddy, true);
sphere(teddy, [0.11, -0.22, 0.04], [0.11, 0.13, 0.12], MAT.teddy, true);

// ---------- Old CRT, keyboard, mouse, and cables ----------
const computer = new THREE.Group();
computer.position.set(0.58, 1.72, 0.03);
scene.add(computer);
box(computer, [0, 0, -0.08], [1.55, 1.18, 0.82], MAT.beige, 0.11);
box(computer, [0, 0.02, 0.355], [1.38, 0.98, 0.08], MAT.beigeDark, 0.08);
box(computer, [0, 0.02, 0.405], [1.3, 0.9, 0.045], MAT.black, 0.07);
// Curved screen surface.
const screenGeo = new THREE.PlaneGeometry(1.22, 0.81, 28, 20);
const sp = screenGeo.attributes.position;
for (let i = 0; i < sp.count; i++) {
  const sx = sp.getX(i) / 0.61;
  const sy = sp.getY(i) / 0.405;
  sp.setZ(i, 0.035 * (1 - sx * sx) + 0.012 * (1 - sy * sy));
}
screenGeo.computeVertexNormals();
const screenCanvas = makeCanvas(960, 640);
const screenCtx = screenCanvas.getContext('2d');
const screenTexture = canvasTexture(screenCanvas);
const screenMat = new THREE.MeshStandardMaterial({
  map: screenTexture,
  emissiveMap: screenTexture,
  emissive: '#ffffff',
  emissiveIntensity: 0.72,
  roughness: 0.34,
  metalness: 0
});
const screenMesh = addMesh(screenGeo, screenMat, computer, [0, 0.035, 0.445]);
screenMesh.castShadow = false;
// Screen glints and scanline glass.
const glassMat = new THREE.MeshPhysicalMaterial({ color: '#c8e2df', transparent: true, opacity: 0.075, roughness: 0.12, metalness: 0.05, depthWrite: false });
const glass = addMesh(screenGeo.clone(), glassMat, computer, [0, 0.035, 0.456]);
glass.castShadow = false;
// Controls, vents, badge, and sturdy base.
for (let i = 0; i < 7; i++) box(computer, [-0.57 + i * 0.1, -0.515, 0.37], [0.055, 0.018, 0.025], MAT.beigeDark);
cyl(computer, [0.52, -0.49, 0.38], [0.045, 0.025, 0.045], MAT.darkPlastic, [Math.PI / 2, 0, 0]);
sphere(computer, [0.4, -0.49, 0.39], [0.018, 0.018, 0.012], new THREE.MeshBasicMaterial({ color: '#80b75b' }), true);
const crtBadge = new THREE.MeshBasicMaterial({ map: labelTexture('BEAN 486', { w: 384, h: 96, bg: '#b9ab8c', color: '#514b40', fontSize: 42, border: false }) });
addMesh(new THREE.PlaneGeometry(0.35, 0.085), crtBadge, computer, [-0.25, -0.49, 0.408]);
box(computer, [0, -0.65, -0.05], [0.64, 0.16, 0.48], MAT.beige, 0.04);
box(computer, [0, -0.72, 0.02], [1.0, 0.07, 0.68], MAT.beigeDark, 0.035);
// Rear ventilation slots.
for (let i = 0; i < 8; i++) box(computer, [0.785, 0.35 - i * 0.09, -0.17], [0.015, 0.035, 0.35], MAT.beigeDark);

const keyboard = new THREE.Group();
keyboard.position.set(-0.22, 1.09, 0.72);
keyboard.rotation.x = -0.07;
scene.add(keyboard);
box(keyboard, [0, 0, 0], [1.45, 0.12, 0.54], MAT.beige, 0.045);
const keyGeo = new RoundedBoxGeometry(0.09, 0.055, 0.075, 2, 0.012);
const keyMat = MAT.beigeLight;
const keyCount = 60;
const keys = new THREE.InstancedMesh(keyGeo, keyMat, keyCount);
keys.castShadow = true;
keys.receiveShadow = true;
keyboard.add(keys);
const keyDummy = new THREE.Object3D();
let keyIndex = 0;
for (let row = 0; row < 5; row++) {
  for (let col = 0; col < 12; col++) {
    keyDummy.position.set(-0.56 + col * 0.102 + (row % 2) * 0.02, 0.085, -0.18 + row * 0.09);
    keyDummy.rotation.set(0, 0, 0);
    keyDummy.updateMatrix();
    keys.setMatrixAt(keyIndex++, keyDummy.matrix);
  }
}
keys.instanceMatrix.needsUpdate = true;
const spacebar = box(keyboard, [0.05, 0.09, 0.225], [0.58, 0.05, 0.07], MAT.beigeLight, 0.012);
// Mouse and curly-looking cord.
const mouse = new THREE.Group();
mouse.position.set(0.75, 1.11, 0.78);
scene.add(mouse);
sphere(mouse, [0, 0, 0], [0.17, 0.08, 0.23], MAT.beigeLight);
box(mouse, [0, 0.07, 0.07], [0.018, 0.02, 0.16], MAT.beigeDark);
const cableMat = MAT.darkPlastic;
function cable(points, radius = 0.018, parent = scene) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)));
  const m = addMesh(new THREE.TubeGeometry(curve, 48, radius, 8, false), cableMat, parent);
  return m;
}
cable([[0.55, 1.5, -0.45], [0.9, 1.0, -0.6], [0.2, 0.55, -0.5], [-0.6, 0.25, -0.9], [-1.1, 0.08, -0.4]], 0.025);
cable([[0.72, 1.1, 0.58], [0.9, 1.07, 0.43], [0.55, 1.06, 0.3], [0.2, 1.05, 0.45]], 0.012);

// ---------- Mr. Bean-inspired articulated 3D character ----------
const bean = new THREE.Group();
bean.position.set(-0.88, 0, -0.55);
scene.add(bean);
// Chair, trousers, and shoes glimpsed below the desk.
const chair = new THREE.Group();
bean.add(chair);
box(chair, [0, 1.0, -0.32], [0.9, 1.15, 0.16], MAT.jacketDark, 0.12);
box(chair, [0, 0.56, -0.02], [0.86, 0.16, 0.76], MAT.jacketDark, 0.08);
cyl(chair, [0, 0.28, -0.03], [0.06, 0.48, 0.06], MAT.darkPlastic);
for (let i = 0; i < 5; i++) {
  const a = i / 5 * Math.PI * 2;
  const leg = cyl(chair, [0, 0, 0], [1, 1, 1], MAT.darkPlastic);
  setSegment(leg, new THREE.Vector3(0, 0.12, -0.03), new THREE.Vector3(Math.sin(a) * 0.46, 0.06, Math.cos(a) * 0.46), 0.025);
  sphere(chair, [Math.sin(a) * 0.46, 0.05, Math.cos(a) * 0.46], [0.06, 0.06, 0.06], MAT.black, true);
}
sphere(bean, [-0.25, 0.65, 0.0], [0.25, 0.42, 0.24], MAT.jacketDark);
sphere(bean, [0.25, 0.65, 0.0], [0.25, 0.42, 0.24], MAT.jacketDark);
sphere(bean, [-0.28, 0.18, 0.27], [0.2, 0.13, 0.39], MAT.shoe);
sphere(bean, [0.28, 0.18, 0.27], [0.2, 0.13, 0.39], MAT.shoe);

const body = new THREE.Group();
body.position.set(0, 1.35, 0);
bean.add(body);
sphere(body, [0, 0, 0], [0.55, 0.72, 0.31], MAT.jacket);
sphere(body, [-0.43, 0.4, 0], [0.23, 0.25, 0.27], MAT.jacket);
sphere(body, [0.43, 0.4, 0], [0.23, 0.25, 0.27], MAT.jacket);
// Shirt opening, lapels, red tie, pocket, and buttons.
const shirtShape = new THREE.Shape();
shirtShape.moveTo(-0.23, 0.55); shirtShape.lineTo(0.23, 0.55); shirtShape.lineTo(0.1, -0.24); shirtShape.lineTo(-0.1, -0.24); shirtShape.closePath();
const shirtFront = addMesh(new THREE.ShapeGeometry(shirtShape), MAT.shirt, body, [0, 0, 0.309]);
shirtFront.castShadow = false;
function lapel(side) {
  const s = new THREE.Shape();
  s.moveTo(side * 0.24, 0.53);
  s.lineTo(side * 0.43, 0.33);
  s.lineTo(side * 0.17, -0.18);
  s.lineTo(side * 0.06, 0.12);
  s.closePath();
  const m = addMesh(new THREE.ShapeGeometry(s), MAT.jacketLight, body, [0, 0, 0.325]);
  m.castShadow = false;
}
lapel(-1); lapel(1);
box(body, [0, 0.36, 0.345], [0.14, 0.13, 0.035], MAT.tieLight, 0.025).rotation.z = Math.PI / 4;
const tieShape = new THREE.Shape();
tieShape.moveTo(-0.055, 0.29); tieShape.lineTo(0.055, 0.29); tieShape.lineTo(0.09, -0.25); tieShape.lineTo(0, -0.36); tieShape.lineTo(-0.09, -0.25); tieShape.closePath();
addMesh(new THREE.ShapeGeometry(tieShape), MAT.tie, body, [0, 0, 0.348]);
for (let i = 0; i < 3; i++) sphere(body, [0.02, -0.18 - i * 0.16, 0.305], [0.027, 0.027, 0.018], MAT.jacketDark, true);
const pocket = box(body, [-0.3, -0.05, 0.275], [0.22, 0.035, 0.025], MAT.jacketLight);
pocket.rotation.z = -0.08;
// Tiny white pocket square, slightly crooked.
const pocketSquare = box(body, [-0.31, 0.0, 0.28], [0.13, 0.09, 0.018], MAT.shirt);
pocketSquare.rotation.z = 0.12;

const head = new THREE.Group();
head.position.set(0, 1.08, 0.02);
body.add(head);
// Long face, chin, cheeks, and ears.
sphere(head, [0, 0, 0], [0.375, 0.52, 0.325], MAT.skin);
sphere(head, [0, -0.34, 0.035], [0.285, 0.245, 0.275], MAT.skin);
sphere(head, [-0.22, -0.13, 0.22], [0.12, 0.15, 0.09], MAT.skinLight, true);
sphere(head, [0.22, -0.13, 0.22], [0.12, 0.15, 0.09], MAT.skinLight, true);
for (const side of [-1, 1]) {
  const ear = sphere(head, [side * 0.385, -0.01, -0.005], [0.09, 0.155, 0.065], MAT.skin);
  ear.rotation.z = side * -0.14;
  sphere(head, [side * 0.41, -0.01, 0.04], [0.042, 0.09, 0.025], MAT.skinShadow, true);
}
// Nose with a long bridge and unmistakable rounded tip.
const noseBridge = sphere(head, [0, -0.015, 0.31], [0.078, 0.19, 0.11], MAT.skinLight);
noseBridge.rotation.x = -0.12;
sphere(head, [0, -0.105, 0.425], [0.105, 0.082, 0.125], MAT.skinLight);
sphere(head, [-0.055, -0.14, 0.492], [0.022, 0.014, 0.012], MAT.skinShadow, true);
sphere(head, [0.055, -0.14, 0.492], [0.022, 0.014, 0.012], MAT.skinShadow, true);
// Eyes have independently animated gaze and lids.
const eyes = [];
for (const side of [-1, 1]) {
  const eye = new THREE.Group();
  eye.position.set(side * 0.145, 0.105, 0.29);
  head.add(eye);
  sphere(eye, [0, 0, 0], [0.116, 0.102, 0.065], MAT.eye);
  const pupil = new THREE.Group();
  pupil.position.set(0, 0, 0.061);
  eye.add(pupil);
  sphere(pupil, [0, 0, 0], [0.047, 0.052, 0.018], MAT.iris, true);
  sphere(pupil, [0, 0, 0.015], [0.024, 0.03, 0.012], MAT.pupil, true);
  sphere(pupil, [-0.014, 0.018, 0.026], [0.009, 0.011, 0.006], new THREE.MeshBasicMaterial({ color: '#ffffff' }), true);
  eyes.push({ group: eye, pupil });
}
// Expressive thick eyebrows built from curved tubes.
const brows = [];
for (const side of [-1, 1]) {
  const brow = new THREE.Group();
  brow.position.set(side * 0.15, 0.265, 0.305);
  head.add(brow);
  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.12, -0.012, 0),
    new THREE.Vector3(-0.05, 0.032, 0.018),
    new THREE.Vector3(0.04, 0.035, 0.018),
    new THREE.Vector3(0.12, -0.005, 0)
  ]);
  addMesh(new THREE.TubeGeometry(curve, 18, 0.025, 8, false), MAT.hair, brow);
  brows.push(brow);
}
// Forehead worry lines.
for (let i = 0; i < 2; i++) {
  const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(-0.13, 0.345 + i * 0.05, 0.275), new THREE.Vector3(0, 0.375 + i * 0.05, 0.3), new THREE.Vector3(0.13, 0.345 + i * 0.05, 0.275));
  const wrinkleMat = material('#b8755c', 0.9);
  addMesh(new THREE.TubeGeometry(curve, 12, 0.006, 5, false), wrinkleMat, head);
}
// Mouth rig: crooked line, open jaw, teeth, and lower lip.
const mouth = new THREE.Group();
mouth.position.set(0, -0.285, 0.3);
head.add(mouth);
const mouthCurve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(-0.14, 0.015, 0), new THREE.Vector3(0, -0.045, 0.025), new THREE.Vector3(0.14, 0.01, 0));
const mouthLine = addMesh(new THREE.TubeGeometry(mouthCurve, 20, 0.014, 7, false), MAT.lips, mouth);
const openMouth = sphere(mouth, [0, -0.015, -0.005], [0.14, 0.06, 0.025], material('#4a2020', 0.8));
const teeth = box(mouth, [0, 0.025, 0.022], [0.19, 0.045, 0.018], MAT.shirt, 0.012);
const lowerLip = sphere(mouth, [0, -0.075, 0.015], [0.12, 0.025, 0.025], MAT.lips, true);
// High forehead, neat side-parted dark hair, and individually animated tufts.
sphere(head, [0, 0.16, -0.11], [0.37, 0.39, 0.25], MAT.hair);
const hairCap = addMesh(new THREE.SphereGeometry(1, 30, 18, 0, Math.PI * 2, 0, Math.PI * 0.34), MAT.hair, head, [0, 0.28, -0.025], [0.39, 0.34, 0.34]);
for (const side of [-1, 1]) {
  const sideburn = sphere(head, [side * 0.33, 0.16, -0.01], [0.065, 0.21, 0.12], MAT.hair);
  sideburn.rotation.z = side * 0.16;
}
const hairTufts = [];
for (let i = 0; i < 7; i++) {
  const tuft = sphere(head, [-0.25 + i * 0.083, 0.53 + Math.sin(i) * 0.012, -0.005 + Math.abs(i - 3) * -0.008], [0.095, 0.075, 0.115], i % 2 ? MAT.hairLight : MAT.hair, true);
  tuft.rotation.z = (i - 3) * -0.12 + (i < 3 ? -0.28 : 0.2);
  tuft.userData.baseRot = tuft.rotation.z;
  hairTufts.push(tuft);
}

// Arms are procedural two-bone rigs, retargeted every frame.
function createHand(side) {
  const hand = new THREE.Group();
  scene.add(hand);
  sphere(hand, [0, 0.04, 0], [0.125, 0.155, 0.075], MAT.skin);
  const fingers = [];
  for (let i = 0; i < 4; i++) {
    const finger = new THREE.Group();
    finger.position.set(-0.083 + i * 0.054, 0.15, 0.005);
    hand.add(finger);
    const f = sphere(finger, [0, 0.045, 0], [0.027, 0.09 - i * 0.005, 0.028], MAT.skinLight, true);
    fingers.push(finger);
  }
  const thumb = sphere(hand, [side * 0.12, 0.03, 0.015], [0.04, 0.1, 0.04], MAT.skinLight, true);
  thumb.rotation.z = side * -0.7;
  return { group: hand, fingers, thumb };
}
function createArm(side) {
  const upper = cyl(scene, [0, 0, 0], [1, 1, 1], MAT.jacket);
  const lower = cyl(scene, [0, 0, 0], [1, 1, 1], MAT.jacket);
  const elbow = sphere(scene, [0, 0, 0], [0.135, 0.135, 0.135], MAT.jacket);
  const cuff = cyl(scene, [0, 0, 0], [1, 1, 1], MAT.shirt);
  const hand = createHand(side);
  return { upper, lower, elbow, cuff, hand, side };
}
const arms = [createArm(-1), createArm(1)];
const vA = new THREE.Vector3();
const vB = new THREE.Vector3();
const vC = new THREE.Vector3();
function updateArm(arm, shoulder, elbow, wrist, fingerTap = 0, scratch = 0) {
  setSegment(arm.upper, shoulder, elbow, 0.135);
  setSegment(arm.lower, elbow, wrist, 0.115);
  arm.elbow.position.copy(elbow);
  const cuffEnd = wrist.clone().lerp(elbow, 0.12);
  setSegment(arm.cuff, wrist, cuffEnd, 0.122);
  arm.hand.group.position.copy(wrist);
  const dir = new THREE.Vector3().subVectors(wrist, elbow).normalize();
  arm.hand.group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  arm.hand.fingers.forEach((f, i) => {
    f.rotation.x = -fingerTap * (0.25 + i * 0.06) + Math.sin(scratch * Math.PI * 6 + i * 0.7) * scratch * 0.28;
    f.rotation.z = (i - 1.5) * 0.035 + Math.sin(scratch * Math.PI * 4 + i) * scratch * 0.12;
  });
}

// ---------- Floating customer inquiries ----------
function inquiryTexture(name, message, color) {
  const c = makeCanvas(768, 400);
  const x = c.getContext('2d');
  x.fillStyle = '#f5f0df';
  roundRect(x, 5, 5, 758, 390, 44); x.fill();
  x.fillStyle = '#263c3e';
  roundRect(x, 5, 5, 758, 82, 40); x.fill();
  x.fillRect(5, 48, 758, 40);
  x.fillStyle = '#8cd2b4';
  x.beginPath(); x.arc(55, 46, 13, 0, Math.PI * 2); x.fill();
  x.font = '800 31px Inter, sans-serif';
  x.fillStyle = '#ffffff';
  x.textAlign = 'left';
  x.fillText('NEW CUSTOMER INQUIRY', 88, 58);
  x.fillStyle = color;
  x.beginPath(); x.arc(105, 204, 66, 0, Math.PI * 2); x.fill();
  x.fillStyle = '#f4d4b4';
  x.beginPath(); x.arc(105, 181, 28, 0, Math.PI * 2); x.fill();
  x.beginPath(); x.arc(105, 252, 48, Math.PI, Math.PI * 2); x.fill();
  x.fillStyle = '#26302f';
  x.font = '800 39px Inter, sans-serif';
  x.fillText(name, 198, 178);
  x.fillStyle = '#66716c';
  x.font = '500 31px Inter, sans-serif';
  x.fillText(message, 198, 232);
  x.fillStyle = '#a84837';
  x.font = '700 25px Inter, sans-serif';
  x.fillText('just now  •  reply needed', 198, 294);
  return canvasTexture(c);
}
const inquirySpecs = [
  { start: 2.65, name: 'Sarah, Notting Hill', msg: 'Can I place an order?', color: '#b86d53' },
  { start: 3.37, name: 'David, Camden', msg: 'Is your site working?', color: '#567f85' },
  { start: 4.09, name: 'Priya, Soho', msg: 'Hello? Hello??', color: '#8d6a96' }
];
const inquiries = inquirySpecs.map((spec, idx) => {
  const group = new THREE.Group();
  scene.add(group);
  box(group, [0, 0, -0.025], [0.95, 0.5, 0.055], MAT.shirt, 0.055);
  const tm = new THREE.MeshBasicMaterial({ map: inquiryTexture(spec.name, spec.msg, spec.color), transparent: true });
  addMesh(new THREE.PlaneGeometry(0.93, 0.485), tm, group, [0, 0, 0.012]);
  const particles = [];
  for (let j = 0; j < 10; j++) {
    const p = addMesh(new THREE.OctahedronGeometry(0.035, 0), new THREE.MeshBasicMaterial({ color: j % 2 ? '#f5cf70' : '#d8f1e5', transparent: true }), scene, [0, 0, 0]);
    p.visible = false;
    particles.push(p);
  }
  return { group, particles, spec, idx };
});

// ---------- Broken retro website drawn onto the CRT ----------
let lastScreenTick = -1;
function drawWebsite(t) {
  const tick = Math.floor(t * 15);
  if (tick === lastScreenTick) return;
  lastScreenTick = tick;
  const x = screenCtx;
  const w = screenCanvas.width;
  const h = screenCanvas.height;
  x.clearRect(0, 0, w, h);
  x.fillStyle = '#c4c0ae';
  x.fillRect(0, 0, w, h);
  x.fillStyle = '#183f59';
  x.fillRect(0, 0, w, 55);
  x.fillStyle = '#fff';
  x.font = '700 25px Inter, sans-serif';
  x.textAlign = 'left';
  x.fillText('BeanNet Navigator', 22, 37);
  for (let i = 0; i < 3; i++) {
    x.fillStyle = ['#b94b3e', '#d4aa45', '#6b9b68'][i];
    x.beginPath(); x.arc(w - 105 + i * 34, 28, 10, 0, Math.PI * 2); x.fill();
  }
  x.fillStyle = '#e8e5d8';
  x.fillRect(18, 70, w - 36, 45);
  x.strokeStyle = '#686c66'; x.lineWidth = 2; x.strokeRect(18, 70, w - 36, 45);
  x.fillStyle = '#3e4844'; x.font = '500 21px monospace';
  x.fillText('http://www.beans-brilliant-business.co.uk', 34, 100);
  x.fillStyle = '#f2edda';
  x.fillRect(0, 130, w, h - 130);
  if (t >= 8.12) {
    x.fillStyle = '#183f59'; x.fillRect(0, 130, w, h - 130);
    x.textAlign = 'center';
    x.fillStyle = '#d7e5dd'; x.font = '900 44px Inter, sans-serif';
    x.fillText('YOUR ONLINE BUSINESS', w / 2, 215);
    x.fillStyle = '#f2c45c'; x.font = '900 132px Inter, sans-serif';
    x.fillText('0', w / 2, 370);
    x.fillStyle = '#ffffff'; x.font = '900 49px Inter, sans-serif';
    x.fillText('CUSTOMERS', w / 2, 440);
    x.fillStyle = '#aac0c1'; x.font = '500 25px monospace';
    x.fillText('but excellent shaking technique.', w / 2, 505);
    x.strokeStyle = '#d7e5dd'; x.lineWidth = 7;
    x.beginPath(); x.arc(w / 2, 564, 28, Math.PI * 1.12, Math.PI * 1.88); x.stroke();
  } else {
    // Intentionally hopeless homemade site.
    x.textAlign = 'center';
    x.fillStyle = '#763329'; x.font = '900 47px Inter, sans-serif';
    x.fillText("BEAN'S BRILLIANT", w / 2, 194);
    x.fillStyle = '#2e625b'; x.font = '900 43px Inter, sans-serif';
    x.fillText('ONLINE BUSINESS', w / 2, 242);
    x.fillStyle = '#68685c'; x.font = '500 20px monospace';
    x.fillText('Welcome to the future. Please wait.', w / 2, 280);
    // Broken image.
    x.fillStyle = '#d7d2bf'; x.fillRect(55, 315, 265, 178);
    x.strokeStyle = '#8a8679'; x.lineWidth = 3; x.strokeRect(55, 315, 265, 178);
    x.strokeStyle = '#a5483c'; x.lineWidth = 11;
    x.beginPath(); x.moveTo(80, 338); x.lineTo(292, 468); x.moveTo(292, 338); x.lineTo(80, 468); x.stroke();
    x.fillStyle = '#6d6a5f'; x.font = '700 17px monospace'; x.fillText('PRODUCT_PHOTO_FINAL2.jpg', 187, 523);
    // Crooked buttons and construction stripes.
    x.save(); x.translate(560, 355); x.rotate(-0.035);
    x.fillStyle = '#3f786a'; roundRect(x, -150, -32, 300, 65, 9); x.fill();
    x.fillStyle = '#fff'; x.font = '800 27px Inter, sans-serif'; x.fillText('BUY NOW (probably)', 0, 10); x.restore();
    x.save(); x.translate(625, 445); x.rotate(0.065);
    x.fillStyle = '#a14b3d'; roundRect(x, -150, -30, 300, 60, 8); x.fill();
    x.fillStyle = '#fff'; x.font = '800 26px Inter, sans-serif'; x.fillText('CONTACT ME!!!', 0, 9); x.restore();
    for (let i = 0; i < 11; i++) {
      x.fillStyle = i % 2 ? '#332e26' : '#e5b848';
      x.save(); x.translate(380 + i * 47, 547); x.rotate(-0.5); x.fillRect(-10, -25, 24, 52); x.restore();
    }
    x.fillStyle = '#3b372f'; x.font = '900 22px Inter, sans-serif';
    x.fillText('UNDER CONSTRUCTION SINCE 1997', 605, 595);
    // Inquiry counter visibly drains away.
    let count = 0;
    if (t > 2.82) count = 1;
    if (t > 3.47) count = 2;
    if (t > 4.15) count = 3;
    if (t > 4.75) count = 1;
    if (t > 5.05) count = 0;
    x.textAlign = 'left';
    x.fillStyle = count ? '#285e50' : '#a33e34';
    roundRect(x, 42, 551, 300, 60, 8); x.fill();
    x.fillStyle = '#fff'; x.font = '900 25px Inter, sans-serif';
    x.fillText(`INQUIRIES: ${count}`, 65, 591);
    // Error pop-up arrives just as he starts to feel confident.
    if ((t > 1.78 && t < 2.5) || (t > 5.05 && t < 6.35)) {
      const px = 195 + Math.sin(t * 3) * 12;
      const py = 240;
      x.fillStyle = '#eee9d8'; x.fillRect(px, py, 570, 210);
      x.strokeStyle = '#363c39'; x.lineWidth = 5; x.strokeRect(px, py, 570, 210);
      x.fillStyle = '#9c362e'; x.fillRect(px, py, 570, 46);
      x.fillStyle = '#fff'; x.font = '800 24px Inter, sans-serif'; x.fillText('BeanNet — Tiny Problem', px + 18, py + 32);
      x.fillStyle = '#3e413b'; x.font = '900 31px Inter, sans-serif';
      x.fillText(t < 3 ? '404: WEBSITE NOT FOUND' : 'ERROR: INQUIRIES MISSING', px + 285, py + 105);
      x.font = '500 21px monospace';
      x.fillText('Have you tried looking under the desk?', px + 285, py + 145);
      x.fillStyle = '#c8c2ae'; x.fillRect(px + 215, py + 163, 140, 34);
      x.strokeStyle = '#666'; x.lineWidth = 2; x.strokeRect(px + 215, py + 163, 140, 34);
      x.fillStyle = '#333'; x.font = '700 18px Inter, sans-serif'; x.textAlign = 'center'; x.fillText('OH DEAR', px + 285, py + 187);
    }
  }
  // Scanlines and intermittent physical glitch during the shake.
  x.globalAlpha = 0.1;
  x.fillStyle = '#0b2428';
  for (let yy = 0; yy < h; yy += 5) x.fillRect(0, yy, w, 1);
  x.globalAlpha = 1;
  if (t > 6.35 && t < 8.12) {
    const strength = smooth(6.35, 6.55, t) * (1 - smooth(7.95, 8.12, t));
    for (let i = 0; i < 9; i++) {
      const yy = hash(tick * 20 + i) * h;
      const hh = 4 + hash(tick * 30 + i) * 25;
      x.fillStyle = i % 3 === 0 ? `rgba(223,75,53,${0.25 * strength})` : `rgba(73,205,195,${0.23 * strength})`;
      x.fillRect((hash(tick + i) - 0.5) * 90 * strength, yy, w, hh);
    }
    if (hash(tick) > 0.72) {
      x.fillStyle = 'rgba(240,235,210,0.65)';
      x.fillRect(0, hash(tick + 1) * h, w, 5 + hash(tick + 2) * 15);
    }
  }
  const vignette = x.createRadialGradient(w / 2, h / 2, h * 0.15, w / 2, h / 2, w * 0.62);
  vignette.addColorStop(0, 'rgba(0,0,0,0)');
  vignette.addColorStop(1, 'rgba(0,16,20,0.38)');
  x.fillStyle = vignette; x.fillRect(0, 0, w, h);
  screenTexture.needsUpdate = true;
}

// ---------- Cinematic typography overlay ----------
const overlayCanvas = makeCanvas(W, H);
const overlayCtx = overlayCanvas.getContext('2d');
const overlayTexture = canvasTexture(overlayCanvas);
const overlayScene = new THREE.Scene();
const overlayCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
overlayCamera.position.z = 1;
const overlayMat = new THREE.MeshBasicMaterial({ map: overlayTexture, transparent: true, depthTest: false, depthWrite: false });
overlayScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), overlayMat));
const vignetteCanvas = makeCanvas(W, H);
{
  const x = vignetteCanvas.getContext('2d');
  const g = x.createRadialGradient(W * 0.5, H * 0.44, H * 0.14, W * 0.5, H * 0.48, H * 0.66);
  g.addColorStop(0, 'rgba(10,8,6,0)');
  g.addColorStop(0.65, 'rgba(10,8,6,0.06)');
  g.addColorStop(1, 'rgba(8,6,5,0.56)');
  x.fillStyle = g; x.fillRect(0, 0, W, H);
  const top = x.createLinearGradient(0, 0, 0, 280);
  top.addColorStop(0, 'rgba(8,9,8,0.28)'); top.addColorStop(1, 'rgba(8,9,8,0)');
  x.fillStyle = top; x.fillRect(0, 0, W, 280);
}
function fitFont(ctx, text, maxWidth, startSize, weight = 900) {
  let size = startSize;
  do {
    ctx.font = `${weight} ${size}px Inter, Arial, sans-serif`;
    if (ctx.measureText(text).width <= maxWidth) break;
    size -= 2;
  } while (size > 20);
  return size;
}
function drawOverlay(t) {
  const x = overlayCtx;
  x.clearRect(0, 0, W, H);
  x.drawImage(vignetteCanvas, 0, 0);
  // Understated establishing locator.
  const locationAlpha = smooth(0.15, 0.45, t) * (1 - smooth(2.05, 2.4, t));
  if (locationAlpha > 0.001) {
    x.globalAlpha = locationAlpha;
    x.fillStyle = 'rgba(19,25,24,0.64)';
    roundRect(x, 72, 100, 360, 72, 36); x.fill();
    x.fillStyle = '#c34c39';
    x.beginPath(); x.arc(111, 136, 8, 0, Math.PI * 2); x.fill();
    x.fillStyle = '#f1eadc';
    x.font = '750 27px Inter, sans-serif';
    x.textAlign = 'left'; x.textBaseline = 'middle';
    x.fillText('LONDON  •  09:17', 137, 137);
    x.globalAlpha = 1;
  }
  // Final commercial end-card, large enough for mobile and clear of the face.
  const end = smooth(8.22, 8.58, t);
  if (end > 0.001) {
    const bg = x.createLinearGradient(0, 1090, 0, H);
    bg.addColorStop(0, 'rgba(13,17,16,0)');
    bg.addColorStop(0.33, `rgba(13,17,16,${0.7 * end})`);
    bg.addColorStop(1, `rgba(13,17,16,${0.94 * end})`);
    x.fillStyle = bg; x.fillRect(0, 1050, W, H - 1050);
    const enter = backOut(end);
    x.save();
    x.translate(W / 2, 1505 + (1 - end) * 70);
    x.scale(0.91 + enter * 0.09, 0.91 + enter * 0.09);
    x.translate(-W / 2, -1505);
    x.globalAlpha = end;
    x.textAlign = 'center'; x.textBaseline = 'middle';
    x.shadowColor = 'rgba(0,0,0,0.55)'; x.shadowBlur = 22; x.shadowOffsetY = 8;
    x.fillStyle = '#c64c38';
    roundRect(x, W / 2 - 35, 1270, 70, 10, 5); x.fill();
    const line1 = 'EVEN MR. BEAN';
    const line2 = 'NEEDS A WEBSITE!';
    fitFont(x, line1, 940, 112);
    x.fillStyle = '#f7f2e7';
    x.fillText(line1, W / 2, 1405);
    fitFont(x, line2, 950, 112);
    x.fillStyle = '#f4c95d';
    x.fillText(line2, W / 2, 1540);
    x.shadowColor = 'transparent';
    x.fillStyle = 'rgba(240,234,217,0.72)';
    x.font = '650 25px Inter, sans-serif';
    x.fillText('GREAT IDEAS DESERVE TO BE FOUND.', W / 2, 1660);
    x.globalAlpha = 1;
    x.restore();
  }
  overlayTexture.needsUpdate = true;
}

// ---------- Performance: 10-second continuous character and camera choreography ----------
const cameraKeys = [
  { t: 0.0, p: [3.25, 3.15, 9.25], q: [0.0, 1.65, 0.02], fov: 44 },
  { t: 2.55, p: [2.55, 2.9, 7.7], q: [-0.05, 1.72, 0.0], fov: 43 },
  { t: 3.0, p: [2.35, 2.85, 7.25], q: [-0.05, 1.8, 0.0], fov: 42 },
  { t: 4.85, p: [1.95, 2.85, 6.85], q: [-0.25, 1.9, -0.05], fov: 41 },
  { t: 5.25, p: [0.85, 2.85, 5.6], q: [-0.73, 2.22, -0.36], fov: 39 },
  { t: 6.18, p: [0.95, 2.82, 5.8], q: [-0.66, 2.18, -0.32], fov: 40 },
  { t: 6.55, p: [2.45, 2.95, 7.4], q: [0.0, 1.78, 0.0], fov: 43 },
  { t: 8.15, p: [2.7, 3.0, 7.85], q: [0.0, 1.75, 0.0], fov: 44 },
  { t: 10.0, p: [3.05, 3.12, 8.6], q: [0.0, 1.73, 0.0], fov: 44 }
];
function updateCamera(t) {
  let i = 0;
  while (i < cameraKeys.length - 2 && t > cameraKeys[i + 1].t) i++;
  const a = cameraKeys[i];
  const b = cameraKeys[i + 1];
  const f = easeInOut(clamp((t - a.t) / (b.t - a.t)));
  camera.position.set(
    lerp(a.p[0], b.p[0], f),
    lerp(a.p[1], b.p[1], f),
    lerp(a.p[2], b.p[2], f)
  );
  const target = new THREE.Vector3(
    lerp(a.q[0], b.q[0], f),
    lerp(a.q[1], b.q[1], f),
    lerp(a.q[2], b.q[2], f)
  );
  if (t > 6.4 && t < 8.05) {
    const shake = smooth(6.4, 6.55, t) * (1 - smooth(7.85, 8.05, t));
    camera.position.x += Math.sin(t * 47) * 0.013 * shake;
    camera.position.y += Math.sin(t * 39) * 0.009 * shake;
  }
  camera.fov = lerp(a.fov, b.fov, f);
  camera.updateProjectionMatrix();
  camera.lookAt(target);
}
function localWorld(obj, xyz) {
  return obj.localToWorld(new THREE.Vector3(...xyz));
}
function setArmFromBean(arm, elbow, wrist, tap, scratch) {
  const side = arm.side;
  const shoulder = localWorld(body, [side * 0.48, 0.42, 0.0]);
  updateArm(arm, shoulder, elbow, wrist, tap, scratch);
}

function animateCharacter(t) {
  const typing = 1 - smooth(2.25, 2.7, t);
  const confusion = smooth(2.0, 2.55, t) * (1 - smooth(6.2, 6.45, t));
  const scratchPhase = smooth(5.05, 5.38, t) * (1 - smooth(6.12, 6.42, t));
  const grabbing = smooth(6.25, 6.55, t) * (1 - smooth(8.02, 8.42, t));
  const shrug = smooth(8.32, 8.75, t);
  const shakeEnv = smooth(6.4, 6.56, t) * (1 - smooth(7.9, 8.12, t));
  const shake = Math.sin(t * 38) * shakeEnv;
  const breath = Math.sin(t * 2.4) * 0.012;

  // Move the CRT before retargeting the hands so his grip stays physically attached.
  computer.position.x = 0.58 + shake * 0.085;
  computer.position.y = 1.72 + Math.abs(shake) * 0.035;
  computer.rotation.z = shake * 0.075;
  computer.rotation.y = shake * 0.04;
  computer.updateMatrixWorld(true);

  body.position.x = 0.18 * grabbing + 0.04 * shrug;
  body.position.y = 1.35 + breath + Math.abs(shake) * 0.012;
  body.rotation.z = -0.055 * scratchPhase - 0.11 * grabbing + shake * 0.018 + 0.035 * shrug;
  body.rotation.y = 0.05 + 0.13 * typing - 0.09 * shrug;

  // Facial performance: determined -> baffled -> head scratch -> panic -> sheepish shrug.
  let gazeX = 0.022 * typing + 0.035 * confusion;
  let gazeY = -0.005;
  if (t > 2.7 && t < 5.0) {
    gazeX = 0.025 + Math.sin(t * 3.4) * 0.024;
    gazeY = 0.012 + Math.sin(t * 4.1) * 0.009;
  }
  if (scratchPhase > 0) {
    gazeX = lerp(gazeX, -0.022, scratchPhase);
    gazeY = lerp(gazeY, 0.036, scratchPhase);
  }
  if (shrug > 0) {
    gazeX = lerp(gazeX, 0.0, shrug);
    gazeY = lerp(gazeY, -0.004, shrug);
  }
  const blinkPulse = (center, width) => clamp(1 - Math.abs(t - center) / width);
  const blink = Math.max(blinkPulse(1.52, 0.07), blinkPulse(4.86, 0.065), blinkPulse(8.72, 0.075));
  const wideEyes = Math.max(confusion * 0.22, grabbing * 0.72);
  eyes.forEach((eye, i) => {
    eye.pupil.position.x = gazeX + (i === 0 && confusion > 0.4 ? -0.007 * confusion : 0);
    eye.pupil.position.y = gazeY;
    eye.group.scale.y = Math.max(0.07, (1 + wideEyes * 0.3) * (1 - blink * 0.96));
    eye.group.scale.x = 1 + wideEyes * 0.08;
  });
  brows[0].position.y = 0.265 + confusion * 0.075 + grabbing * 0.055 - shrug * 0.015;
  brows[1].position.y = 0.265 - confusion * 0.025 + grabbing * 0.09 + shrug * 0.065;
  brows[0].rotation.z = -0.08 - confusion * 0.22 + shrug * 0.18;
  brows[1].rotation.z = 0.06 + confusion * 0.28 - shrug * 0.12;

  head.rotation.y = 0.2 * typing + 0.18 * confusion - 0.16 * scratchPhase - 0.16 * grabbing - 0.03 * shrug;
  head.rotation.z = -0.04 * typing - 0.12 * confusion - 0.13 * scratchPhase + shake * 0.035 + 0.09 * shrug;
  head.rotation.x = 0.025 * Math.sin(t * 2.0) + 0.06 * scratchPhase - 0.04 * grabbing + 0.035 * shrug;
  head.position.y = 1.08 + Math.sin(t * 2.4) * 0.006;

  const mouthOpen = grabbing * (0.65 + Math.abs(shake) * 0.25) + confusion * 0.12;
  openMouth.scale.y = 0.04 + mouthOpen * 0.16;
  openMouth.scale.x = 0.12 + confusion * 0.025 + grabbing * 0.025;
  openMouth.visible = mouthOpen > 0.04;
  teeth.visible = mouthOpen > 0.28;
  lowerLip.visible = mouthOpen > 0.2;
  mouthLine.visible = mouthOpen <= 0.28;
  mouthLine.scale.y = 1 - confusion * 0.55 + shrug * 1.2;
  mouthLine.rotation.z = -0.11 * confusion + 0.09 * shrug;
  mouth.position.y = -0.285 - mouthOpen * 0.025;
  lowerLip.position.y = -0.055 - mouthOpen * 0.08;

  const hairWiggle = scratchPhase * Math.sin(t * 32) * 0.09 + shake * 0.028;
  hairTufts.forEach((tuft, i) => {
    tuft.rotation.z = tuft.userData.baseRot + hairWiggle * (0.7 + i * 0.08);
  });

  bean.updateMatrixWorld(true);
  // Default typing pose.
  const clickA = Math.pow(Math.max(0, Math.sin(t * 10.5)), 8) * typing;
  const clickB = Math.pow(Math.max(0, Math.sin(t * 12.2 + 1.7)), 8) * typing;
  let leftElbow = new THREE.Vector3(-1.36, 1.35, 0.02);
  let rightElbow = new THREE.Vector3(-0.25, 1.34, 0.0);
  let leftWrist = new THREE.Vector3(-0.63, 1.18 + clickA * 0.055, 0.66);
  let rightWrist = new THREE.Vector3(-0.06, 1.18 + clickB * 0.055, 0.68);
  let tapL = clickA;
  let tapR = clickB;
  let scratch = 0;

  if (confusion > 0 && grabbing === 0) {
    // Hands slow down and hover uncertainly above the keyboard.
    leftWrist.y += confusion * 0.1;
    rightWrist.y += confusion * 0.13;
    rightWrist.x += confusion * 0.07;
  }
  if (scratchPhase > 0) {
    const s = scratchPhase;
    const scratchElbow = new THREE.Vector3(-0.12, 2.13, -0.02);
    const scratchWrist = new THREE.Vector3(-0.62 + Math.sin(t * 30) * 0.025, 2.86, -0.18);
    rightElbow.lerp(scratchElbow, s);
    rightWrist.lerp(scratchWrist, s);
    leftWrist.y += s * 0.08;
    scratch = s * smooth(5.32, 5.48, t) * (1 - smooth(5.95, 6.15, t));
    tapR = 0.25;
  }
  if (grabbing > 0) {
    computer.updateMatrixWorld(true);
    const leftGrab = localWorld(computer, [-0.77, 0.02, 0.16]);
    const rightGrab = localWorld(computer, [0.77, 0.02, 0.16]);
    leftElbow.lerp(new THREE.Vector3(-0.65, 1.48, 0.25), grabbing);
    rightElbow.lerp(new THREE.Vector3(0.92, 1.3, 0.25), grabbing);
    leftWrist.lerp(leftGrab, grabbing);
    rightWrist.lerp(rightGrab, grabbing);
    tapL = 0.65; tapR = 0.65;
  }
  if (shrug > 0) {
    const s = shrug;
    leftElbow.lerp(new THREE.Vector3(-1.45, 1.4, -0.12), s);
    rightElbow.lerp(new THREE.Vector3(-0.12, 1.43, -0.08), s);
    leftWrist.lerp(new THREE.Vector3(-1.53, 1.88 + Math.sin(t * 3) * 0.015, 0.12), s);
    rightWrist.lerp(new THREE.Vector3(-0.12, 1.91 + Math.sin(t * 3 + 1) * 0.015, 0.15), s);
    tapL = -0.55; tapR = -0.55;
  }
  setArmFromBean(arms[0], leftElbow, leftWrist, tapL, 0);
  setArmFromBean(arms[1], rightElbow, rightWrist, tapR, scratch);

  // Physical comedy transfers into the computer and nearby props.
  keyboard.position.y = 1.09 + Math.abs(shake) * 0.005;
  mouse.rotation.y = shake * 0.08;
  mug.rotation.z = shake * 0.018;
  spoon.rotation.z = -0.18 + shake * 0.09;
  teddy.rotation.z = -shake * 0.012;
  paperGroups.forEach((p, i) => {
    p.position.y = 1.025 + i * 0.001 + Math.max(0, shake * Math.sin(i * 2.1)) * 0.018;
    p.rotation.z = shake * 0.015 * (i % 2 ? 1 : -1);
  });
}

function animateInquiries(t) {
  for (const item of inquiries) {
    const { group, particles, spec, idx } = item;
    const lt = t - spec.start;
    group.visible = lt >= 0 && lt < 1.2;
    if (group.visible) {
      const enter = easeOut(clamp(lt / 0.32));
      const vanish = smooth(0.72, 1.18, lt);
      const baseX = 1.05 + idx * 0.08;
      const baseY = 2.62 - idx * 0.28;
      group.position.set(
        lerp(3.3, baseX, enter),
        baseY + Math.sin(t * 4 + idx) * 0.045,
        1.05 + idx * 0.08
      );
      group.position.lerp(new THREE.Vector3(0.6, 1.74, 0.62), vanish);
      const s = Math.max(0.001, enter * (1 - vanish * 0.98));
      group.scale.setScalar(s);
      group.lookAt(camera.position);
      group.rotateZ((1 - enter) * -0.18 + vanish * (idx % 2 ? 1 : -1) * 1.3);
    }
    const puff = lt - 1.0;
    particles.forEach((p, j) => {
      p.visible = puff > 0 && puff < 0.42;
      if (p.visible) {
        const a = j / particles.length * Math.PI * 2 + idx;
        const rr = puff * (0.35 + hash(j + idx * 20) * 0.45);
        p.position.set(0.6 + Math.cos(a) * rr, 1.74 + Math.sin(a) * rr, 0.75 + puff * 0.25);
        p.scale.setScalar((1 - puff / 0.42) * (0.7 + hash(j) * 0.6));
        p.material.opacity = 1 - puff / 0.42;
        p.rotation.x = p.rotation.y = t * 4 + j;
      }
    });
  }
}

function renderFrame(time) {
  const t = clamp(time, 0, DURATION);
  updateCamera(t);
  animateCharacter(t);
  animateInquiries(t);
  drawWebsite(t);
  drawOverlay(t);
  rainDrops.forEach((drop, i) => {
    const range = 1.75;
    const raw = drop.userData.startY - t * drop.userData.speed;
    drop.position.y = ((raw + 0.875) % range + range) % range - 0.875;
  });
  // Slight practical-light flutter only while the CRT is being abused.
  bulb.intensity = 7 + (t > 6.4 && t < 8.1 ? Math.sin(t * 31) * 0.5 : 0);
  renderer.render(scene, camera);
  renderer.autoClear = false;
  renderer.render(overlayScene, overlayCamera);
  renderer.autoClear = true;
  return canvas;
}
window.renderFrame = renderFrame;
window.captureFrame = time => {
  renderFrame(time);
  return canvas.toDataURL('image/jpeg', 0.965);
};
window.sceneInfo = () => ({
  meshes: renderer.info.render.calls,
  triangles: renderer.info.render.triangles,
  duration: DURATION,
  width: W,
  height: H
});

await document.fonts.load('900 100px Inter');
await document.fonts.ready;
renderFrame(0);
window.__ready = true;
