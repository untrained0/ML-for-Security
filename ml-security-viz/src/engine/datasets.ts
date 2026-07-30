/**
 * datasets.js — Built-in 2D dataset generators
 */

function randNormal() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function normalize({ X, y }) {
  const d = X[0].length;
  const mins = new Array(d).fill(Infinity);
  const maxs = new Array(d).fill(-Infinity);
  for (const x of X) for (let j = 0; j < d; j++) {
    if (x[j] < mins[j]) mins[j] = x[j];
    if (x[j] > maxs[j]) maxs[j] = x[j];
  }
  const nX = X.map(x => x.map((v, j) => ((v - mins[j]) / (maxs[j] - mins[j] || 1)) * 4 - 2));
  return { X: nX, y };
}

export function makeMoons(n = 200, noise = 0.15) {
  const X = [], y = [], half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const a = Math.PI * i / half;
    X.push([Math.cos(a) + randNormal() * noise, Math.sin(a) + randNormal() * noise]);
    y.push(1);
  }
  for (let i = 0; i < n - half; i++) {
    const a = Math.PI * i / (n - half);
    X.push([1 - Math.cos(a) + randNormal() * noise, 1 - Math.sin(a) - 0.5 + randNormal() * noise]);
    y.push(-1);
  }
  return normalize({ X, y });
}

export function makeCircles(n = 200, noise = 0.1, factor = 0.5) {
  const X = [], y = [], half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const a = 2 * Math.PI * i / half;
    X.push([Math.cos(a) + randNormal() * noise, Math.sin(a) + randNormal() * noise]);
    y.push(1);
  }
  for (let i = 0; i < n - half; i++) {
    const a = 2 * Math.PI * i / (n - half);
    X.push([factor * Math.cos(a) + randNormal() * noise, factor * Math.sin(a) + randNormal() * noise]);
    y.push(-1);
  }
  return normalize({ X, y });
}

export function makeBlobs(n = 200, separation = 2.5) {
  const X = [], y = [], half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    X.push([-separation / 2 + randNormal() * 0.6, randNormal() * 0.6]);
    y.push(1);
  }
  for (let i = 0; i < n - half; i++) {
    X.push([separation / 2 + randNormal() * 0.6, randNormal() * 0.6]);
    y.push(-1);
  }
  return normalize({ X, y });
}

export function makeXOR(n = 200, noise = 0.2) {
  const X = [], y = [], perQ = Math.floor(n / 4);
  const centers = [
    { cx: -1, cy: -1, l: 1 }, { cx: 1, cy: 1, l: 1 },
    { cx: -1, cy: 1, l: -1 }, { cx: 1, cy: -1, l: -1 }
  ];
  for (let q = 0; q < 4; q++) {
    const ct = q < 3 ? perQ : n - 3 * perQ;
    for (let i = 0; i < ct; i++) {
      X.push([centers[q].cx * 0.5 + randNormal() * noise, centers[q].cy * 0.5 + randNormal() * noise]);
      y.push(centers[q].l);
    }
  }
  return normalize({ X, y });
}

export function makeGaussian(n = 200, overlap = 0.4) {
  const X = [], y = [], half = Math.floor(n / 2), sp = overlap * 1.5;
  for (let i = 0; i < half; i++) { X.push([-0.5 + randNormal() * sp, 0.3 + randNormal() * sp]); y.push(1); }
  for (let i = 0; i < n - half; i++) { X.push([0.5 + randNormal() * sp, -0.3 + randNormal() * sp]); y.push(-1); }
  return normalize({ X, y });
}

export function makeSpiral(n = 200, noise = 0.15) {
  const X = [], y = [], half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const r = i / half * 1.5, t = 1.75 * i / half * 2 * Math.PI + Math.PI / 2;
    X.push([r * Math.cos(t) + randNormal() * noise, r * Math.sin(t) + randNormal() * noise]);
    y.push(1);
  }
  for (let i = 0; i < n - half; i++) {
    const r = i / (n - half) * 1.5, t = 1.75 * i / (n - half) * 2 * Math.PI + Math.PI / 2 + Math.PI;
    X.push([r * Math.cos(t) + randNormal() * noise, r * Math.sin(t) + randNormal() * noise]);
    y.push(-1);
  }
  return normalize({ X, y });
}

export function splitDataset({ X, y }, trainR = 0.6, validR = 0.2) {
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const tEnd = Math.floor(n * trainR), vEnd = tEnd + Math.floor(n * validR);
  const s = is => ({ X: is.map(i => X[i]), y: is.map(i => y[i]) });
  return { train: s(idx.slice(0, tEnd)), valid: s(idx.slice(tEnd, vEnd)), test: s(idx.slice(vEnd)) };
}

/**
 * Split a dataset that includes image data and separate SVM features
 * Used for MNIST where canvas shows 2D PCA but SVM trains on higher-dim PCA
 */
export function splitDatasetWithImages({ X2D, Xsvm, y, images, labels }, trainR = 0.6, validR = 0.2) {
  const n = y.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const tEnd = Math.floor(n * trainR), vEnd = tEnd + Math.floor(n * validR);

  const split = (idxs) => ({
    X: idxs.map(i => Xsvm[i]),    // SVM training features
    X2D: idxs.map(i => X2D[i]),   // 2D canvas features
    y: idxs.map(i => y[i]),
    images: idxs.map(i => images[i]),
    labels: idxs.map(i => labels[i]),
  });

  return {
    train: split(idx.slice(0, tEnd)),
    valid: split(idx.slice(tEnd, vEnd)),
    test: split(idx.slice(vEnd)),
  };
}

export const DATASETS = {
  moons:    { name: 'Moons',    fn: makeMoons,    icon: '🌙', desc: 'Two interleaving crescents' },
  circles:  { name: 'Circles',  fn: makeCircles,  icon: '⭕', desc: 'Concentric rings' },
  blobs:    { name: 'Blobs',    fn: makeBlobs,     icon: '🫧', desc: 'Linearly separable clusters' },
  xor:      { name: 'XOR',      fn: makeXOR,       icon: '✖',  desc: 'XOR-pattern quadrants' },
  gaussian: { name: 'Gaussian', fn: makeGaussian,  icon: '📊', desc: 'Overlapping Gaussians' },
  spiral:   { name: 'Spiral',   fn: makeSpiral,    icon: '🌀', desc: 'Interleaving spirals' },
  mnist:    { name: 'MNIST 1v7', fn: null, icon: '🔢', desc: 'Handwritten digits 1 vs 7 (Biggio 2012)', isMNIST: true },
};

