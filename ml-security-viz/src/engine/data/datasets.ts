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

// ── Regression Dataset Generators ──

// Jagielski et al. 2018 §II-A: x ∈ [0,1]^d and y ∈ [0,1]. The InvFlip/BFlip initialisations
// (y ← 1−y, y ← round(1−y)) and the feasible-domain projection assume exactly this range.
function normalizeRegression({ X, y }: { X: number[][], y: number[] }) {
  const d = X[0].length;
  const mins = new Array(d).fill(Infinity);
  const maxs = new Array(d).fill(-Infinity);
  for (const x of X) for (let j = 0; j < d; j++) {
    if (x[j] < mins[j]) mins[j] = x[j];
    if (x[j] > maxs[j]) maxs[j] = x[j];
  }
  const nX = X.map(x => x.map((v, j) => (v - mins[j]) / (maxs[j] - mins[j] || 1)));
  const yMin = Math.min(...y), yMax = Math.max(...y);
  const nY = y.map(v => (v - yMin) / (yMax - yMin || 1));
  return { X: nX, y: nY };
}

export function makeLinearRegression(n = 100, noise = 0.3) {
  const X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 4 - 2);
    X.push([x]);
    y.push(2 * x + 1 + randNormal() * noise);
  }
  return normalizeRegression({ X, y });
}

export function makeQuadraticRegression(n = 100, noise = 0.3) {
  const X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 4 - 2);
    X.push([x]);
    y.push(x * x + randNormal() * noise);
  }
  return normalizeRegression({ X, y });
}

export function makeSinRegression(n = 100, noise = 0.2) {
  const X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 4 - 2);
    X.push([x]);
    y.push(Math.sin(2 * x) + randNormal() * noise);
  }
  return normalizeRegression({ X, y });
}

// ── Biggio et al. 2012 §3.1 artificial data ──

/**
 * "each class follows a Gaussian distribution with … µ− = [−1.5, 0], µ+ = [1.5, 0],
 * Σ− = Σ+ = 0.6 I" (so σ = √0.6 per axis, NOT 0.6). Training and validation sets have 25 and
 * 500 points per class. The paper gives no test size for this experiment; we draw 500/class.
 * Raw coordinates, no normalisation: Fig. 1 plots the box [−5,5]² and bounds the linear-kernel
 * attack to [−4,4]².
 */
export function makeBiggioGaussian() {
  const sd = Math.sqrt(0.6);
  const draw = (perClass: number) => {
    const X: number[][] = [], y: number[] = [];
    for (const [mu, label] of [[-1.5, -1], [1.5, 1]]) {
      for (let i = 0; i < perClass; i++) {
        X.push([mu + randNormal() * sd, randNormal() * sd]);
        y.push(label);
      }
    }
    return { X, y };
  };
  return {
    train: draw(25),
    valid: draw(500),
    test: draw(500),
    bounds: [-4, 4] as [number, number],
    displayRange: { xMin: -5, xMax: 5, yMin: -5, yMax: 5 },
  };
}

/**
 * Dataset registry. An entry either
 *   • has `fn(n)` returning unsplit {X, y}, split by `splitDataset` using `split` ratios, or
 *   • has `load(n)` returning (a promise of) an already-split dataset {train, valid, test, …}.
 * Optional dataset-level fields understood by the engine/UI:
 *   view         PCA window for drawing high-dimensional data (engine/data/view.ts)
 *   bounds       [lo, hi] box every feature of a poisoning point is projected onto
 *   displayRange canvas extent {xMin, xMax, yMin, yMax}
 *   imageShape   the features are an image of this shape ([28,28] MNIST, [32,32,3] CIFAR-10 channels last)
 */
const REG_SPLIT: [number, number] = [1 / 3, 1 / 3];           // Jagielski §V: 1/3 train/valid/test
const REG_RANGE = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };

const mnistPair = (pair: [number, number]) => () =>
  import('./loaders/mnistReal').then(m => m.loadMNISTPair(pair));
// Pang 2021: the training split is the online stream (burn-in, then the accumulative batches);
// S_val is 500 images like the reference code's validation batch. CIFAR's 3072-d images cost 4×
// MNIST's, so its stream and test sample are halved.
const pangImages = (source: 'mnist' | 'cifar10', pair: [number, number]) => () =>
  import('./loaders/images').then(m => m.loadImagePair(source, pair, source === 'mnist'
    ? { train: 2000, valid: 500, test: 'all' }
    : { train: 1000, valid: 500, test: 1000 }));
const regression = (key: 'warfarin' | 'loan' | 'house') => () =>
  import('./loaders/regression').then(m => m.loadRegressionDataset(key));

export const DATASETS: Record<string, any> = {
  moons:    { name: 'Moons',    fn: makeMoons,    icon: '🌙', desc: 'Two interleaving crescents' },
  circles:  { name: 'Circles',  fn: makeCircles,  icon: '⭕', desc: 'Concentric rings' },
  blobs:    { name: 'Blobs',    fn: makeBlobs,     icon: '🫧', desc: 'Linearly separable clusters' },
  xor:      { name: 'XOR',      fn: makeXOR,       icon: '✖',  desc: 'XOR-pattern quadrants' },
  gaussian: { name: 'Gaussian', fn: makeGaussian,  icon: '📊', desc: 'Overlapping Gaussians' },
  spiral:   { name: 'Spiral',   fn: makeSpiral,    icon: '🌀', desc: 'Interleaving spirals' },

  // Biggio et al. 2012 benchmarks
  biggioGaussian: { name: 'Gaussian (paper)', load: makeBiggioGaussian, icon: '📊', desc: 'Biggio 2012 §3.1: µ± = (±1.5, 0), Σ = 0.6I, 25/class train, 500/class validation' },
  mnist71: { name: 'MNIST 7 vs 1', load: mnistPair([7, 1]), icon: '🔢', desc: 'Real MNIST, 784 raw pixels in [0,1] (Biggio 2012 §3.2)' },
  mnist98: { name: 'MNIST 9 vs 8', load: mnistPair([9, 8]), icon: '🔢', desc: 'Real MNIST, 784 raw pixels in [0,1] (Biggio 2012 §3.2)' },
  mnist40: { name: 'MNIST 4 vs 0', load: mnistPair([4, 0]), icon: '🔢', desc: 'Real MNIST, 784 raw pixels in [0,1] (Biggio 2012 §3.2)' },

  // Pang et al. 2021 §4 — real MNIST / CIFAR-10 in pixel space: the victim and the ε-bounded
  // perturbations both work on the raw [0,1] pixels; `view` only draws them
  pangMnist35: { name: 'MNIST 3 vs 5', load: pangImages('mnist', [3, 5]), icon: '🔢', desc: 'Real MNIST 3 (+1) vs 5 (−1), 784 raw pixels: 2000-image stream, 500 S_val, full test set' },
  pangMnist71: { name: 'MNIST 7 vs 1', load: pangImages('mnist', [7, 1]), icon: '🔢', desc: 'Real MNIST 7 (+1) vs 1 (−1), 784 raw pixels — nearly separable, so the hardest to break' },
  pangCifarShipFrog: { name: 'CIFAR-10 Ship vs Frog', load: pangImages('cifar10', [8, 6]), icon: '🖼️', desc: 'Real CIFAR-10 ship (+1) vs frog (−1), 32×32×3 raw pixels: 1000-image stream, 500 S_val, 1000 test' },
  pangCifarPlaneBird: { name: 'CIFAR-10 Plane vs Bird', load: pangImages('cifar10', [0, 2]), icon: '🖼️', desc: 'Real CIFAR-10 airplane (+1) vs bird (−1), 32×32×3 raw pixels: 1000-image stream, 500 S_val, 1000 test' },

  // Regression: synthetic 1-D illustrations
  linearReg:    { name: 'Linear',    fn: makeLinearRegression,    icon: '📈', desc: 'y = 2x + 1 + ε', isRegression: true, split: REG_SPLIT, displayRange: REG_RANGE },
  quadraticReg: { name: 'Quadratic', fn: makeQuadraticRegression, icon: '📐', desc: 'y = x² + ε',     isRegression: true, split: REG_SPLIT, displayRange: REG_RANGE },
  sinReg:       { name: 'Sine',      fn: makeSinRegression,       icon: '〰️', desc: 'y = sin(2x) + ε', isRegression: true, split: REG_SPLIT, displayRange: REG_RANGE },

  // Jagielski et al. 2018 §V-A benchmarks (authors' preprocessed data)
  warfarin: { name: 'Warfarin (Health)', load: regression('warfarin'), icon: '💊', desc: 'IWPC Warfarin dose, 204 features, 1400 records (Jagielski 2018)', isRegression: true },
  loan:     { name: 'Lending Club (Loan)', load: regression('loan'), icon: '💸', desc: 'Lending Club interest rate, 88 features, 1400 records (Jagielski 2018)', isRegression: true },
  house:    { name: 'Ames House Prices', load: regression('house'), icon: '🏠', desc: 'Ames sale price, 274 features, 1400 records (Jagielski 2018)', isRegression: true },
};
