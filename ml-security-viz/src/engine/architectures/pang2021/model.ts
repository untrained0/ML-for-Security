/**
 * Pang et al. 2021 — the online victim: a binary classifier on raw image pixels.
 *
 * The victim is a feed-forward network  x → x̃ → [W_1, b_1] → φ → … → [W_L, b_L] → z  with one
 * output logit. With no hidden layer it is logistic regression; with one to three hidden layers it
 * is a multi-layer perceptron — the neural-network victim the paper's attack is designed for (it
 * attacks a ResNet-18; a dependency-free engine keeps the network small and fully connected).
 * Both share this one implementation, so everything below is exact for either.
 *
 * Kept as in the paper and the reference code (github.com/ShawnXYang/AccumulativeAttack):
 *   • inputs are the pixels in [0,1], and the perturbations live in that pixel space;
 *   • the model normalises its input internally, x̃ = (x − μ_c)/σ_c per colour channel, like the
 *     reference ResNet's forward(); we also divide by √d so a single learning rate β suits both
 *     784-d MNIST and 3072-d CIFAR-10 (a fixed rescaling of the features, nothing learned);
 *   • training is online SGD, θ_{t+1} = θ_t − β∇_θ L(S_t; θ_t) (Eq. 3), with the reference code's
 *     momentum (μ = 0.9; weight decay 1e-4 during burn-in only) — `optimStep`.
 *
 * Hidden layers see their input scaled by 1/√(fan-in) as well (the "NTK" parameterisation), for
 * the same reason as the 1/√d on the pixels: every layer's update then has the same size for a
 * given β, so one β fits logistic regression and the MLP alike, at any width.
 *
 * Labels are ±1; L is the mean logistic loss  L(S; θ) = (1/n) Σ log(1 + exp(−y_i z_i)).
 * Algorithm 1 needs, besides ∇_θ L, two second-order quantities: Hessian-vector products (for the
 * accumulation term G_t, Eq. 9) and the pixel gradient of a directional derivative, ∇_x[qᵀ∇_θ L]
 * (to craft batches whose gradient points along q). Both are one application of Pearlmutter's
 * R-operator, R_q{·} = ∂/∂r (·)(θ + r q)|_{r=0}, to the backward pass: R_q{∇_θ L} = H q and
 * R_q{∇_x L} = ∇_x[qᵀ∇_θ L]. `rop` below runs it exactly — no finite differences, no autodiff.
 */

export type Activation = 'relu' | 'tanh';

export interface Norm {
  mean: number[];     // per channel
  std: number[];      // per channel
  channels: number;   // pixels are channels-last, so feature j belongs to channel j % channels
  scale: number;      // 1/√d
}

/**
 * The live model. `sizes` = [d, h_1, …, h_k, K]; k = 0 is logistic regression. K = 1 is the binary
 * victim (one logit, labels ±1, logistic loss); K ≥ 2 is a softmax head over classes 0 … K−1 with
 * cross-entropy (an ablation option, used with the 10-class samples).
 */
export interface Net {
  sizes: number[];
  activation: Activation;
  /** Every parameter, layer by layer: W_l (out × in, row-major), b_l, then γ_l, β_l if batch norm. */
  theta: Float64Array;
  norm: Norm;
  /**
   * Batch norm after every hidden linear layer (ablation option, off by default). Train-mode
   * passes normalise with the batch's own statistics — so a sample's gradient depends on the whole
   * batch — and eval-mode passes with the running statistics in `bnState`, as in PyTorch.
   */
  bn?: boolean;
  /** Running statistics per hidden layer: mean (out values) then variance (out values). */
  bnState?: Float64Array;
}

/**
 * The JSON-safe form kept on the clean model and on every TraceFrame: the parameters as base64
 * IEEE-754 bytes. A CIFAR-10 MLP has ~200k parameters, so frames carry them as float32 — a third
 * of the size of JSON numbers, and far below anything that moves a prediction.
 */
export interface PackedNet {
  sizes: number[];
  activation: Activation;
  norm: Norm;
  params: string;
  bits: 32 | 64;
  bn?: boolean;
  bnState?: string;   // base64 float64
}

/** A parameter-space vector, laid out like `theta`. */
export type Grad = Float64Array;

/** PyTorch's BatchNorm1d defaults. */
const BN_EPS = 1e-5;
const BN_MOMENTUM = 0.1;

// ── layout ──

interface Layer {
  in: number; out: number;
  W: number;          // offset of W_l in theta
  b: number;          // offset of b_l in theta
  g: number;          // offset of γ_l (−1 without batch norm; never on the output layer)
  beta: number;       // offset of β_l
  run: number;        // offset of this layer's running mean in bnState (variance follows)
  scale: number;      // factor on this layer's input: 1 on x̃ (already 1/√d), 1/√in on hidden units
}

const layoutCache = new Map<string, { layers: Layer[]; total: number; running: number }>();

function layout(sizes: number[], bn = false) {
  const key = sizes.join(',') + (bn ? ':bn' : '');
  let lay = layoutCache.get(key);
  if (!lay) {
    const layers: Layer[] = [];
    let off = 0, run = 0;
    for (let l = 0; l + 1 < sizes.length; l++) {
      const inp = sizes[l], out = sizes[l + 1];
      const hidden = l + 2 < sizes.length;
      const L: Layer = { in: inp, out, W: off, b: off + out * inp, g: -1, beta: -1, run: -1, scale: l === 0 ? 1 : 1 / Math.sqrt(inp) };
      off += out * inp + out;
      if (bn && hidden) {
        L.g = off; L.beta = off + out; off += 2 * out;
        L.run = run; run += 2 * out;
      }
      layers.push(L);
    }
    lay = { layers, total: off, running: run };
    layoutCache.set(key, lay);
  }
  return lay;
}

export function paramCount(sizes: number[], bn = false): number {
  return layout(sizes, bn).total;
}

/** Number of outputs: 1 for the binary victim, K for a softmax head. */
export function numOutputs(m: { sizes: number[] }): number {
  return m.sizes[m.sizes.length - 1];
}

/** "784 → 64 → 1, ReLU" */
export function describeNet(net: { sizes: number[]; activation: Activation; bn?: boolean }): string {
  return net.sizes.length <= 2
    ? `logistic regression, ${net.sizes[0]} inputs`
    : `${net.sizes.join(' → ')}, ${net.activation === 'relu' ? 'ReLU' : 'tanh'}${net.bn ? ', batch norm' : ''}`;
}

// ── activations: a = s·φ(z), with the next layer's input scale s folded in ──

function act(kind: Activation, z: number, s: number): number {
  return kind === 'relu' ? (z > 0 ? s * z : 0) : s * Math.tanh(z);
}
/** ∂a/∂z */
function act1(kind: Activation, z: number, s: number): number {
  if (kind === 'relu') return z > 0 ? s : 0;
  const t = Math.tanh(z);
  return s * (1 - t * t);
}
/** ∂²a/∂z² (zero almost everywhere for ReLU) */
function act2(kind: Activation, z: number, s: number): number {
  if (kind === 'relu') return 0;
  const t = Math.tanh(z);
  return -2 * s * t * (1 - t * t);
}

// ── loss ──

export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** ℓ = log(1 + e^{−m}) for the margin m = y·z, without overflow */
export function logisticLoss(margin: number): number {
  return margin > 0 ? Math.log1p(Math.exp(-margin)) : -margin + Math.log1p(Math.exp(margin));
}

/** Softmax of z[o … o+K) into p[o … o+K); returns log Σ e^z (for the cross-entropy). */
function softmaxInto(z: Float64Array, o: number, K: number, p: Float64Array): number {
  let mx = -Infinity;
  for (let k = 0; k < K; k++) mx = Math.max(mx, z[o + k]);
  let sum = 0;
  for (let k = 0; k < K; k++) { p[o + k] = Math.exp(z[o + k] - mx); sum += p[o + k]; }
  for (let k = 0; k < K; k++) p[o + k] /= sum;
  return mx + Math.log(sum);
}

// ── input normalisation ──

/** Per-channel mean/std of the training pixels (the reference model's μ, σ). */
export function fitNorm(X: number[][], channels = 1): Norm {
  const d = X[0].length;
  const sum = new Array(channels).fill(0), sq = new Array(channels).fill(0);
  for (const x of X) for (let j = 0; j < d; j++) { sum[j % channels] += x[j]; sq[j % channels] += x[j] * x[j]; }
  const count = (X.length * d) / channels;
  const mean = sum.map(s => s / count);
  const std = sq.map((s, c) => Math.sqrt(Math.max(s / count - mean[c] ** 2, 1e-12)));
  return { mean, std, channels, scale: 1 / Math.sqrt(d) };
}

const jacCache = new WeakMap<Norm, { mu: Float64Array; jac: Float64Array }>();

/** Per-pixel μ and ∂x̃_j/∂x_j = 1/(σ√d), so x̃_j = (x_j − μ_j)·jac_j. */
function pixelNorm(norm: Norm, d: number) {
  let c = jacCache.get(norm);
  if (!c || c.jac.length !== d) {
    const C = norm.channels;
    const mu = new Float64Array(d), jac = new Float64Array(d);
    for (let j = 0; j < d; j++) { mu[j] = norm.mean[j % C]; jac[j] = norm.scale / norm.std[j % C]; }
    c = { mu, jac };
    jacCache.set(norm, c);
  }
  return c;
}

/** ∂x̃_j/∂x_j — the chain-rule factor from normalised features back to pixels. */
export function inputJacobian(norm: Norm, d: number): Float64Array {
  return pixelNorm(norm, d).jac;
}

// ── construction, packing ──

function gaussian(): number {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

/**
 * A fresh victim. Logistic regression starts at θ = 0, as before. A hidden layer's weights are
 * drawn N(0, g²) — its input already has unit scale (x̃ has norm ≈ 1, hidden units carry 1/√in) —
 * with the He gain g² = 2 for ReLU and 1 for tanh; the output layer uses g = 1; biases start at 0.
 * Batch norm starts at γ = 1, β = 0, running mean 0 and variance 1 (PyTorch's defaults).
 */
export function createNet(sizes: number[], activation: Activation, norm: Norm, bn = false): Net {
  const useBn = bn && sizes.length > 2;
  const { layers, total, running } = layout(sizes, useBn);
  const theta = new Float64Array(total);
  if (layers.length > 1) {
    layers.forEach((L, l) => {
      const g = l === layers.length - 1 ? 1 : activation === 'relu' ? Math.SQRT2 : 1;
      for (let k = 0; k < L.out * L.in; k++) theta[L.W + k] = g * gaussian();
      if (L.g >= 0) for (let k = 0; k < L.out; k++) theta[L.g + k] = 1;
    });
  }
  const net: Net = { sizes: [...sizes], activation, theta, norm };
  if (useBn) {
    net.bn = true;
    net.bnState = new Float64Array(running);
    for (const L of layers) if (L.run >= 0) for (let k = 0; k < L.out; k++) net.bnState[L.run + L.out + k] = 1;
  }
  return net;
}

export function cloneModel(m: Net): Net {
  const c: Net = { sizes: m.sizes, activation: m.activation, theta: m.theta.slice(), norm: m.norm };
  if (m.bn) { c.bn = true; c.bnState = m.bnState!.slice(); }
  return c;
}

function toBase64(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000) as any);
  return btoa(s);
}

function fromBase64(b64: string): Uint8Array {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

const bytesOf = (a: Float64Array | Float32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

export function packNet(m: Net, bits: 32 | 64 = 64): PackedNet {
  const arr = bits === 64 ? m.theta : Float32Array.from(m.theta);
  const p: PackedNet = { sizes: m.sizes, activation: m.activation, norm: m.norm, bits, params: toBase64(bytesOf(arr)) };
  if (m.bn) { p.bn = true; p.bnState = toBase64(bytesOf(m.bnState!)); }
  return p;
}

const unpacked = new WeakMap<object, Net>();

/** The live model behind a packed one (decoded once per object, then cached). */
export function unpackNet(raw: Net | PackedNet): Net {
  if (!raw) throw new Error('No model');
  if ((raw as Net).theta) {
    const n = raw as Net;
    // A Net that crossed JSON has plain arrays for theta and the running statistics
    if (n.theta instanceof Float64Array && (!n.bn || n.bnState instanceof Float64Array)) return n;
    return { ...n, theta: Float64Array.from(n.theta as any), ...(n.bn ? { bnState: Float64Array.from(n.bnState as any) } : {}) };
  }
  let net = unpacked.get(raw);
  if (!net) {
    const p = raw as PackedNet;
    if (typeof p.params !== 'string' || !p.sizes) throw new Error('Not a Pang 2021 victim model — retrain the clean model');
    const bytes = fromBase64(p.params);
    const theta = p.bits === 32
      ? Float64Array.from(new Float32Array(bytes.buffer, 0, bytes.length / 4))
      : new Float64Array(bytes.buffer, 0, bytes.length / 8);
    if (theta.length !== paramCount(p.sizes, !!p.bn)) throw new Error('Packed model has the wrong number of parameters');
    net = { sizes: p.sizes, activation: p.activation, theta, norm: p.norm };
    if (p.bn) {
      const b = fromBase64(p.bnState!);
      net.bn = true;
      net.bnState = new Float64Array(b.buffer, 0, b.length / 8);
    }
    unpacked.set(raw, net);
  }
  return net;
}

// ── forward (eval mode: batch norm uses the running statistics) ──

/** All K outputs for one raw pixel vector. */
export function outputs(raw: Net | PackedNet, x: number[]): Float64Array {
  const m = unpackNet(raw);
  const { layers } = layout(m.sizes, !!m.bn);
  const d = m.sizes[0], th = m.theta, rs = m.bnState;
  const { mu, jac } = pixelNorm(m.norm, d);
  let a = new Float64Array(d);
  for (let j = 0; j < d; j++) a[j] = (x[j] - mu[j]) * jac[j];
  for (let l = 0; l < layers.length; l++) {
    const L = layers[l], last = l === layers.length - 1;
    const s = last ? 1 : layers[l + 1].scale;
    const next = new Float64Array(L.out);
    for (let k = 0; k < L.out; k++) {
      let z = th[L.b + k];
      const row = L.W + k * L.in;
      for (let j = 0; j < L.in; j++) z += th[row + j] * a[j];
      if (L.g >= 0) z = th[L.g + k] * (z - rs![L.run + k]) / Math.sqrt(rs![L.run + L.out + k] + BN_EPS) + th[L.beta + k];
      next[k] = last ? z : act(m.activation, z, s);
    }
    a = next;
  }
  return a;
}

/** z(x) of the binary victim (the first output of a softmax head). */
export function logit(raw: Net | PackedNet, x: number[]): number {
  return outputs(raw, x)[0];
}

export function predictProb(m: Net | PackedNet, x: number[]): number {
  return sigmoid(logit(m, x));
}

/** Accuracy and mean loss in one pass (logistic for K = 1, cross-entropy for a softmax head). */
export function evaluate(m: Net, X: number[][], y: number[]): { acc: number; loss: number } {
  const K = numOutputs(m);
  let correct = 0, loss = 0;
  const p = new Float64Array(K);
  for (let i = 0; i < X.length; i++) {
    if (K === 1) {
      const yi = y[i] > 0 ? 1 : -1;
      const z = logit(m, X[i]);
      if ((z >= 0 ? 1 : -1) === yi) correct++;
      loss += logisticLoss(yi * z);
    } else {
      const z = outputs(m, X[i]);
      const lse = softmaxInto(z, 0, K, p);
      let best = 0;
      for (let k = 1; k < K; k++) if (z[k] > z[best]) best = k;
      if (best === y[i]) correct++;
      loss += lse - z[y[i]];
    }
  }
  return X.length ? { acc: correct / X.length, loss: loss / X.length } : { acc: 0, loss: 0 };
}

export function computeAccuracy(m: Net, X: number[][], y: number[]): number {
  return evaluate(m, X, y).acc;
}

export function computeLoss(m: Net, X: number[][], y: number[]): number {
  return evaluate(m, X, y).loss;
}

// ── first and second derivatives of a batch ──

/** Batch-norm quantities of one hidden layer for a pass (batch or running statistics). */
interface BnCache {
  s: Float64Array;      // 1/√(var + ε) per unit
  c: Float64Array;      // z − μ (n × out)
  xhat: Float64Array;   // (z − μ)·s
  mean: Float64Array;   // μ used (batch or running)
  varB: Float64Array;   // batch variance (biased), for the running update
  train: boolean;
}

/**
 * One forward and backward pass over a batch, kept for the R-operator:
 *   A[l]  input of layer l (n × in_l), A[0] = x̃;   Zl[l] its linear output (n × out_l);
 *   Y[l]  the activation's input (Zl, or its batch-norm output);  D[l] = ∂L/∂Zl[l] (1/n included);
 *   DY[l] = ∂L/∂Y[l] for batch-norm layers;          P  σ(z) (K = 1) or softmax (n × K);
 *   grad = ∇_θ L(S; θ).
 * `mode` 'train' normalises with the batch's statistics (the reference code's crafting mode), 'eval'
 * with the running ones; without batch norm the two are the same.
 */
export interface BatchStats {
  net: Net;
  n: number;
  y: number[];
  A: Float64Array[];
  Zl: Float64Array[];
  Y: Float64Array[];
  bn: (BnCache | null)[];
  D: Float64Array[];
  DY: (Float64Array | null)[];
  P: Float64Array;
  grad: Grad;
}

export function batchStats(m: Net, X: number[][], y: number[], mode: 'train' | 'eval' = 'train'): BatchStats {
  const { layers, total } = layout(m.sizes, !!m.bn);
  const n = X.length, d = m.sizes[0], th = m.theta, f = m.activation;
  const nL = layers.length, K = numOutputs(m);
  const { mu, jac } = pixelNorm(m.norm, d);

  const A: Float64Array[] = [new Float64Array(n * d)];
  for (let i = 0; i < n; i++) {
    const x = X[i], o = i * d;
    for (let j = 0; j < d; j++) A[0][o + j] = (x[j] - mu[j]) * jac[j];
  }
  const Zl: Float64Array[] = [], Y: Float64Array[] = [], bn: (BnCache | null)[] = [];
  for (let l = 0; l < nL; l++) {
    const L = layers[l], a = A[l], z = new Float64Array(n * L.out);
    for (let i = 0; i < n; i++) {
      const ai = i * L.in;
      for (let k = 0; k < L.out; k++) {
        let s = th[L.b + k];
        const row = L.W + k * L.in;
        for (let j = 0; j < L.in; j++) s += th[row + j] * a[ai + j];
        z[i * L.out + k] = s;
      }
    }
    Zl.push(z);
    let yv = z, cache: BnCache | null = null;
    if (L.g >= 0) {
      const O = L.out, train = mode === 'train';
      const mean = new Float64Array(O), varB = new Float64Array(O), s = new Float64Array(O);
      for (let i = 0; i < n; i++) for (let k = 0; k < O; k++) mean[k] += z[i * O + k] / n;
      for (let i = 0; i < n; i++) for (let k = 0; k < O; k++) varB[k] += (z[i * O + k] - mean[k]) ** 2 / n;
      const useMean = train ? mean : m.bnState!.subarray(L.run, L.run + O);
      for (let k = 0; k < O; k++) s[k] = 1 / Math.sqrt((train ? varB[k] : m.bnState![L.run + O + k]) + BN_EPS);
      const c = new Float64Array(n * O), xhat = new Float64Array(n * O);
      yv = new Float64Array(n * O);
      for (let i = 0; i < n; i++) for (let k = 0; k < O; k++) {
        const q = i * O + k;
        c[q] = z[q] - useMean[k];
        xhat[q] = c[q] * s[k];
        yv[q] = th[L.g + k] * xhat[q] + th[L.beta + k];
      }
      cache = { s, c, xhat, mean: Float64Array.from(useMean), varB, train };
    }
    bn.push(cache);
    Y.push(yv);
    if (l < nL - 1) {
      const sc = layers[l + 1].scale, next = new Float64Array(yv.length);
      for (let q = 0; q < yv.length; q++) next[q] = act(f, yv[q], sc);
      A.push(next);
    }
  }

  // Output: D[L] = ∂L/∂z = (p − onehot)/n (K = 1: σ(z) − 1[y = +1])
  const zOut = Zl[nL - 1];
  const P = new Float64Array(n * K), Dout = new Float64Array(n * K);
  for (let i = 0; i < n; i++) {
    if (K === 1) {
      P[i] = sigmoid(zOut[i]);
      Dout[i] = (P[i] - (y[i] > 0 ? 1 : 0)) / n;
    } else {
      softmaxInto(zOut, i * K, K, P);
      for (let k = 0; k < K; k++) Dout[i * K + k] = (P[i * K + k] - (k === y[i] ? 1 : 0)) / n;
    }
  }

  const grad = new Float64Array(total);
  const D: Float64Array[] = new Array(nL), DY: (Float64Array | null)[] = new Array(nL).fill(null);
  D[nL - 1] = Dout;
  for (let l = nL - 1; l >= 0; l--) {
    const L = layers[l], a = A[l], dl = D[l];
    for (let i = 0; i < n; i++) {
      const ai = i * L.in;
      for (let k = 0; k < L.out; k++) {
        const dk = dl[i * L.out + k];
        if (dk === 0) continue;
        grad[L.b + k] += dk;
        const row = L.W + k * L.in;
        for (let j = 0; j < L.in; j++) grad[row + j] += dk * a[ai + j];
      }
    }
    if (l > 0) {
      // ∂L/∂Y[l−1] = a′(Y[l−1]) ⊙ W_lᵀ D[l]
      const P1 = layers[l - 1], prev = new Float64Array(n * L.in), yp = Y[l - 1];
      for (let i = 0; i < n; i++) {
        const pi = i * L.in;
        for (let k = 0; k < L.out; k++) {
          const dk = dl[i * L.out + k];
          if (dk === 0) continue;
          const row = L.W + k * L.in;
          for (let j = 0; j < L.in; j++) prev[pi + j] += th[row + j] * dk;
        }
        for (let j = 0; j < L.in; j++) prev[pi + j] *= act1(f, yp[pi + j], L.scale);
      }
      const cache = bn[l - 1];
      if (!cache) { D[l - 1] = prev; continue; }
      // Batch norm: ∇γ = Σ dy·x̂, ∇β = Σ dy, and ∂L/∂z through the (batch or running) statistics
      DY[l - 1] = prev;
      D[l - 1] = bnBackward(P1, th, cache, prev, n, grad);
    }
  }
  return { net: m, n, y, A, Zl, Y, bn, D, DY, P, grad };
}

/**
 * Batch-norm backward. With dx̂ = dy·γ: train mode (batch statistics)
 *   ∂L/∂z = s (dx̂ − mean(dx̂) − x̂ · mean(dx̂ x̂)),
 * eval mode (running statistics, constants) ∂L/∂z = s·dx̂. Accumulates ∇γ, ∇β into `grad`.
 */
function bnBackward(L: Layer, th: Float64Array, c: BnCache, dy: Float64Array, n: number, grad: Float64Array): Float64Array {
  const O = L.out, out = new Float64Array(n * O);
  for (let k = 0; k < O; k++) {
    const g = th[L.g + k];
    let m1 = 0, m2 = 0, dg = 0, db = 0;
    for (let i = 0; i < n; i++) {
      const q = i * O + k, dx = dy[q] * g;
      dg += dy[q] * c.xhat[q]; db += dy[q];
      m1 += dx / n; m2 += (dx * c.xhat[q]) / n;
    }
    grad[L.g + k] += dg; grad[L.beta + k] += db;
    for (let i = 0; i < n; i++) {
      const q = i * O + k, dx = dy[q] * g;
      out[q] = c.train ? c.s[k] * (dx - m1 - c.xhat[q] * m2) : c.s[k] * dx;
    }
  }
  return out;
}

/** Accuracy and mean loss of the batch, read off the forward pass already done. */
export function statsEval(s: BatchStats, y: number[]): { acc: number; loss: number } {
  const z = s.Zl[s.Zl.length - 1], K = numOutputs(s.net);
  let correct = 0, loss = 0;
  for (let i = 0; i < s.n; i++) {
    if (K === 1) {
      const yi = y[i] > 0 ? 1 : -1;
      if ((z[i] >= 0 ? 1 : -1) === yi) correct++;
      loss += logisticLoss(yi * z[i]);
    } else {
      let best = 0;
      for (let k = 1; k < K; k++) if (z[i * K + k] > z[i * K + best]) best = k;
      if (best === y[i]) correct++;
      loss += -Math.log(Math.max(s.P[i * K + y[i]], 1e-300));
    }
  }
  return s.n ? { acc: correct / s.n, loss: loss / s.n } : { acc: 0, loss: 0 };
}

/** ∇_θ L(S; θ) */
export function gradFromStats(s: BatchStats): Grad {
  return s.grad;
}

export function lossGradient(m: Net, X: number[][], y: number[]): Grad {
  return batchStats(m, X, y).grad;
}

/**
 * The running statistics after this train-mode pass (PyTorch: r ← (1 − 0.1) r + 0.1 · batch, the
 * variance unbiased). Returns the same model when there is no batch norm.
 */
export function updateRunningStats(m: Net, s: BatchStats): Net {
  if (!m.bn) return m;
  const { layers } = layout(m.sizes, true);
  const next = { ...m, bnState: m.bnState!.slice() };
  layers.forEach((L, l) => {
    const c = s.bn[l];
    if (!c || !c.train) return;
    const unbias = s.n > 1 ? s.n / (s.n - 1) : 1;
    for (let k = 0; k < L.out; k++) {
      next.bnState[L.run + k] = (1 - BN_MOMENTUM) * next.bnState[L.run + k] + BN_MOMENTUM * c.mean[k];
      next.bnState[L.run + L.out + k] = (1 - BN_MOMENTUM) * next.bnState[L.run + L.out + k] + BN_MOMENTUM * c.varB[k] * unbias;
    }
  });
  return next;
}

/**
 * Pearlmutter's R-operator in direction v on a batch: returns
 *   Hv  = ∇²_θ L(S; θ) · v, and, when `input` is set,
 *   dXt = R_v{∂L/∂x̃_i} = ∂/∂x̃_i [vᵀ ∇_θ L(S; θ)]   (n × d, normalised-input units).
 * Forward:  R{A[0]} = 0,  R{Zl[l]} = W_l R{A[l]} + V_l A[l] + c_l,
 *           batch norm (train): R{μ} = mean R{z}, R{c} = R{z} − R{μ}, R{var} = 2 mean(c R{c}),
 *             R{s} = −½ s³ R{var}, R{x̂} = s R{c} + c R{s}, R{Y} = R{γ} x̂ + γ R{x̂} + R{β}
 *             (eval: R{x̂} = s R{z});
 *           R{A[l+1]} = a′(Y[l]) ⊙ R{Y[l]}.
 * Output:   R{D[L]} = h ⊙ R{z}/n (K = 1), (p ⊙ R{z} − p (pᵀR{z}))/n (softmax).
 * Backward: R{∇W_l} = R{D[l]} A[l]ᵀ + D[l] R{A[l]}ᵀ,   R{∇b_l} = R{D[l]},
 *           R{DY} = a″(Y) ⊙ R{Y} ⊙ W_lᵀD[l] + a′(Y) ⊙ (V_lᵀD[l] + W_lᵀR{D[l]}),
 *           R{∇γ} = Σ (R{DY} x̂ + DY R{x̂}), R{∇β} = Σ R{DY}, and R{D} through the batch-norm
 *           backward (bnRBackward); R{∂L/∂x̃} = V_1ᵀD[1] + W_1ᵀR{D[1]}.
 * For logistic regression this reduces to the closed forms (1/n) Σ h_i (vᵀx̃_i) x̃_i and
 * (1/n)[h_i (v_wᵀx̃_i + v_b) w + r_i v_w].
 */
export function rop(s: BatchStats, v: Grad, input = false): { Hv: Grad; dXt?: Float64Array } {
  const m = s.net, { layers, total } = layout(m.sizes, !!m.bn);
  const { n, A, Y, D, DY, P } = s;
  const th = m.theta, f = m.activation, nL = layers.length, K = numOutputs(m);

  // Forward
  const RA: (Float64Array | null)[] = [null];
  const RY: Float64Array[] = [], RX: (Float64Array | null)[] = [], RS: (Float64Array | null)[] = [];
  for (let l = 0; l < nL; l++) {
    const L = layers[l], a = A[l], ra = RA[l], rz = new Float64Array(n * L.out);
    for (let i = 0; i < n; i++) {
      const ai = i * L.in;
      for (let k = 0; k < L.out; k++) {
        let t = v[L.b + k];
        const row = L.W + k * L.in;
        for (let j = 0; j < L.in; j++) t += v[row + j] * a[ai + j];
        if (ra) for (let j = 0; j < L.in; j++) t += th[row + j] * ra[ai + j];
        rz[i * L.out + k] = t;
      }
    }
    let ry = rz, rx: Float64Array | null = null, rs: Float64Array | null = null;
    const c = s.bn[l];
    if (c) {
      const O = L.out;
      rx = new Float64Array(n * O); ry = new Float64Array(n * O);
      if (c.train) {
        rs = new Float64Array(O);
        for (let k = 0; k < O; k++) {
          let rmu = 0;
          for (let i = 0; i < n; i++) rmu += rz[i * O + k] / n;
          let rvar = 0;
          for (let i = 0; i < n; i++) rvar += (2 * c.c[i * O + k] * (rz[i * O + k] - rmu)) / n;
          rs[k] = -0.5 * c.s[k] ** 3 * rvar;
          for (let i = 0; i < n; i++) {
            const q = i * O + k;
            rx[q] = c.s[k] * (rz[q] - rmu) + c.c[q] * rs[k];
          }
        }
      } else {
        for (let q = 0; q < n * O; q++) rx[q] = c.s[q % O] * rz[q];
      }
      for (let i = 0; i < n; i++) for (let k = 0; k < O; k++) {
        const q = i * O + k;
        ry[q] = v[L.g + k] * c.xhat[q] + th[L.g + k] * rx[q] + v[L.beta + k];
      }
    }
    RY.push(ry); RX.push(rx); RS.push(rs);
    if (l < nL - 1) {
      const sc = layers[l + 1].scale, yv = Y[l], next = new Float64Array(ry.length);
      for (let q = 0; q < ry.length; q++) next[q] = act1(f, yv[q], sc) * ry[q];
      RA.push(next);
    }
  }

  // Output
  const rzOut = RY[nL - 1];
  let RD: Float64Array = new Float64Array(n * K);
  for (let i = 0; i < n; i++) {
    if (K === 1) {
      RD[i] = (P[i] * (1 - P[i]) * rzOut[i]) / n;
    } else {
      let pr = 0;
      for (let k = 0; k < K; k++) pr += P[i * K + k] * rzOut[i * K + k];
      for (let k = 0; k < K; k++) RD[i * K + k] = (P[i * K + k] * (rzOut[i * K + k] - pr)) / n;
    }
  }

  // Backward
  const Hv = new Float64Array(total);
  let dXt: Float64Array | undefined;
  for (let l = nL - 1; l >= 0; l--) {
    const L = layers[l], a = A[l], ra = RA[l], dl = D[l];
    for (let i = 0; i < n; i++) {
      const ai = i * L.in;
      for (let k = 0; k < L.out; k++) {
        const rd = RD[i * L.out + k], dk = dl[i * L.out + k];
        Hv[L.b + k] += rd;
        const row = L.W + k * L.in;
        if (rd !== 0) for (let j = 0; j < L.in; j++) Hv[row + j] += rd * a[ai + j];
        if (ra && dk !== 0) for (let j = 0; j < L.in; j++) Hv[row + j] += dk * ra[ai + j];
      }
    }
    if (l > 0 || input) {
      // back = W_lᵀD[l] (only needed below the first layer), mixed = V_lᵀD[l] + W_lᵀR{D[l]}
      const out = new Float64Array(n * L.in);
      const back = l > 0 ? new Float64Array(L.in) : null;
      for (let i = 0; i < n; i++) {
        const pi = i * L.in;
        if (back) back.fill(0);
        for (let k = 0; k < L.out; k++) {
          const dk = dl[i * L.out + k], rd = RD[i * L.out + k];
          const row = L.W + k * L.in;
          if (dk !== 0) {
            for (let j = 0; j < L.in; j++) out[pi + j] += v[row + j] * dk;
            if (back) for (let j = 0; j < L.in; j++) back[j] += th[row + j] * dk;
          }
          if (rd !== 0) for (let j = 0; j < L.in; j++) out[pi + j] += th[row + j] * rd;
        }
        if (back) {
          const yp = Y[l - 1], ryp = RY[l - 1];
          for (let j = 0; j < L.in; j++) {
            const q = pi + j;
            out[q] = act2(f, yp[q], L.scale) * ryp[q] * back[j] + act1(f, yp[q], L.scale) * out[q];
          }
        }
      }
      if (l === 0) { dXt = out; break; }
      const c = s.bn[l - 1];
      RD = c ? bnRBackward(layers[l - 1], th, v, c, DY[l - 1]!, out, RX[l - 1]!, RS[l - 1], n, Hv) : out;
    }
  }
  return { Hv, dXt };
}

/**
 * R-operator of the batch-norm backward, given DY = ∂L/∂Y and R{DY}:
 *   R{∇γ} += Σ (R{DY} x̂ + DY R{x̂}),  R{∇β} += Σ R{DY};  dx̂ = DY γ,  R{dx̂} = R{DY} γ + DY R{γ};
 *   train: R{D} = R{s}(dx̂ − m₁ − x̂ m₂) + s(R{dx̂} − R{m₁} − R{x̂} m₂ − x̂ R{m₂}),
 *          m₁ = mean dx̂, m₂ = mean dx̂ x̂, R{m₁} = mean R{dx̂}, R{m₂} = mean(R{dx̂} x̂ + dx̂ R{x̂});
 *   eval:  R{D} = s R{dx̂}.
 */
function bnRBackward(
  L: Layer, th: Float64Array, v: Grad, c: BnCache, dy: Float64Array, rdy: Float64Array,
  rx: Float64Array, rs: Float64Array | null, n: number, Hv: Float64Array,
): Float64Array {
  const O = L.out, out = new Float64Array(n * O);
  for (let k = 0; k < O; k++) {
    const g = th[L.g + k], rg = v[L.g + k];
    let hg = 0, hb = 0, m1 = 0, m2 = 0, rm1 = 0, rm2 = 0;
    for (let i = 0; i < n; i++) {
      const q = i * O + k;
      hg += rdy[q] * c.xhat[q] + dy[q] * rx[q];
      hb += rdy[q];
      const dx = dy[q] * g, rdx = rdy[q] * g + dy[q] * rg;
      m1 += dx / n; m2 += (dx * c.xhat[q]) / n;
      rm1 += rdx / n; rm2 += (rdx * c.xhat[q] + dx * rx[q]) / n;
    }
    Hv[L.g + k] += hg; Hv[L.beta + k] += hb;
    for (let i = 0; i < n; i++) {
      const q = i * O + k, dx = dy[q] * g, rdx = rdy[q] * g + dy[q] * rg;
      out[q] = c.train
        ? rs![k] * (dx - m1 - c.xhat[q] * m2) + c.s[k] * (rdx - rm1 - rx[q] * m2 - c.xhat[q] * rm2)
        : c.s[k] * rdx;
    }
  }
  return out;
}

/** ∇²_θ L(S; θ) · v */
export function hessVec(s: BatchStats, v: Grad): Grad {
  return rop(s, v).Hv;
}

/**
 * ∇_{x_i} [qᵀ ∇_θ L(S; θ)] for every image of S, with q held fixed — the step from a
 * parameter-space objective back to the pixels (the R-operator's input output, × ∂x̃/∂x).
 */
export function pixelGradient(s: BatchStats, q: Grad): number[][] {
  const d = s.net.sizes[0];
  const dXt = rop(s, q, true).dXt!;
  const jac = inputJacobian(s.net.norm, d);
  return Array.from({ length: s.n }, (_, i) => {
    const out = new Array(d);
    for (let j = 0; j < d; j++) out[j] = dXt[i * d + j] * jac[j];
    return out;
  });
}

// ── training ──

/** θ ← θ − β g */
export function applyGradient(m: Net, g: Grad, lr: number): Net {
  const next = cloneModel(m);
  for (let j = 0; j < next.theta.length; j++) next.theta[j] -= lr * g[j];
  return next;
}

/** One online update on a batch — Eq. (3): θ_{t+1} = θ_t − β ∇_θ L(S_t; θ_t) */
export function sgdStep(m: Net, X: number[][], y: number[], lr: number): Net {
  return applyGradient(m, lossGradient(m, X, y), lr);
}

/**
 * The victim's optimiser, as in the reference code: PyTorch's SGD with momentum μ and weight
 * decay λ_wd. The paper trains with μ = 0.9 (and λ_wd = 1e-4 during burn-in); μ = 0 is Eq. 3.
 */
export interface Optim { lr: number; momentum: number; weightDecay: number }

/** The model and its momentum buffer v (null before the first step, as in a fresh optimiser). */
export interface Learner { net: Net; v: Grad | null }

/**
 * One step of torch.optim.SGD on the gradient g:
 *   d = g + λ_wd θ,   v ← μ v + d  (v ← d on a fresh optimiser),   θ ← θ − β v.
 * `momentum` overrides the optimiser's μ for this step only — the paper's "weight momentum"
 * trick raises it to 1.1 during the accumulative phase.
 */
export function optimStep(l: Learner, g: Grad, opt: Optim, momentum = opt.momentum): Learner {
  const th = l.net.theta, n = th.length;
  const v = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const d = g[j] + opt.weightDecay * th[j];
    v[j] = l.v ? momentum * l.v[j] + d : d;
  }
  return { net: applyGradient(l.net, v, opt.lr), v };
}

/** The update direction optimStep would take (θ_{t+1} = θ_t − β·direction), without taking it. */
export function stepDirection(l: Learner, g: Grad, opt: Optim, momentum = opt.momentum): Grad {
  const th = l.net.theta, out = new Float64Array(th.length);
  for (let j = 0; j < th.length; j++) {
    const d = g[j] + opt.weightDecay * th[j];
    out[j] = l.v ? momentum * l.v[j] + d : d;
  }
  return out;
}

/** One step on a batch: the train-mode forward also updates batch norm's running statistics. */
export function learnerStep(l: Learner, X: number[][], y: number[], opt: Optim, momentum?: number): Learner {
  const s = batchStats(l.net, X, y);
  return optimStep({ net: updateRunningStats(l.net, s), v: l.v }, s.grad, opt, momentum);
}

/** Split a set into consecutive mini-batches of size B (the last may be smaller). */
export function toBatches(X: number[][], y: number[], B: number, order?: number[]) {
  const idx = order ?? X.map((_, i) => i);
  const out: { X: number[][]; y: number[]; idx: number[] }[] = [];
  for (let i = 0; i < idx.length; i += B) {
    const part = idx.slice(i, i + B);
    out.push({ X: part.map(k => X[k]), y: part.map(k => y[k]), idx: part });
  }
  return out;
}

export function shuffled(n: number): number[] {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * The burn-in phase of §4: `epochs` passes of online SGD (batch size B, the victim's optimiser)
 * over the clean training data, from a fresh victim. The accumulative phase starts from the
 * result, θ_0, with a fresh optimiser (the reference code builds a new one, momentum buffer empty).
 */
export function burnIn(X: number[][], y: number[], init: Net, opt: Optim, epochs: number, B: number): Net {
  let l: Learner = { net: init, v: null };
  for (let e = 0; e < epochs; e++) {
    for (const batch of toBatches(X, y, B, shuffled(X.length))) l = learnerStep(l, batch.X, batch.y, opt);
  }
  return l.net;
}

/** ‖θ − θ'‖₂ */
export function paramDistance(a: Net, b: Net): number {
  let s = 0;
  for (let j = 0; j < a.theta.length; j++) s += (a.theta[j] - b.theta[j]) ** 2;
  return Math.sqrt(s);
}

/**
 * Display-friendly state (the fields the shared panels read). `w` and `b` are the output layer —
 * the weights of the logit on the last hidden layer (on x̃ itself for logistic regression) — and
 * `wNorm` is the norm of every weight matrix together, biases excluded (‖w‖ for logistic
 * regression, the Frobenius norm of W_1 … W_L for the MLP).
 */
export function getModelState(
  m: Net, trainX: number[][], trainY: number[], testX: number[][], testY: number[],
  testEval?: { acc: number; loss: number },       // when the caller already evaluated m on the test set
) {
  const { layers } = layout(m.sizes, !!m.bn);
  const out = layers[layers.length - 1];
  let wsq = 0;
  for (const L of layers) for (let k = 0; k < L.out * L.in; k++) wsq += m.theta[L.W + k] ** 2;
  const tr = evaluate(m, trainX, trainY), te = testEval ?? evaluate(m, testX, testY);
  return {
    w: Array.from(m.theta.subarray(out.W, out.W + out.in)),
    wNorm: Math.sqrt(wsq),
    b: m.theta[out.b],
    architecture: describeNet(m),
    numParams: m.theta.length,
    trainAccuracy: tr.acc,
    testAccuracy: te.acc,
    trainLoss: tr.loss,
    testLoss: te.loss,
    hingeLoss: te.loss,        // shown where the SVM panels show hinge loss
    numSupportVectors: 0,
  };
}
