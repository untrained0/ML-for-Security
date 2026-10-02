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

/** The live model. `sizes` = [d, h_1, …, h_k, 1]; k = 0 is logistic regression. */
export interface Net {
  sizes: number[];
  activation: Activation;
  /** Every parameter, layer by layer: W_l (out × in, row-major), then b_l. */
  theta: Float64Array;
  norm: Norm;
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
}

/** A parameter-space vector, laid out like `theta`. */
export type Grad = Float64Array;

// ── layout ──

interface Layer {
  in: number; out: number;
  W: number;          // offset of W_l in theta
  b: number;          // offset of b_l in theta
  scale: number;      // factor on this layer's input: 1 on x̃ (already 1/√d), 1/√in on hidden units
}

const layoutCache = new Map<string, { layers: Layer[]; total: number }>();

function layout(sizes: number[]) {
  const key = sizes.join(',');
  let lay = layoutCache.get(key);
  if (!lay) {
    const layers: Layer[] = [];
    let off = 0;
    for (let l = 0; l + 1 < sizes.length; l++) {
      const inp = sizes[l], out = sizes[l + 1];
      layers.push({ in: inp, out, W: off, b: off + out * inp, scale: l === 0 ? 1 : 1 / Math.sqrt(inp) });
      off += out * inp + out;
    }
    lay = { layers, total: off };
    layoutCache.set(key, lay);
  }
  return lay;
}

export function paramCount(sizes: number[]): number {
  return layout(sizes).total;
}

/** "784 → 64 → 1, ReLU" */
export function describeNet(net: { sizes: number[]; activation: Activation }): string {
  return net.sizes.length <= 2
    ? `logistic regression, ${net.sizes[0]} inputs`
    : `${net.sizes.join(' → ')}, ${net.activation === 'relu' ? 'ReLU' : 'tanh'}`;
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
 */
export function createNet(sizes: number[], activation: Activation, norm: Norm): Net {
  const { layers, total } = layout(sizes);
  const theta = new Float64Array(total);
  if (layers.length > 1) {
    layers.forEach((L, l) => {
      const g = l === layers.length - 1 ? 1 : activation === 'relu' ? Math.SQRT2 : 1;
      for (let k = 0; k < L.out * L.in; k++) theta[L.W + k] = g * gaussian();
    });
  }
  return { sizes: [...sizes], activation, theta, norm };
}

export function cloneModel(m: Net): Net {
  return { sizes: m.sizes, activation: m.activation, theta: m.theta.slice(), norm: m.norm };
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

export function packNet(m: Net, bits: 32 | 64 = 64): PackedNet {
  const arr = bits === 64 ? m.theta : Float32Array.from(m.theta);
  return {
    sizes: m.sizes, activation: m.activation, norm: m.norm, bits,
    params: toBase64(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)),
  };
}

const unpacked = new WeakMap<object, Net>();

/** The live model behind a packed one (decoded once per object, then cached). */
export function unpackNet(raw: Net | PackedNet): Net {
  if (!raw) throw new Error('No model');
  if ((raw as Net).theta) {
    const n = raw as Net;
    // A Net that crossed JSON has a plain array for theta
    return n.theta instanceof Float64Array ? n : { ...n, theta: Float64Array.from(n.theta as any) };
  }
  let net = unpacked.get(raw);
  if (!net) {
    const p = raw as PackedNet;
    if (typeof p.params !== 'string' || !p.sizes) throw new Error('Not a Pang 2021 victim model — retrain the clean model');
    const bytes = fromBase64(p.params);
    const theta = p.bits === 32
      ? Float64Array.from(new Float32Array(bytes.buffer, 0, bytes.length / 4))
      : new Float64Array(bytes.buffer, 0, bytes.length / 8);
    if (theta.length !== paramCount(p.sizes)) throw new Error('Packed model has the wrong number of parameters');
    net = { sizes: p.sizes, activation: p.activation, theta, norm: p.norm };
    unpacked.set(raw, net);
  }
  return net;
}

// ── forward ──

/** z(x) for one raw pixel vector. */
export function logit(raw: Net | PackedNet, x: number[]): number {
  const m = unpackNet(raw);
  const { layers } = layout(m.sizes);
  const d = m.sizes[0], th = m.theta;
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
      next[k] = last ? z : act(m.activation, z, s);
    }
    a = next;
  }
  return a[0];
}

export function predictProb(m: Net | PackedNet, x: number[]): number {
  return sigmoid(logit(m, x));
}

/** Accuracy and mean logistic loss in one pass. */
export function evaluate(m: Net, X: number[][], y: number[]): { acc: number; loss: number } {
  let correct = 0, loss = 0;
  for (let i = 0; i < X.length; i++) {
    const yi = y[i] > 0 ? 1 : -1;
    const z = logit(m, X[i]);
    if ((z >= 0 ? 1 : -1) === yi) correct++;
    loss += logisticLoss(yi * z);
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

/**
 * One forward and backward pass over a batch, kept for the R-operator:
 *   A[l]  input of layer l (n × in_l), A[0] = x̃;   Z[l]  its pre-activation (n × out_l);
 *   D[l]  = ∂L/∂Z[l] (the 1/n included);            r_i = ∂ℓ/∂z = σ(z_i) − 1[y_i = +1],
 *   h_i = ∂²ℓ/∂z² = σ(z_i)(1 − σ(z_i));             grad = ∇_θ L(S; θ).
 */
export interface BatchStats {
  net: Net;
  n: number;
  A: Float64Array[];
  Z: Float64Array[];
  D: Float64Array[];
  r: Float64Array;
  h: Float64Array;
  grad: Grad;
}

export function batchStats(m: Net, X: number[][], y: number[]): BatchStats {
  const { layers, total } = layout(m.sizes);
  const n = X.length, d = m.sizes[0], th = m.theta, f = m.activation;
  const nL = layers.length;
  const { mu, jac } = pixelNorm(m.norm, d);

  const A: Float64Array[] = [new Float64Array(n * d)];
  for (let i = 0; i < n; i++) {
    const x = X[i], o = i * d;
    for (let j = 0; j < d; j++) A[0][o + j] = (x[j] - mu[j]) * jac[j];
  }
  const Z: Float64Array[] = [];
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
    Z.push(z);
    if (l < nL - 1) {
      const sc = layers[l + 1].scale, next = new Float64Array(z.length);
      for (let q = 0; q < z.length; q++) next[q] = act(f, z[q], sc);
      A.push(next);
    }
  }

  const zOut = Z[nL - 1];
  const r = new Float64Array(n), h = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = sigmoid(zOut[i]);
    r[i] = p - (y[i] > 0 ? 1 : 0);
    h[i] = p * (1 - p);
  }

  const grad = new Float64Array(total);
  const D: Float64Array[] = new Array(nL);
  D[nL - 1] = r.map(v => v / n);
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
      // D[l−1] = a′(Z[l−1]) ⊙ W_lᵀ D[l]
      const prev = new Float64Array(n * L.in), zp = Z[l - 1];
      for (let i = 0; i < n; i++) {
        const pi = i * L.in;
        for (let k = 0; k < L.out; k++) {
          const dk = dl[i * L.out + k];
          if (dk === 0) continue;
          const row = L.W + k * L.in;
          for (let j = 0; j < L.in; j++) prev[pi + j] += th[row + j] * dk;
        }
        for (let j = 0; j < L.in; j++) prev[pi + j] *= act1(f, zp[pi + j], L.scale);
      }
      D[l - 1] = prev;
    }
  }
  return { net: m, n, A, Z, D, r, h, grad };
}

/** Accuracy and mean logistic loss of the batch, read off the forward pass already done. */
export function statsEval(s: BatchStats, y: number[]): { acc: number; loss: number } {
  const z = s.Z[s.Z.length - 1];
  let correct = 0, loss = 0;
  for (let i = 0; i < s.n; i++) {
    const yi = y[i] > 0 ? 1 : -1;
    if ((z[i] >= 0 ? 1 : -1) === yi) correct++;
    loss += logisticLoss(yi * z[i]);
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
 * Pearlmutter's R-operator in direction v on a batch: returns
 *   Hv  = ∇²_θ L(S; θ) · v, and, when `input` is set,
 *   dXt = R_v{∂L/∂x̃_i} = ∂/∂x̃_i [vᵀ ∇_θ L(S; θ)]   (n × d, normalised-input units).
 * Forward:  R{A[0]} = 0,  R{Z[l]} = W_l R{A[l]} + V_l A[l] + c_l,  R{A[l+1]} = a′(Z[l]) ⊙ R{Z[l]}.
 * Output:   R{D[L]} = h ⊙ R{z} / n.
 * Backward: R{∇W_l} = R{D[l]} A[l]ᵀ + D[l] R{A[l]}ᵀ,   R{∇b_l} = R{D[l]},
 *           R{D[l−1]} = a″(Z[l−1]) ⊙ R{Z[l−1]} ⊙ W_lᵀD[l] + a′(Z[l−1]) ⊙ (V_lᵀD[l] + W_lᵀR{D[l]}),
 *           R{∂L/∂x̃} = V_1ᵀD[1] + W_1ᵀR{D[1]}.
 * For logistic regression this reduces to the closed forms (1/n) Σ h_i (vᵀx̃_i) x̃_i and
 * (1/n)[h_i (v_wᵀx̃_i + v_b) w + r_i v_w].
 */
export function rop(s: BatchStats, v: Grad, input = false): { Hv: Grad; dXt?: Float64Array } {
  const m = s.net, { layers, total } = layout(m.sizes);
  const { n, A, Z, D, h } = s;
  const th = m.theta, f = m.activation, nL = layers.length;

  // Forward
  const RA: (Float64Array | null)[] = [null];
  const RZ: Float64Array[] = [];
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
    RZ.push(rz);
    if (l < nL - 1) {
      const sc = layers[l + 1].scale, z = Z[l], next = new Float64Array(rz.length);
      for (let q = 0; q < rz.length; q++) next[q] = act1(f, z[q], sc) * rz[q];
      RA.push(next);
    }
  }

  // Backward
  const Hv = new Float64Array(total);
  let RD = new Float64Array(n);
  for (let i = 0; i < n; i++) RD[i] = (h[i] * RZ[nL - 1][i]) / n;
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
          const zp = Z[l - 1], rzp = RZ[l - 1];
          for (let j = 0; j < L.in; j++) {
            const q = pi + j;
            out[q] = act2(f, zp[q], L.scale) * rzp[q] * back[j] + act1(f, zp[q], L.scale) * out[q];
          }
        }
      }
      if (l > 0) RD = out;
      else dXt = out;
    }
  }
  return { Hv, dXt };
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

export function learnerStep(l: Learner, X: number[][], y: number[], opt: Optim, momentum?: number): Learner {
  return optimStep(l, lossGradient(l.net, X, y), opt, momentum);
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
  const { layers } = layout(m.sizes);
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
