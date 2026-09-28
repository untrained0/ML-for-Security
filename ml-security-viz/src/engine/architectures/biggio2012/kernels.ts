/**
 * kernels.js — Kernel functions for SVM
 */

import { dot, dist2 } from '../../linalg';

/** Linear kernel: K(x,y) = x·y */
export function linearKernel(x, y) { return dot(x, y); }

/** RBF kernel: K(x,y) = exp(-γ||x-y||²) */
export function rbfKernel(x, y, gamma) { return Math.exp(-gamma * dist2(x, y)); }

/** Polynomial kernel: K(x,y) = (x·y + c)^d */
export function polyKernel(x, y, degree, coef0) { return Math.pow(dot(x, y) + coef0, degree); }

/** Factory: create kernel function from config */
export function createKernel({ type, gamma = 1, degree = 3, coef0 = 1 }) {
  switch (type) {
    case 'linear': {
      // Tagged so predict() can use the primal w·x + b instead of summing over support vectors.
      const k: any = (x, y) => linearKernel(x, y);
      k.linear = true;
      return k;
    }
    case 'rbf':    return (x, y) => rbfKernel(x, y, gamma);
    case 'poly':   return (x, y) => polyKernel(x, y, degree, coef0);
    default: throw new Error(`Unknown kernel: ${type}`);
  }
}

/**
 * Kernel gradient ∂K(xᵢ, x_c)/∂x_c — Biggio et al. 2012 §2.2 ("Kernelization"):
 *   linear  K = xᵢ·x_c              → xᵢ
 *   poly    K = (xᵢ·x_c + R)^d      → d (xᵢ·x_c + R)^{d−1} xᵢ
 *   RBF     K = exp(−γ‖xᵢ − x_c‖²)  → 2γ K (xᵢ − x_c)
 * The paper writes the RBF kernel with γ/2 in the exponent, hence its γK(xᵢ − x_c); we use the
 * LibSVM/sklearn convention exp(−γ‖·‖²) throughout, so the factor is 2γ. The step t multiplying
 * each expression in the paper is applied by the attack loop, not here.
 */
export function createKernelGrad({ type, gamma = 1, degree = 3, coef0 = 1 }) {
  switch (type) {
    case 'linear': return (xi, xc) => [...xi];
    case 'rbf': return (xi, xc) => {
      const k = rbfKernel(xi, xc, gamma);
      return xi.map((v, j) => 2 * gamma * k * (v - xc[j]));
    };
    case 'poly': return (xi, xc) => {
      const f = degree * Math.pow(dot(xi, xc) + coef0, degree - 1);
      return xi.map(v => f * v);
    };
    default: throw new Error(`Unknown kernel: ${type}`);
  }
}

/**
 * sklearn's gamma='scale': γ = 1 / (d · Var(X)), with the variance over every feature value.
 * An RBF width has to match the data's scale: on 784-pixel MNIST squared distances between
 * digits are ~80, so a fixed γ = 0.8 gives K ≈ e⁻⁶⁴ ≈ 0 for every pair — the SVM degenerates to
 * a constant classifier (chance accuracy) and every kernel gradient vanishes.
 */
export function scaleGamma(X: number[][]): number {
  let n = 0, mean = 0, sq = 0;
  for (const x of X) for (const v of x) { n++; mean += v; sq += v * v; }
  mean /= n;
  const variance = sq / n - mean * mean;
  return variance > 1e-12 ? 1 / (X[0].length * variance) : 1;
}

/** Kernel parameters from the UI config: γ is data-scaled unless the user chose 'manual'. */
export function kernelSpec(config: Record<string, any>, X: number[][]) {
  const gamma = config.gammaMode === 'manual' ? Number(config.kernelGamma) : scaleGamma(X);
  return { type: config.kernelType, gamma };
}

/**
 * Memoise a kernel on the identity of its two argument arrays. During the attack only x_c
 * changes between steps, while every training, validation and test row is the same array
 * object, so all kernel values not involving x_c are computed once instead of every
 * iteration (784-d RBF on MNIST: ~10× fewer kernel evaluations). WeakMaps let the entries for
 * discarded x_c arrays be garbage-collected. The `linear` tag is kept for predict().
 */
export function cachedKernel(k: any) {
  const cache = new WeakMap<object, WeakMap<object, number>>();
  const f: any = (a: number[], b: number[]) => {
    const hit = cache.get(a)?.get(b) ?? cache.get(b)?.get(a);
    if (hit !== undefined) return hit;
    const v = k(a, b);
    let row = cache.get(a);
    if (!row) { row = new WeakMap(); cache.set(a, row); }
    row.set(b, v);
    return v;
  };
  f.linear = k.linear;
  return f;
}

/** Compute full kernel matrix K[i][j] = kernel(X[i], X[j]) */
export function computeKernelMatrix(X, kernelFn) {
  const n = X.length;
  const K = new Array(n);
  // Initialize all rows first so symmetric assignment K[j][i] works
  for (let i = 0; i < n; i++) {
    K[i] = new Float64Array(n);
  }
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const val = kernelFn(X[i], X[j]);
      K[i][j] = val;
      K[j][i] = val;
    }
  }
  return K;
}

/** Compute kernel row for single point vs all data */
export function computeKernelRow(x, X, kernelFn) {
  return X.map(xi => kernelFn(x, xi));
}
