/**
 * svm.ts — C-SVM dual solver: SMO with LIBSVM's maximal-violating-pair working-set selection
 * (Keerthi et al. 2001; Fan, Chen & Lin 2005, "WSS1"). Labels must be +1/−1.
 *
 *   min_α ½ αᵀQα − 1ᵀα   s.t. 0 ≤ α ≤ C,  yᵀα = 0,   Q_ij = y_i y_j K(x_i, x_j)
 *
 * This replaces a "simplified SMO" whose fixed max|E_i − E_j| partner choice stalled and exited
 * with KKT violations of 1–2, i.e. far from the optimum. Biggio's gradient (Eq. 10) is derived
 * from the KKT conditions and the margin set S = {0 < α < C}, so it is only valid on an
 * accurately solved SVM.
 */

import { computeKernelMatrix } from './kernels';

const EPS = 1e-6;          // α below this is treated as 0 (support-vector threshold)
const KKT_TOL = 1e-5;      // stop when the maximal KKT violation m(α) − M(α) is below this
const TAU = 1e-12;

/** Train SVM */
export function trainSVM(X, y, kernelFn, C = 1.0) {
  const n = X.length;
  const d = X[0].length;
  const K = computeKernelMatrix(X, kernelFn);
  const alpha = new Float64Array(n);
  const G = new Float64Array(n).fill(-1);          // ∇ of the dual: Qα − 1, α = 0 initially

  const isUp = (t) => (y[t] === 1 ? alpha[t] < C : alpha[t] > 0);    // I_up
  const isLow = (t) => (y[t] === 1 ? alpha[t] > 0 : alpha[t] < C);   // I_low

  const maxIter = Math.max(10_000_000, 100 * n);
  for (let iter = 0; iter < maxIter; iter++) {
    // Working set: i = argmax_{I_up} −y_t G_t,  j = argmin_{I_low} −y_t G_t
    let i = -1, j = -1, gMax = -Infinity, gMin = Infinity;
    for (let t = 0; t < n; t++) {
      const v = -y[t] * G[t];
      if (isUp(t) && v > gMax) { gMax = v; i = t; }
      if (isLow(t) && v < gMin) { gMin = v; j = t; }
    }
    if (i < 0 || j < 0 || gMax - gMin < KKT_TOL) break;

    const oldI = alpha[i], oldJ = alpha[j];
    // LIBSVM's QD_i + QD_j ∓ 2Q_ij with Q_ij = y_i y_j K_ij is K_ii + K_jj − 2K_ij in both cases
    const quad = Math.max(K[i][i] + K[j][j] - 2 * K[i][j], TAU);
    if (y[i] !== y[j]) {
      const delta = (-G[i] - G[j]) / quad;
      const diff = alpha[i] - alpha[j];
      alpha[i] += delta; alpha[j] += delta;
      if (diff > 0) { if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = diff; } }
      else if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = -diff; }
      if (diff > 0) { if (alpha[i] > C) { alpha[i] = C; alpha[j] = C - diff; } }
      else if (alpha[j] > C) { alpha[j] = C; alpha[i] = C + diff; }
    } else {
      const delta = (G[i] - G[j]) / quad;
      const sum = alpha[i] + alpha[j];
      alpha[i] -= delta; alpha[j] += delta;
      if (sum > C) { if (alpha[i] > C) { alpha[i] = C; alpha[j] = sum - C; } }
      else if (alpha[j] < 0) { alpha[j] = 0; alpha[i] = sum; }
      if (sum > C) { if (alpha[j] > C) { alpha[j] = C; alpha[i] = sum - C; } }
      else if (alpha[i] < 0) { alpha[i] = 0; alpha[j] = sum; }
    }

    const dI = alpha[i] - oldI, dJ = alpha[j] - oldJ;
    for (let k = 0; k < n; k++) G[k] += y[k] * (y[i] * K[i][k] * dI + y[j] * K[j][k] * dJ);
  }

  // Bias (LIBSVM calculate_rho): average y_t G_t over free SVs; f(x) = Σ α y K − ρ, b = −ρ
  let ub = Infinity, lb = -Infinity, sumFree = 0, nFree = 0;
  for (let t = 0; t < n; t++) {
    const yG = y[t] * G[t];
    if (alpha[t] >= C) { if (y[t] === -1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else if (alpha[t] <= 0) { if (y[t] === 1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
    else { sumFree += yG; nFree++; }
  }
  const rho = nFree > 0 ? sumFree / nFree : (ub + lb) / 2;
  const b = Number.isFinite(rho) ? -rho : 0;

  const supportIndices = [];
  for (let i = 0; i < n; i++) if (alpha[i] > EPS) supportIndices.push(i);
  const w = new Array(d).fill(0);
  for (const i of supportIndices)
    for (let j = 0; j < d; j++) w[j] += alpha[i] * y[i] * X[i][j];

  return { alpha: Array.from(alpha), b, X, y, K, supportIndices, w, kernelFn, C };
}

/** Decision value for a point */
export function predict(model, x) {
  if (model.kernelFn?.linear && model.w) {
    let s = model.b;
    for (let j = 0; j < x.length; j++) s += model.w[j] * x[j];
    return s;
  }
  let sum = 0;
  for (const i of model.supportIndices)
    sum += model.alpha[i] * model.y[i] * model.kernelFn(model.X[i], x);
  return sum + model.b;
}

/** Class prediction (+1 or -1) */
export function predictClass(model, x) { return predict(model, x) >= 0 ? 1 : -1; }

/** Accuracy on test set */
export function accuracy(model, X, y) {
  let c = 0;
  for (let i = 0; i < X.length; i++) if (predictClass(model, X[i]) === y[i]) c++;
  return c / X.length;
}

/** Hinge loss */
export function hingeLoss(model, X, y) {
  let loss = 0;
  for (let i = 0; i < X.length; i++) loss += Math.max(0, 1 - y[i] * predict(model, X[i]));
  return loss / X.length;
}

/**
 * Minimal copy of a trained model for trace frames: support vectors only, no kernel matrix and
 * no kernelFn (functions can't cross postMessage; page.tsx re-attaches one). A frame per attack
 * step would otherwise carry the full n×n K and every training row.
 */
export function compactModel(model) {
  const idx = model.supportIndices;
  return {
    alpha: idx.map(i => model.alpha[i]),
    y: idx.map(i => model.y[i]),
    X: idx.map(i => model.X[i]),
    supportIndices: idx.map((_, k) => k),
    b: model.b, w: model.w, C: model.C,
  };
}

/** SVM objective value */
export function svmObjective(model) {
  let wNormSq = 0;
  for (const i of model.supportIndices)
    for (const j of model.supportIndices)
      wNormSq += model.alpha[i] * model.alpha[j] * model.y[i] * model.y[j] * model.K[i][j];
  return 0.5 * wNormSq + model.C * hingeLoss(model, model.X, model.y) * model.X.length;
}

/** Extract model state for display */
export function getModelState(model) {
  const svs = model.supportIndices.map(i => ({
    index: i, point: [...model.X[i]], label: model.y[i], alpha: model.alpha[i]
  }));
  return {
    w: model.w, b: model.b, supportVectors: svs,
    numSupportVectors: svs.length, C: model.C,
    objective: svmObjective(model), hingeLoss: hingeLoss(model, model.X, model.y),
    accuracy: accuracy(model, model.X, model.y),
    wNorm: Math.sqrt(model.w.reduce((s, v) => s + v * v, 0))
  };
}
