/**
 * svm.js — SVM solver using Sequential Minimal Optimization (SMO)
 * Based on Platt's SMO (1998). Labels must be +1/-1.
 */

import { computeKernelMatrix } from './kernels';

const TOL = 1e-3;
const EPS = 1e-6;
const MAX_PASSES = 25;

/** Train SVM using SMO */
export function trainSVM(X, y, kernelFn, C = 1.0) {
  const n = X.length;
  const d = X[0].length;
  const K = computeKernelMatrix(X, kernelFn);
  const alpha = new Float64Array(n);
  let b = 0;
  const E = new Float64Array(n);
  for (let i = 0; i < n; i++) E[i] = -y[i];

  let passes = 0;
  while (passes < MAX_PASSES) {
    let numChanged = 0;
    for (let i = 0; i < n; i++) {
      E[i] = _dv(i, alpha, y, K, b) - y[i];
      if ((y[i] * E[i] < -TOL && alpha[i] < C) ||
          (y[i] * E[i] > TOL && alpha[i] > 0)) {
        let j = _selectJ(i, E, n);
        E[j] = _dv(j, alpha, y, K, b) - y[j];
        const oldAI = alpha[i], oldAJ = alpha[j];
        let L, H;
        if (y[i] !== y[j]) {
          L = Math.max(0, alpha[j] - alpha[i]);
          H = Math.min(C, C + alpha[j] - alpha[i]);
        } else {
          L = Math.max(0, alpha[i] + alpha[j] - C);
          H = Math.min(C, alpha[i] + alpha[j]);
        }
        if (Math.abs(L - H) < EPS) continue;
        const eta = 2 * K[i][j] - K[i][i] - K[j][j];
        if (eta >= 0) continue;
        alpha[j] -= y[j] * (E[i] - E[j]) / eta;
        alpha[j] = Math.max(L, Math.min(H, alpha[j]));
        if (Math.abs(alpha[j] - oldAJ) < EPS) { alpha[j] = oldAJ; continue; }
        alpha[i] += y[i] * y[j] * (oldAJ - alpha[j]);
        const b1 = b - E[i] - y[i] * (alpha[i] - oldAI) * K[i][i] - y[j] * (alpha[j] - oldAJ) * K[i][j];
        const b2 = b - E[j] - y[i] * (alpha[i] - oldAI) * K[i][j] - y[j] * (alpha[j] - oldAJ) * K[j][j];
        b = (alpha[i] > 0 && alpha[i] < C) ? b1 : (alpha[j] > 0 && alpha[j] < C) ? b2 : (b1 + b2) / 2;
        for (let k = 0; k < n; k++) E[k] = _dv(k, alpha, y, K, b) - y[k];
        numChanged++;
      }
    }
    passes = numChanged === 0 ? passes + 1 : 0;
  }

  const supportIndices = [];
  for (let i = 0; i < n; i++) if (alpha[i] > EPS) supportIndices.push(i);
  const w = new Array(d).fill(0);
  for (let i = 0; i < n; i++) if (alpha[i] > EPS)
    for (let j = 0; j < d; j++) w[j] += alpha[i] * y[i] * X[i][j];

  return { alpha: Array.from(alpha), b, X, y, K, supportIndices, w, kernelFn, C };
}

/** Decision value for a point */
export function predict(model, x) {
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

function _dv(idx, alpha, y, K, b) {
  let sum = 0;
  for (let j = 0; j < alpha.length; j++) if (alpha[j] > EPS) sum += alpha[j] * y[j] * K[idx][j];
  return sum + b;
}

function _selectJ(i, E, n) {
  let maxD = -1, bestJ = (i + 1) % n;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    const d = Math.abs(E[i] - E[j]);
    if (d > maxD) { maxD = d; bestJ = j; }
  }
  return bestJ;
}
