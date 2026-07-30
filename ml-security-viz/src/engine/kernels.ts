/**
 * kernels.js — Kernel functions for SVM
 */

import { dot, dist2 } from './linalg';

/** Linear kernel: K(x,y) = x·y */
export function linearKernel(x, y) { return dot(x, y); }

/** RBF kernel: K(x,y) = exp(-γ||x-y||²) */
export function rbfKernel(x, y, gamma) { return Math.exp(-gamma * dist2(x, y)); }

/** Polynomial kernel: K(x,y) = (x·y + c)^d */
export function polyKernel(x, y, degree, coef0) { return Math.pow(dot(x, y) + coef0, degree); }

/** Factory: create kernel function from config */
export function createKernel({ type, gamma = 1, degree = 3, coef0 = 1 }) {
  switch (type) {
    case 'linear': return (x, y) => linearKernel(x, y);
    case 'rbf':    return (x, y) => rbfKernel(x, y, gamma);
    case 'poly':   return (x, y) => polyKernel(x, y, degree, coef0);
    default: throw new Error(`Unknown kernel: ${type}`);
  }
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
