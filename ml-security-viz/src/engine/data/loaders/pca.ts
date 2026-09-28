/**
 * pca.js — Principal Component Analysis for dimensionality reduction
 * Used to project MNIST 784-dim features to 2D for canvas visualization
 * and to ~20 dims for efficient SVM training
 */

import { zeros, transpose, mmul, colMean } from '../../linalg';

/**
 * Perform PCA on a data matrix X (n × d)
 * Returns the top `numComponents` principal components
 *
 * @param {number[][]} X - n × d data matrix
 * @param {number} numComponents - number of components to keep
 * @returns {{ projected: number[][], components: number[][], mean: number[], variance: number[] }}
 */
export function pca(X, numComponents = 2) {
  const n = X.length;
  const d = X[0].length;

  // 1. Center the data
  const mean = colMean(X);
  const centered = X.map(row => row.map((v, j) => v - mean[j]));

  // 2. Compute covariance matrix (d × d)
  // For high-dim (d >> n), use the dual trick: compute n × n matrix instead
  if (d > n) {
    return _pcaDual(centered, mean, n, d, numComponents);
  }

  // Standard: covariance = (1/n) X^T X
  const Xt = transpose(centered);
  const cov = mmul(Xt, centered);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      cov[i][j] /= (n - 1);

  // 3. Power iteration to find top eigenvectors
  const components = [];
  const variance = [];
  let deflated = cov;

  for (let k = 0; k < numComponents; k++) {
    const { vector, value } = powerIteration(deflated, d);
    components.push(vector);
    variance.push(value);

    // Deflate: remove this component from covariance
    deflated = deflateMatrix(deflated, vector, value, d);
  }

  // 4. Project: each row of X projected onto components
  const projected = centered.map(row =>
    components.map(comp => {
      let s = 0;
      for (let j = 0; j < d; j++) s += row[j] * comp[j];
      return s;
    })
  );

  return { projected, components, mean, variance };
}

/**
 * Project new data points using pre-computed PCA components
 */
export function pcaTransform(X, mean, components) {
  return X.map(row => {
    const centered = row.map((v, j) => v - mean[j]);
    return components.map(comp => {
      let s = 0;
      for (let j = 0; j < row.length; j++) s += centered[j] * comp[j];
      return s;
    });
  });
}

/**
 * Project low-dimensional coordinates back to the original space using PCA inverse transform.
 * X_approx = Z * W^T + mean
 */
export function pcaInverseTransform(Z, mean, components) {
  const numComponents = components.length;
  const d = mean.length;
  return Z.map(z => {
    const x = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let k = 0; k < numComponents; k++) {
        sum += z[k] * components[k][j];
      }
      x[j] = sum + mean[j];
    }
    return x;
  });
}

/**
 * Dual PCA trick for when d >> n (e.g., MNIST 784 >> 200 samples)
 * Compute eigenvectors of (1/n) X X^T (n×n) then recover d-dim eigenvectors
 */
function _pcaDual(centered, mean, n, d, numComponents) {
  // Gram matrix: G = X * X^T (n × n)
  const G = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += centered[i][k] * centered[j][k];
      G[i][j] = s / (n - 1);
      G[j][i] = G[i][j];
    }
  }

  // Find top eigenvectors of G
  const components = [];
  const variance = [];
  let deflated = G;

  for (let k = 0; k < numComponents; k++) {
    const { vector: alphaK, value: lambdaK } = powerIteration(deflated, n);
    variance.push(lambdaK);

    // Recover d-dimensional eigenvector: v = X^T * alpha / (sqrt(lambda * (n-1)))
    const vd = new Array(d).fill(0);
    const scale = Math.sqrt(lambdaK * (n - 1));
    if (scale > 1e-10) {
      for (let j = 0; j < d; j++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += centered[i][j] * alphaK[i];
        vd[j] = s / scale;
      }
    }
    // Normalize
    let normV = 0;
    for (let j = 0; j < d; j++) normV += vd[j] * vd[j];
    normV = Math.sqrt(normV);
    if (normV > 1e-10) {
      for (let j = 0; j < d; j++) vd[j] /= normV;
    }

    components.push(vd);
    deflated = deflateMatrix(deflated, alphaK, lambdaK, n);
  }

  // Project
  const projected = centered.map(row =>
    components.map(comp => {
      let s = 0;
      for (let j = 0; j < d; j++) s += row[j] * comp[j];
      return s;
    })
  );

  return { projected, components, mean, variance };
}

/**
 * Power iteration to find dominant eigenvector
 */
function powerIteration(M, dim, maxIter = 200, tol = 1e-8) {
  // Random initial vector
  let v = new Array(dim);
  for (let i = 0; i < dim; i++) v[i] = Math.random() - 0.5;

  // Normalize
  let n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  v = v.map(x => x / n);

  let eigenvalue = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    // w = M * v
    const w = new Array(dim).fill(0);
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        w[i] += (M[i][j] || 0) * v[j];
      }
    }

    // Eigenvalue estimate
    eigenvalue = 0;
    for (let i = 0; i < dim; i++) eigenvalue += w[i] * v[i];

    // Normalize
    n = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
    if (n < 1e-15) break;

    const vNew = w.map(x => x / n);

    // Check convergence
    let diff = 0;
    for (let i = 0; i < dim; i++) diff += (vNew[i] - v[i]) ** 2;
    v = vNew;
    if (Math.sqrt(diff) < tol) break;
  }

  return { vector: v, value: eigenvalue };
}

/**
 * Deflate matrix: remove component of eigenvector v with eigenvalue lambda
 * M' = M - lambda * v * v^T
 */
function deflateMatrix(M, v, lambda, dim) {
  const D = zeros(dim, dim);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      D[i][j] = (M[i][j] || 0) - lambda * v[i] * v[j];
    }
  }
  return D;
}
