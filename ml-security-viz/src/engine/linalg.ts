/**
 * linalg.js — Lightweight linear algebra for 2D ML computations
 * All operations work on arrays. Zero external dependencies.
 */

/** Create a rows×cols matrix of zeros */
export function zeros(rows, cols) {
  const m = new Array(rows);
  for (let i = 0; i < rows; i++) m[i] = new Float64Array(cols);
  return m;
}

/** Identity matrix of size n */
export function eye(n) {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m[i][i] = 1;
  return m;
}

/** Dot product of two vectors */
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Euclidean norm */
export function norm(v) {
  return Math.sqrt(dot(v, v));
}

/** Scale vector by scalar */
export function scale(v, s) {
  return v.map(x => x * s);
}

/** Element-wise add */
export function vadd(a, b) {
  return a.map((x, i) => x + b[i]);
}

/** Element-wise subtract */
export function vsub(a, b) {
  return a.map((x, i) => x - b[i]);
}

/** Matrix-vector multiply */
export function mvmul(M, v) {
  return M.map(row => {
    let s = 0;
    for (let j = 0; j < row.length; j++) s += row[j] * v[j];
    return s;
  });
}

/** Matrix multiply A(m×p) * B(p×n) */
export function mmul(A, B) {
  const m = A.length, p = B.length, n = B[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++)
    for (let k = 0; k < p; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) C[i][j] += aik * B[k][j];
    }
  return C;
}

/** Transpose */
export function transpose(M) {
  const rows = M.length, cols = M[0].length;
  const T = zeros(cols, rows);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) T[j][i] = M[i][j];
  return T;
}

/**
 * Invert a square matrix via Gauss-Jordan with partial pivoting. Columns left of `col` are
 * already reduced to 0 in every row but their pivot row, so each sweep starts at `col`: the
 * skipped updates were `x − f·0`, i.e. exact no-ops.
 */
export function invert(M) {
  const n = M.length, w = 2 * n;
  const aug = zeros(n, w);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i][j] = M[i][j];
    aug[i][n + i] = 1;
  }
  for (let col = 0; col < n; col++) {
    let maxRow = col, maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(aug[row][col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxVal < 1e-12) aug[col][col] += 1e-8;
    if (maxRow !== col) [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const pr = aug[col];
    const pivot = pr[col];
    for (let j = col; j < w; j++) pr[j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const r = aug[row];
      const factor = r[col];
      if (factor === 0) continue;
      for (let j = col; j < w; j++) r[j] -= factor * pr[j];
    }
  }
  const inv = zeros(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) inv[i][j] = aug[i][n + j];
  return inv;
}

/** Squared Euclidean distance */
export function dist2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s;
}

/** Clip values to [lo, hi] */
export function clip(v, lo, hi) {
  return v.map(x => Math.max(lo, Math.min(hi, x)));
}

/** Outer product */
export function outer(a, b) {
  const m = a.length, n = b.length;
  const M = zeros(m, n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) M[i][j] = a[i] * b[j];
  return M;
}

/** Column means */
export function colMean(M) {
  const rows = M.length, cols = M[0].length;
  const mu = new Array(cols).fill(0);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) mu[j] += M[i][j];
  for (let j = 0; j < cols; j++) mu[j] /= rows;
  return mu;
}
