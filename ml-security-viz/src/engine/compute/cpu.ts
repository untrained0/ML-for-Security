/**
 * CPU backend — the reference implementation of ComputeBackend (backend.ts), used in the browser
 * and by the server whenever CUDA is unavailable or a matrix is too small for the GPU.
 *
 * The loops keep the summation order of the code they replaced (RegressionFit's XᵀX build, its
 * Sherman–Morrison `rankOne` and `mv`), so moving to this interface changed no result bit.
 */

import { invert } from '../linalg';
import { registerDefaultBackend, type ComputeBackend, type DenseMatrix, type RankOneInverse } from './backend';

/** y = M·v for row-major n × n M, row by row (same order as the old `mv`). */
function matVec(M: Float64Array, n: number, v: ArrayLike<number>, out: Float64Array = new Float64Array(n)) {
  for (let i = 0, o = 0; i < n; i++, o += n) {
    let s = 0;
    for (let j = 0; j < n; j++) s += M[o + j] * v[j];
    out[i] = s;
  }
  return out;
}

class CpuRankOneInverse implements RankOneInverse {
  private au: Float64Array;
  constructor(private inv: Float64Array, readonly n: number) {
    this.au = new Float64Array(n);
  }

  update(u: ArrayLike<number>, s: number) {
    const { n, inv } = this;
    const Au = matVec(inv, n, u, this.au);
    let uAu = 0;
    for (let j = 0; j < n; j++) uAu += u[j] * Au[j];
    const denom = 1 + s * uAu;
    if (Math.abs(denom) < 1e-12) return;
    const f = s / denom;
    for (let i = 0, o = 0; i < n; i++, o += n) {
      const fi = f * Au[i];
      if (fi === 0) continue;
      for (let j = 0; j < n; j++) inv[o + j] -= fi * Au[j];
    }
  }

  apply(v: ArrayLike<number>) {
    return matVec(this.inv, this.n, v);
  }

  read() {
    return this.inv.slice();
  }

  dispose() {}
}

class CpuDenseMatrix implements DenseMatrix {
  constructor(private A: Float64Array, readonly m: number, readonly n: number) {}

  mul(v: ArrayLike<number>) {
    const { A, m, n } = this, out = new Float64Array(m);
    for (let i = 0, o = 0; i < m; i++, o += n) {
      let s = 0;
      for (let j = 0; j < n; j++) s += A[o + j] * v[j];
      out[i] = s;
    }
    return out;
  }

  mulT(v: ArrayLike<number>) {
    const { A, m, n } = this, out = new Float64Array(n);
    for (let i = 0, o = 0; i < m; i++, o += n) {
      const vi = v[i];
      if (vi === 0) continue;
      for (let j = 0; j < n; j++) out[j] += A[o + j] * vi;
    }
    return out;
  }

  dispose() {}
}

export const cpuBackend: ComputeBackend = {
  name: 'cpu',
  device: 'CPU (JavaScript)',

  gram(A, m, n) {
    // Row by row, skipping zero entries (one-hot columns are mostly 0)
    const C = new Float64Array(n * n);
    for (let i = 0, a = 0; i < m; i++, a += n) {
      for (let j = 0; j < n; j++) {
        const aj = A[a + j];
        if (aj === 0) continue;
        const o = j * n;
        for (let k = 0; k < n; k++) C[o + k] += aj * A[a + k];
      }
    }
    return C;
  },

  inverse(M, n) {
    const rows = Array.from({ length: n }, (_, i) => M.subarray(i * n, (i + 1) * n));
    const inv = invert(rows);
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) out.set(inv[i], i * n);
    return out;
  },

  rankOneInverse(Ainv, n) {
    return new CpuRankOneInverse(Ainv.slice(), n);
  },

  denseMatrix(A, m, n) {
    return new CpuDenseMatrix(A, m, n);
  },
};

registerDefaultBackend(cpuBackend);
