/**
 * CUDA implementation of the engine's ComputeBackend (src/engine/compute/backend.ts), for the
 * attack server only. Wraps the native addon native/build/mlsv_cuda.node (native/cuda/*.cu,
 * built by scripts/build_cuda.mjs).
 *
 * Each operation goes to the GPU only when the matrix is large enough for the launch and copy
 * overhead to pay off; below that the CPU backend is faster, so it is used instead. The
 * crossovers were measured on an RTX 4090 against the CPU backend in Node
 * (npm run bench:compute, scripts/bench_compute.ts). Every result is FP64, like the CPU path.
 */

import path from 'node:path';
import { cpuBackend } from '@/engine/compute/cpu';
import type { ComputeBackend, DenseMatrix, RankOneInverse } from '@/engine/compute/backend';

interface Addon {
  info(): { device: string; computeCapability: string; memoryBytes: number; cudaRuntime: number; cudaDriver: number };
  gram(A: Float64Array, m: number, n: number): Float64Array;
  inverse(M: Float64Array, n: number): Float64Array | null;
  RankOneInverse: new (Ainv: Float64Array, n: number) => {
    update(u: Float64Array, s: number): void;
    apply(v: Float64Array): Float64Array;
    read(): Float64Array;
    dispose(): void;
  };
  DenseMatrix: new (A: Float64Array, m: number, n: number) => {
    mul(v: Float64Array): Float64Array;
    mulT(v: Float64Array): Float64Array;
    dispose(): void;
  };
}

/**
 * Smallest sizes worth sending to the GPU. Measured on an RTX 4090 vs the CPU backend in Node 20
 * (npm run bench:compute; cpu → cuda):
 *   gram      n=32 (m=64) 122 → 75 µs,  n=275 65 → 0.56 ms,  n=784 1.5 s → 3.5 ms
 *   inverse   n=32  83 → 102 µs,  n=64 0.50 → 0.17 ms,  n=275 34 → 1.6 ms
 *   rank-one  n=128 99 → 120 µs,  n=160 151 → 37 µs,  n=275 437 → 37 µs   (per row swap)
 *   A·v+Aᵀ·r  n=96 (m=163) 40 → 37 µs,  n=128 69 → 37 µs,  n=275 299 → 36 µs
 */
export const CUDA_THRESHOLDS = {
  gramFlops: 6e4,       // m·n² for AᵀA
  inverseN: 48,         // n for an n × n inverse
  rankOneN: 160,        // n for a device-resident Sherman–Morrison inverse
  matVecSize: 2.5e4,    // m·n for a device-resident matrix (A·v and Aᵀ·v)
};

export interface CudaStatus {
  available: boolean;
  device?: string;
  computeCapability?: string;
  memoryGB?: number;
  cuda?: string;
  reason?: string;       // why it is unavailable
}

let addon: Addon | null | undefined;
let status: CudaStatus = { available: false, reason: 'not loaded yet' };

/** Load the addon once per process; null (with `status.reason`) when there is no usable GPU. */
function loadAddon(): Addon | null {
  if (addon !== undefined) return addon;
  const file = process.env.MLSV_CUDA_ADDON
    ?? path.join(/*turbopackIgnore: true*/ process.cwd(), 'native', 'build', 'mlsv_cuda.node');
  try {
    const mod = { exports: {} as Addon };
    process.dlopen(mod, file);
    const info = mod.exports.info();
    addon = mod.exports;
    const v = (x: number) => `${Math.floor(x / 1000)}.${(x % 1000) / 10}`;
    status = {
      available: true,
      device: info.device,
      computeCapability: info.computeCapability,
      memoryGB: Math.round(info.memoryBytes / 2 ** 30),
      cuda: v(info.cudaRuntime),
    };
  } catch (e: any) {
    addon = null;
    const msg = String(e?.message ?? e);
    status = {
      available: false,
      reason: /cannot open shared object|no such file/i.test(msg) && msg.includes('mlsv_cuda.node')
        ? 'addon not built (npm run build:cuda)'
        : msg.split('\n')[0],
    };
  }
  return addon;
}

export function cudaStatus(): CudaStatus {
  loadAddon();
  return status;
}

class CudaRankOneInverse implements RankOneInverse {
  private handle: InstanceType<Addon['RankOneInverse']>;
  private buf: Float64Array;
  constructor(a: Addon, Ainv: Float64Array, readonly n: number) {
    this.handle = new a.RankOneInverse(Ainv, n);
    this.buf = new Float64Array(n);
  }
  private staged(v: ArrayLike<number>) {
    if (v instanceof Float64Array) return v;
    this.buf.set(v);
    return this.buf;
  }
  update(u: ArrayLike<number>, s: number) { this.handle.update(this.staged(u), s); }
  apply(v: ArrayLike<number>) { return this.handle.apply(this.staged(v)); }
  read() { return this.handle.read(); }
  dispose() { this.handle.dispose(); }
}

class CudaDenseMatrix implements DenseMatrix {
  private handle: InstanceType<Addon['DenseMatrix']>;
  constructor(a: Addon, A: Float64Array, readonly m: number, readonly n: number) {
    this.handle = new a.DenseMatrix(A, m, n);
  }
  mul(v: ArrayLike<number>) { return this.handle.mul(v instanceof Float64Array ? v : Float64Array.from(v)); }
  mulT(v: ArrayLike<number>) { return this.handle.mulT(v instanceof Float64Array ? v : Float64Array.from(v)); }
  dispose() { this.handle.dispose(); }
}

/** The CUDA backend, or null when no GPU/addon is available. */
export function cudaBackend(): ComputeBackend | null {
  const a = loadAddon();
  if (!a) return null;
  return {
    name: 'cuda',
    device: status.device!,
    gram(A, m, n) {
      return m * n * n >= CUDA_THRESHOLDS.gramFlops ? a.gram(A, m, n) : cpuBackend.gram(A, m, n);
    },
    inverse(M, n) {
      if (n < CUDA_THRESHOLDS.inverseN) return cpuBackend.inverse(M, n);
      // An exactly singular pivot: fall back to Gauss–Jordan, which regularises tiny pivots
      return a.inverse(M, n) ?? cpuBackend.inverse(M, n);
    },
    rankOneInverse(Ainv, n) {
      return n >= CUDA_THRESHOLDS.rankOneN ? new CudaRankOneInverse(a, Ainv, n) : cpuBackend.rankOneInverse(Ainv, n);
    },
    denseMatrix(A, m, n) {
      return m * n >= CUDA_THRESHOLDS.matVecSize ? new CudaDenseMatrix(a, A, m, n) : cpuBackend.denseMatrix(A, m, n);
    },
  };
}
