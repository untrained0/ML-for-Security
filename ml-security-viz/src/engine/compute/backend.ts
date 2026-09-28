/**
 * Pluggable dense linear algebra for the engine.
 *
 * Engine code calls `compute()` for the few operations that dominate an attack's run time on
 * real data (profiled on Jagielski 2018 / House, 275 × 275 systems): XᵀX, a full inverse, an
 * inverse kept current under rank-one row swaps, and a matrix-vector product with a resident
 * matrix (a regression objective's residuals, evaluated at every line-search step). In the
 * browser this is always the CPU backend (cpu.ts, plain TypeScript); the attack server swaps in
 * the CUDA backend (src/server/compute/cuda.ts) when an NVIDIA GPU is available, which itself
 * falls back to the CPU for matrices too small to be worth a kernel launch.
 *
 * Matrices are row-major Float64Array. Everything is FP64: the regression systems are
 * ill-conditioned (OLS on one-hot data), so single precision is not an option.
 */

export interface RankOneInverse {
  readonly n: number;
  /** A⁻¹ ← (A + s·uuᵀ)⁻¹ by Sherman–Morrison, s = ±1; skipped when 1 + s·uᵀA⁻¹u ≈ 0. */
  update(u: ArrayLike<number>, s: number): void;
  /** A⁻¹v */
  apply(v: ArrayLike<number>): Float64Array;
  /** A⁻¹, row-major n × n */
  read(): Float64Array;
  /** Release device memory (no-op on the CPU). */
  dispose(): void;
}

/** A matrix kept where the backend computes (device memory on the GPU) for repeated products. */
export interface DenseMatrix {
  readonly m: number;
  readonly n: number;
  /** A·v (length m) */
  mul(v: ArrayLike<number>): Float64Array;
  /** Aᵀ·v (length n) */
  mulT(v: ArrayLike<number>): Float64Array;
  dispose(): void;
}

export interface ComputeBackend {
  readonly name: 'cpu' | 'cuda';
  /** Human-readable device, e.g. "CPU (JavaScript)" or "NVIDIA GeForce RTX 4090". */
  readonly device: string;
  /** AᵀA for a row-major m × n matrix A (n × n, exactly symmetric). */
  gram(A: Float64Array, m: number, n: number): Float64Array;
  /** M⁻¹ for a row-major n × n M (Gauss–Jordan semantics of linalg.invert on the CPU). */
  inverse(M: Float64Array, n: number): Float64Array;
  /** An inverse A⁻¹ (row-major n × n) that can then be updated in place by rank-one terms. */
  rankOneInverse(Ainv: Float64Array, n: number): RankOneInverse;
  /** A row-major m × n matrix for repeated A·v and Aᵀ·v. */
  denseMatrix(A: Float64Array, m: number, n: number): DenseMatrix;
}

let active: ComputeBackend | null = null;
let fallback: ComputeBackend | null = null;

/** Registered by cpu.ts at import time, so `compute()` always has something to return. */
export function registerDefaultBackend(b: ComputeBackend) {
  fallback = b;
}

/** The backend engine code should use right now. */
export function compute(): ComputeBackend {
  const b = active ?? fallback;
  if (!b) throw new Error('No compute backend registered (import engine/compute/cpu)');
  return b;
}

/** Install a backend for this JS realm (the attack server does this once at start-up). */
export function setComputeBackend(b: ComputeBackend | null) {
  active = b;
}
