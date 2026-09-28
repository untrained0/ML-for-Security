/**
 * Benchmarks the CPU and CUDA compute backends on the three operations the engine offloads, to
 * choose CUDA_THRESHOLDS in src/server/compute/cuda.ts.
 *
 *   npm run bench:compute
 *
 * "rank-one" is the pattern of one RegressionFit row swap: two Sherman–Morrison updates and one
 * A⁻¹v, i.e. what Jagielski 2018 does hundreds of times per outer iteration.
 */

import { cpuBackend } from '../src/engine/compute/cpu';
import type { ComputeBackend } from '../src/engine/compute/backend';
import { cudaBackend, cudaStatus, CUDA_THRESHOLDS } from '../src/server/compute/cuda';

const cuda = cudaBackend();
if (!cuda) {
  console.log(`CUDA unavailable: ${cudaStatus().reason}`);
  process.exit(0);
}
// Benchmark the raw GPU paths: disable the size dispatch
Object.assign(CUDA_THRESHOLDS, { gramFlops: 0, inverseN: 0, rankOneN: 0, matVecSize: 0 });

const rand = (n: number) => Float64Array.from({ length: n }, () => Math.random());
function spd(n: number) {
  const A = rand(2 * n * n), G = cpuBackend.gram(A, 2 * n, n);
  for (let i = 0; i < n; i++) G[i * n + i] += 1;
  return G;
}

/** Median time in ms of `fn`, after one warm-up run, repeated until ~200 ms have been spent. */
function time(fn: () => void) {
  fn();
  const samples: number[] = [];
  const start = performance.now();
  while (samples.length < 5 || (performance.now() - start < 200 && samples.length < 200)) {
    const t = performance.now();
    fn();
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  return samples[samples.length >> 1];
}

const ops: Record<string, (b: ComputeBackend, n: number) => () => void> = {
  'gram (m = 2n)': (b, n) => { const A = rand(2 * n * n); return () => b.gram(A, 2 * n, n); },
  inverse: (b, n) => { const M = spd(n); return () => b.inverse(M, n); },
  'rank-one swap': (b, n) => {
    const inv = b.rankOneInverse(cpuBackend.inverse(spd(n), n), n);
    const u = rand(n), w = rand(n), v = rand(n);
    return () => { inv.update(u, -1); inv.update(w, +1); inv.apply(v); };
  },
  // A regression objective: residuals and gradient of an m × n design, m ≈ 1.7n like House
  'A·v + Aᵀ·r (m = 1.7n)': (b, n) => {
    const m = Math.round(1.7 * n), A = b.denseMatrix(rand(m * n), m, n), v = rand(n);
    return () => { A.mulT(A.mul(v)); };
  },
};

// Let the GPU leave its idle clocks first, or the first sizes measure the ramp-up
{
  const inv = cuda.rankOneInverse(new Float64Array(512 * 512), 512), u = rand(512);
  const start = performance.now();
  while (performance.now() - start < 1500) { inv.update(u, 1); inv.apply(u); }
  inv.dispose();
}

console.log(`CPU (Node ${process.version}) vs ${cuda.device}\n`);
for (const [name, make] of Object.entries(ops)) {
  console.log(name);
  for (const n of [32, 64, 96, 128, 160, 200, 275, 384, 512, 784]) {
    const c = time(make(cpuBackend, n)), g = time(make(cuda, n));
    const fmt = (ms: number) => (ms < 1 ? `${(ms * 1000).toFixed(0)} µs` : `${ms.toFixed(2)} ms`).padStart(9);
    console.log(`  n=${String(n).padStart(4)}  cpu ${fmt(c)}  cuda ${fmt(g)}  ${(c / g).toFixed(1).padStart(5)}×`);
  }
}
process.exit(0);
