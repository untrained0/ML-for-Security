#!/usr/bin/env node
/**
 * Builds the optional CUDA addon native/build/mlsv_cuda.node from native/cuda/mlsv_cuda.cu.
 *
 *   node scripts/build_cuda.mjs           # build if missing or older than the source
 *   node scripts/build_cuda.mjs --force   # always rebuild
 *
 * Needs nvcc (CUDA ≥ 11; $CUDA_HOME, /usr/local/cuda or PATH) and Node's own headers
 * (<node prefix>/include/node/node_api.h). No node-gyp: the addon uses the plain Node-API C
 * interface, so one nvcc invocation is the whole build.
 *
 * Like fetch_datasets.mjs this runs before `dev`/`build` and NEVER fails them: on a machine
 * without a GPU toolchain it prints why it skipped, and the server falls back to the CPU backend
 * (src/server/compute/cuda.ts).
 */

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const SRC = path.join(ROOT, 'native', 'cuda', 'mlsv_cuda.cu');
const OUT_DIR = path.join(ROOT, 'native', 'build');
const OUT = path.join(OUT_DIR, 'mlsv_cuda.node');
const FORCE = process.argv.includes('--force');

const skip = (why) => {
  console.log(`[cuda] skipped: ${why} — attacks run on the CPU backend`);
  process.exit(0);
};

function findNvcc() {
  const candidates = [
    process.env.CUDA_HOME && path.join(process.env.CUDA_HOME, 'bin', 'nvcc'),
    process.env.CUDA_PATH && path.join(process.env.CUDA_PATH, 'bin', 'nvcc'),
    '/usr/local/cuda/bin/nvcc',
  ].filter(Boolean);
  for (const c of candidates) if (fs.existsSync(c)) return c;
  try {
    return execFileSync('which', ['nvcc'], { encoding: 'utf8' }).trim() || null;
  } catch {
    return null;
  }
}

function findNodeHeaders() {
  const candidates = [
    path.resolve(process.execPath, '..', '..', 'include', 'node'),
    '/usr/include/node',
    '/usr/local/include/node',
  ];
  return candidates.find(d => fs.existsSync(path.join(d, 'node_api.h'))) ?? null;
}

if (process.platform !== 'linux') skip(`unsupported platform ${process.platform}`);
if (!fs.existsSync(SRC)) skip('native/cuda/mlsv_cuda.cu not found');
if (!FORCE && fs.existsSync(OUT) && fs.statSync(OUT).mtimeMs >= fs.statSync(SRC).mtimeMs) {
  console.log('[cuda] native/build/mlsv_cuda.node is up to date');
  process.exit(0);
}

const nvcc = findNvcc();
if (!nvcc) skip('nvcc not found (set CUDA_HOME)');
const nodeInclude = findNodeHeaders();
if (!nodeInclude) skip('Node headers (node_api.h) not found');
const cudaLib = path.join(path.dirname(path.dirname(fs.realpathSync(nvcc))), 'lib64');

fs.mkdirSync(OUT_DIR, { recursive: true });
const common = [
  '-O3', '-std=c++17', '-shared', '-Xcompiler', '-fPIC',
  '-I', nodeInclude,
  SRC, '-o', OUT,
  '-L', cudaLib, '-lcublas', '-lcusolver',
  // Find libcublas/libcusolver at run time without LD_LIBRARY_PATH
  '-Xlinker', `-rpath=${cudaLib}`,
];

// Compile for the GPU in this machine; if none is visible (e.g. a build container), emit code
// for common architectures plus PTX the driver can JIT for newer ones.
const archSets = [
  ['-arch=native'],
  ['-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_86,code=sm_86',
   '-gencode=arch=compute_89,code=sm_89', '-gencode=arch=compute_89,code=compute_89'],
];

let lastError = null;
for (const arch of archSets) {
  try {
    const t0 = Date.now();
    execFileSync(nvcc, [...arch, ...common], { stdio: ['ignore', 'pipe', 'pipe'] });
    console.log(`[cuda] built native/build/mlsv_cuda.node (${arch[0]}) in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    process.exit(0);
  } catch (e) {
    lastError = String(e.stderr || e.message).trim().split('\n').slice(-6).join('\n');
  }
}
console.log(`[cuda] build failed:\n${lastError}`);
skip('nvcc could not build the addon');
