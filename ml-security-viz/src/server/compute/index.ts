/**
 * Chooses the attack server's compute backend, once per process.
 *
 *   MLSV_COMPUTE=auto  (default) CUDA if the addon loads and a GPU is present, else CPU
 *   MLSV_COMPUTE=cuda  CUDA or fail loudly (/api/compute reports why)
 *   MLSV_COMPUTE=cpu   never touch the GPU
 *
 * The choice is process-wide rather than per request: the engine reads `compute()` from module
 * state, and two concurrent attacks switching it back and forth would race.
 */

import { cpuBackend, setComputeBackend, type ComputeBackend } from '@/engine/compute';
import { cudaBackend, cudaStatus, type CudaStatus } from './cuda';

export interface ComputeInfo {
  backend: 'cpu' | 'cuda';
  device: string;
  mode: string;
  cuda: CudaStatus;
  error?: string;
}

let chosen: ComputeInfo | null = null;

export function serverCompute(): ComputeInfo {
  if (chosen) return chosen;
  const mode = (process.env.MLSV_COMPUTE ?? 'auto').toLowerCase();
  let backend: ComputeBackend = cpuBackend;
  let error: string | undefined;
  if (mode !== 'cpu') {
    const cuda = cudaBackend();
    if (cuda) backend = cuda;
    else if (mode === 'cuda') error = `MLSV_COMPUTE=cuda but CUDA is unavailable: ${cudaStatus().reason}`;
  }
  setComputeBackend(backend);
  chosen = { backend: backend.name, device: backend.device, mode, cuda: cudaStatus(), error };
  return chosen;
}
