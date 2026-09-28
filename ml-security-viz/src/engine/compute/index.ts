/** Engine entry point for the compute backend; importing it registers the CPU default. */
import './cpu';

export { compute, setComputeBackend } from './backend';
export type { ComputeBackend, RankOneInverse, DenseMatrix } from './backend';
export { cpuBackend } from './cpu';
