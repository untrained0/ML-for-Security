import type { AlgorithmModule } from '@/engine/architectures/registry';

/**
 * The config an algorithm actually runs with: its defaults, then the legacy top-level store
 * fields (kernelType, svmC, … kept for biggio2012 back-compat), then the user's overrides.
 */
export function mergedConfig(alg: AlgorithmModule, state: any): Record<string, any> {
  const merged: Record<string, any> = { ...alg.defaultConfig };
  for (const field of alg.configSchema) {
    if (state[field.key] !== undefined) merged[field.key] = state[field.key];
  }
  return Object.assign(merged, state.algorithmConfig);
}
