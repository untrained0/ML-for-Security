import { createKernel, createKernelGrad, kernelSpec, cachedKernel } from './kernels';
import { defaultBounds, type BiggioAttackParams } from './attack';

/** Build the attack parameters from a dataset and the merged UI config. */
export function biggioParams(dataset: any, config: Record<string, any>, cleanAccuracy: number): BiggioAttackParams {
  const kernel = kernelSpec(config, dataset.train.X);
  return {
    train: dataset.train,
    valid: dataset.valid,
    test: dataset.test,
    kernelFn: cachedKernel(createKernel(kernel)),
    dK: createKernelGrad(kernel),
    C: config.svmC,
    numPoison: config.numPoison,
    yc: Number(config.attackLabel),
    stepSize: config.stepSize,
    maxIter: config.itersPerPoint,
    stopRule: config.stopRule ?? 'holdout',
    eps: 1e-5,
    bounds: defaultBounds(dataset),
    initStrategy: config.initStrategy,
    cleanAccuracy,
  };
}
