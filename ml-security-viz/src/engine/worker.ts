import { createKernel, createKernelGrad, kernelSpec, cachedKernel } from './architectures/biggio2012/kernels';
import { biggioAttack, objectiveSurface } from './architectures/biggio2012/attack';
import { biggioParams } from './architectures/biggio2012/params';

self.onmessage = (e: MessageEvent) => {
  const { type, payload } = e.data;
  if (type === 'START_ATTACK') {
    runAttack(payload);
  } else if (type === 'COMPUTE_HEATMAP') {
    computeHeatmap(payload);
  }
};

function computeHeatmap(payload: any) {
  const { dataset, config, gridResolution } = payload;
  try {
    const kernel = kernelSpec(config, dataset.train.X);
    const range = dataset.displayRange ?? { xMin: -3.5, xMax: 3.5, yMin: -3.5, yMax: 3.5 };
    const data = objectiveSurface(
      dataset.train, dataset.valid, cachedKernel(createKernel(kernel)), createKernelGrad(kernel),
      config.svmC, Number(config.attackLabel), range, gridResolution,
    );
    self.postMessage({
      type: 'HEATMAP_COMPLETE',
      payload: { data, resolution: gridResolution, min: range.xMin, max: range.xMax },
    });
  } catch (err: any) {
    self.postMessage({ type: 'ERROR', payload: err.message });
  }
}

function runAttack(payload: any) {
  const { dataset, cleanModelAccuracy, config } = payload;
  try {
    const params = biggioParams(dataset, config, cleanModelAccuracy);
    for (const frame of biggioAttack(params)) {
      self.postMessage({ type: 'PROGRESS', payload: frame });
    }
    self.postMessage({ type: 'COMPLETE' });
  } catch (err: any) {
    self.postMessage({ type: 'ERROR', payload: err.message });
  }
}
