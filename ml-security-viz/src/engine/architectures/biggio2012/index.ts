/**
 * Biggio et al. 2012 — "Poisoning Attacks against Support Vector Machines"
 * Registers the SVM poisoning attack (attack.ts) with the AlgorithmModule interface.
 *
 * page.tsx runs this attack in the Web Worker (engine/worker.ts); `runAttack` below is the
 * equivalent main-thread path. Both drive the same `biggioAttack` generator.
 */

import { AlgorithmModule, TraceFrame, registerAlgorithm } from '../registry';
import { createKernel, kernelSpec } from './kernels';
import { trainSVM, getModelState, accuracy, predict as svmPredict } from './model';
import { biggioAttack } from './attack';
import { biggioParams } from './params';

const biggio2012: AlgorithmModule = {
  key: 'biggio2012',
  name: 'SVM Poisoning',
  paper: 'https://arxiv.org/abs/1206.6389',
  paperShort: 'Biggio et al. 2012',
  modelType: 'classification',

  // 2-D generators: keep the usual training points but give the attacker a 500-point validation
  // set (and a 500-point test set), as the paper does, instead of 20% of ~150 points.
  evalPoints: 500,

  datasets: ['biggioGaussian', 'mnist71', 'mnist98', 'mnist40', 'moons', 'circles', 'blobs', 'xor', 'gaussian', 'spiral'],

  configSchema: [
    { key: 'kernelType', label: 'Kernel', type: 'select', section: 'model',
      options: [
        { value: 'linear', label: 'LINEAR' },
        { value: 'rbf', label: 'RBF' },
        { value: 'poly', label: 'POLY (d=3, R=1)' },
      ],
      tooltip: 'Kernel function (§2.2). The paper uses linear and RBF on the Gaussian data, linear on MNIST' },
    { key: 'gammaMode', label: 'Gamma (γ)', type: 'select', section: 'model',
      options: [
        { value: 'scale', label: 'Auto: 1 / (d · Var X)' },
        { value: 'manual', label: 'Manual' },
      ],
      tooltip: 'RBF/poly width. Auto is sklearn’s gamma="scale", which adapts to the data: ' +
        '≈0.017 on 784-pixel MNIST, where any fixed γ ≥ 0.5 makes every kernel value ≈ 0' },
    { key: 'kernelGamma', label: 'Manual γ', type: 'range', section: 'model',
      min: 0.05, max: 5, step: 0.05,
      tooltip: 'RBF width in K = exp(−γ‖x−x′‖²). The paper writes exp(−γ/2‖·‖²), so its γ = 0.5 is 0.25 here' },
    { key: 'svmC', label: 'Regularization (C)', type: 'range', section: 'model',
      min: 0.1, max: 10, step: 0.1, tooltip: 'SVM regularization; C = 1 in every experiment of the paper' },
    { key: 'numPoison', label: 'Poison Points', type: 'range', section: 'attack',
      min: 1, max: 20, step: 1,
      tooltip: 'Attack points, optimised one after another (sequential single-point attacks, §4)' },
    { key: 'attackLabel', label: 'Attacking class y_c', type: 'select', section: 'attack',
      options: [{ value: '-1', label: '−1 (paper)' }, { value: '1', label: '+1' }],
      tooltip: 'The fixed label of every attack point (§2.1). Points are cloned from the other class' },
    { key: 'stepSize', label: 'Step Size (t)', type: 'range', section: 'attack',
      min: 0.01, max: 1, step: 0.01,
      tooltip: 'Fixed step along the unit gradient (Alg. 1 line 7). The paper rejects line search (§2.3)' },
    { key: 'itersPerPoint', label: 'Max Iterations / Point', type: 'range', section: 'attack',
      min: 10, max: 500, step: 10, tooltip: 'Gradient steps allowed for each attack point' },
    { key: 'stopRule', label: 'Stopping Rule', type: 'select', section: 'attack',
      options: [
        { value: 'holdout', label: 'Early stop on held-out validation' },
        { value: 'paper', label: 'Paper: ΔL < 1e-5' },
        { value: 'none', label: 'None (full budget)' },
      ],
      tooltip: 'Held-out: ascend on 80% of D_val, stop when the other 20% stops improving and keep ' +
        'the best point there — later steps only fit the attacker’s own sample. ' +
        'Paper: Alg. 1 line 8, which ends at the first dip caused by an SV-set change' },
    { key: 'initStrategy', label: 'Init Strategy', type: 'select', section: 'attack',
      options: [
        { value: 'random', label: 'Clone + flip (paper)' },
        { value: 'furthest', label: 'Deepest point (heuristic)' },
      ],
      tooltip: 'Paper: clone a random point of the attacked class and flip its label' },
  ],

  defaultConfig: {
    kernelType: 'linear',
    gammaMode: 'scale',
    kernelGamma: 0.25,
    svmC: 1.0,
    numPoison: 5,
    attackLabel: '-1',
    stepSize: 0.25,
    itersPerPoint: 100,
    stopRule: 'holdout',
    initStrategy: 'random',
  },

  trainClean(dataset, config) {
    const kernel = kernelSpec(config, dataset.train.X);
    const model = trainSVM(dataset.train.X, dataset.train.y, createKernel(kernel), config.svmC);
    const state: any = getModelState(model);
    state.kernel = kernel;
    state.testAccuracy = accuracy(model, dataset.test.X, dataset.test.y);
    state.rawModel = model;
    return { modelState: state, rawModel: model };
  },

  runAttack(dataset, cleanModelState, config, onProgress, onComplete, onError, signal) {
    try {
      const params = biggioParams(dataset, config, cleanModelState.testAccuracy);
      const frames = biggioAttack(params);
      const step = () => {
        try {
          if (signal?.aborted) { frames.return(undefined); return; }
          const started = performance.now();
          while (performance.now() - started < 30) {
            const next = frames.next();
            if (next.done) { onComplete(); return; }
            const frame = next.value as TraceFrame;       // no strictNullChecks → no narrowing
            frame.poisonedRawModel.kernelFn = params.kernelFn;
            onProgress(frame);
          }
          setTimeout(step, 0);       // yield so React can paint
        } catch (e: any) {
          onError(e.message);
        }
      };
      setTimeout(step, 0);
    } catch (e: any) {
      onError(e.message);
    }
  },

  predict(rawModel, x) {
    if (!rawModel.kernelFn) throw new Error('SVM model has no kernelFn (re-attach after postMessage)');
    return svmPredict(rawModel, x);
  },
};

registerAlgorithm(biggio2012);
export default biggio2012;
