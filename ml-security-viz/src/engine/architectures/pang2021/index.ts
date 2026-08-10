/**
 * Pang et al. 2021 — "Accumulative Poisoning Attacks on Real-time Data"
 * (NeurIPS 2021, arXiv:2106.09993)
 *
 * Online learning setting only (no federated learning).
 * Uses logistic regression with mini-batch SGD for 2D visualization.
 * Registration module — model in model.ts, attack in attack.ts, explainer in explainer.ts.
 */

import { AlgorithmModule, registerAlgorithm } from '../registry';
import { trainFull, getModelState, computeAccuracy, predictProb } from './model';
import { runAccumulativeAttack } from './attack';
import { pang2021ExplainerSteps } from './explainer';

const pang2021: AlgorithmModule = {
  key: 'pang2021',
  name: 'Accumulative Poisoning',
  paper: 'https://arxiv.org/abs/2106.09993',
  paperShort: 'Pang et al. 2021',
  modelType: 'classification',

  datasets: ['gaussian', 'moons', 'circles', 'mnist', 'cifar'],

  configSchema: [
    // Model section
    {
      key: 'learningRate', label: 'Learning Rate (β)', type: 'range', section: 'model',
      min: 0.01, max: 2, step: 0.01,
      tooltip: 'SGD learning rate for online model updates (θ_{t+1} = θ_t − β ∇L)'
    },
    // Attack section
    {
      key: 'epsilon', label: 'Perturbation Budget (ε)', type: 'range', section: 'attack',
      min: 0.05, max: 1.0, step: 0.05,
      tooltip: 'L∞ bound on perturbation: ‖δ‖∞ ≤ ε'
    },
    {
      key: 'numAccumBatches', label: 'Accumulative Batches', type: 'range', section: 'attack',
      min: 2, max: 20, step: 1,
      tooltip: 'Number of batches in the accumulative phase before the trigger'
    },
    {
      key: 'secrecyThreshold', label: 'Secrecy Threshold (τ)', type: 'range', section: 'attack',
      min: 0.5, max: 0.95, step: 0.05,
      tooltip: 'Minimum accuracy during accumulation — if accuracy drops below τ, perturbation is discarded'
    },
    {
      key: 'pgdSteps', label: 'PGD Steps', type: 'range', section: 'attack',
      min: 1, max: 20, step: 1,
      tooltip: 'Number of PGD iterations for crafting each perturbation'
    },
    {
      key: 'pgdStepSize', label: 'PGD Step Size (α)', type: 'range', section: 'attack',
      min: 0.01, max: 0.2, step: 0.01,
      tooltip: 'Step size for each PGD iteration'
    },
  ],

  defaultConfig: {
    learningRate: 0.5,
    epsilon: 0.3,
    numAccumBatches: 8,
    secrecyThreshold: 0.7,
    pgdSteps: 5,
    pgdStepSize: 0.05,
  },

  trainClean(dataset, config) {
    const lr = config.learningRate || 0.5;
    const model = trainFull(dataset.train.X, dataset.train.y, lr, 100);
    const testY = dataset.test.Y || dataset.test.y;
    const state = getModelState(model, dataset.train.X, dataset.train.y, dataset.test.X, testY);
    return { modelState: state, rawModel: model };
  },

  runAttack: runAccumulativeAttack,

  predict(rawModel, x) {
    const p = predictProb(rawModel, x);
    return p >= 0.5 ? 1 : -1;
  },

  explainerSteps: pang2021ExplainerSteps,
};

registerAlgorithm(pang2021);
export default pang2021;
