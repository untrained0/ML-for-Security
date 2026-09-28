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
      key: 'burnInEpochs', label: 'Burn-in Epochs', type: 'range', section: 'model',
      min: 2, max: 100, step: 1,
      tooltip: 'How far the victim is trained before the attack begins. The paper attacks a model that is still learning online — a fully converged model has near-zero gradients and nothing to exploit, so pushing this to the maximum defuses the attack'
    },
    {
      key: 'epsilon', label: 'Perturbation Budget (ε)', type: 'range', section: 'attack',
      min: 0.05, max: 1.0, step: 0.05,
      tooltip: 'L∞ bound on perturbation: ‖δ‖∞ ≤ ε'
    },
    {
      key: 'numAccumBatches', label: 'Accumulative Batches', type: 'range', section: 'attack',
      min: 2, max: 40, step: 1,
      tooltip: 'Number of batches in the accumulative phase before the trigger'
    },
    {
      key: 'lambda', label: 'Accumulation Strength (λ)', type: 'range', section: 'attack',
      min: 0, max: 5, step: 0.1,
      tooltip: 'Mixing weight in d_t = ĝ(S_t) + λĜ_t. Both terms are unit vectors, so λ≈1 means "half honest learning, half accumulation". Set λ=0 for the control: the trigger fires with nothing accumulated behind it'
    },
    {
      key: 'gamma', label: 'Stealth Budget (γ)', type: 'range', section: 'attack',
      min: 0.01, max: 1.0, step: 0.01,
      tooltip: 'Eq. 7 constraint: validation loss may exceed the clean trajectory by at most γ. Exceeding it early-stops accumulation'
    },
    {
      key: 'secrecyThreshold', label: 'Accuracy Floor (τ)', type: 'range', section: 'attack',
      min: 0.5, max: 0.95, step: 0.05,
      tooltip: 'Hard early-stop: if validation accuracy falls below τ the accumulation is abandoned'
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

  // Tuned so the phenomenon is actually visible in 2-D. A logistic regression
  // with three parameters gives the attacker far less room than the deep nets
  // in the paper, so ε and λ have to be larger here than anything you would
  // call "imperceptible" — see the caveat in explainer.ts.
  defaultConfig: {
    learningRate: 1.5,
    burnInEpochs: 20,
    epsilon: 0.8,
    numAccumBatches: 12,
    lambda: 2,
    gamma: 0.6,
    secrecyThreshold: 0.6,
    pgdSteps: 8,
    pgdStepSize: 0.15,
  },

  trainClean(dataset, config) {
    const lr = config.learningRate || 0.5;
    // Burn-in, not convergence. The victim in the paper is mid-training when
    // the attacker arrives; a converged 3-parameter model has vanishing
    // gradients and a wide margin, which is exactly the regime where a single
    // bounded update provably cannot change a prediction.
    const model = trainFull(dataset.train.X, dataset.train.y, lr, config.burnInEpochs ?? 20);
    const testY = dataset.test.Y || dataset.test.y;
    const state = getModelState(model, dataset.train.X, dataset.train.y, dataset.test.X, testY);
    return { modelState: state, rawModel: model };
  },

  runAttack: runAccumulativeAttack,

  predict(rawModel, x) {
    const p = predictProb(rawModel, x);
    return p >= 0.5 ? 1 : -1;
  },

  // Smooth boundary for the canvas: P(y=+1|x) − 0.5 changes sign exactly where predict does
  score(rawModel, x) {
    return predictProb(rawModel, x) - 0.5;
  },

  explainerSteps: pang2021ExplainerSteps,
};

registerAlgorithm(pang2021);
export default pang2021;
