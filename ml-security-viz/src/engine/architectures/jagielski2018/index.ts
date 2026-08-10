/**
 * Jagielski et al. 2018 — "Manipulating Machine Learning: Poisoning Attacks
 * and Countermeasures for Regression Learning" (arXiv:1804.00308)
 *
 * Registration module. Model logic in model.ts, attack logic in attack.ts.
 */

import { AlgorithmModule, registerAlgorithm } from '../registry';
import { trainRidge, trainLasso, ridgePredict, getRidgeModelState } from './model';
import { runRegressionAttack } from './attack';

const jagielski2018: AlgorithmModule = {
  key: 'jagielski2018',
  name: 'Regression Poisoning',
  paper: 'https://arxiv.org/abs/1804.00308',
  paperShort: 'Jagielski et al. 2018',
  modelType: 'regression',

  datasets: ['linearReg', 'quadraticReg', 'sinReg', 'warfarin', 'lendingClub', 'housePricing'],

  configSchema: [
    {
      key: 'regType', label: 'Regression Type', type: 'select', section: 'model',
      options: [
        { label: 'Ordinary Least Squares (OLS)', value: 'ols' },
        { label: 'Ridge (L2)', value: 'ridge' },
        { label: 'LASSO (L1)', value: 'lasso' }
      ],
      tooltip: 'Choose the regularization technique'
    },
    {
      key: 'ridgeLambda', label: 'Regularization (λ)', type: 'range', section: 'model',
      min: 0, max: 5, step: 0.1, tooltip: 'Regularization parameter — 0 = OLS, higher = more regularization'
    },
    {
      key: 'numPoison', label: 'Poison Points', type: 'range', section: 'attack',
      min: 1, max: 15, step: 1, tooltip: 'Number of adversarial data points to inject'
    },
    {
      key: 'attackEta', label: 'Step Size (η)', type: 'range', section: 'attack',
      min: 0.01, max: 1, step: 0.01, tooltip: 'Gradient ascent step size for poison optimization'
    },
    {
      key: 'attackMaxIter', label: 'Max Iterations', type: 'range', section: 'attack',
      min: 5, max: 50, step: 1, tooltip: 'Maximum optimization iterations'
    },
  ],

  defaultConfig: {
    regType: 'ridge',
    ridgeLambda: 0.1,
    numPoison: 3,
    attackEta: 0.3,
    attackMaxIter: 25,
    attackBeta: 0.5,
  },

  trainClean(dataset, config) {
    const lambda = config.regType === 'ols' ? 0 : config.ridgeLambda;
    const trainFn = config.regType === 'lasso' ? trainLasso : trainRidge;
    
    const model = trainFn(dataset.train.X, dataset.train.y, lambda);
    const state = getRidgeModelState(model, dataset.train.X, dataset.train.y, dataset.test.X, dataset.test.y);
    return { modelState: state, rawModel: model };
  },

  runAttack: runRegressionAttack,

  predict(rawModel, x) {
    return ridgePredict(rawModel, x);
  },
};

registerAlgorithm(jagielski2018);
export default jagielski2018;
