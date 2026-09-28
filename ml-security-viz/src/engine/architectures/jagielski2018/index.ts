/**
 * Jagielski et al. 2018 — "Manipulating Machine Learning: Poisoning Attacks and Countermeasures
 * for Regression Learning" (arXiv:1804.00308)
 *
 * Registration module. Model logic in model.ts, attack logic in attack.ts.
 */

import { AlgorithmModule, TraceFrame, registerAlgorithm } from '../registry';
import { fitLinear, ridgePredict, getRidgeModelState, mse, type RegType } from './model';
import { regressionAttack } from './attack';

/** λ grid searched when "cross-validation" is selected (§V-B: λ "set with cross validation"). */
const LAMBDA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];

const jagielski2018: AlgorithmModule = {
  key: 'jagielski2018',
  name: 'Regression Poisoning',
  paper: 'https://arxiv.org/abs/1804.00308',
  paperShort: 'Jagielski et al. 2018',
  modelType: 'regression',

  datasets: ['warfarin', 'loan', 'house', 'linearReg', 'quadraticReg', 'sinReg'],

  configSchema: [
    {
      key: 'regType', label: 'Regression Type', type: 'select', section: 'model',
      options: [
        { label: 'Ordinary Least Squares (OLS)', value: 'ols' },
        { label: 'Ridge (L2)', value: 'ridge' },
        { label: 'LASSO (L1)', value: 'lasso' },
        { label: 'Elastic Net (ρ = 0.5)', value: 'elasticnet' },
      ],
      tooltip: 'The four linear regression models of §II'
    },
    {
      key: 'ridgeLambda', label: 'Regularization (λ)', type: 'select', section: 'model',
      options: [
        { value: 'cv', label: 'Cross-validated (paper)' },
        ...LAMBDA_GRID.map(v => ({ value: String(v), label: String(v) })),
      ],
      tooltip: 'Per-sample λ of Eq. (1): L = (1/N)Σ r² + λΩ(w). The paper sets it by cross-validation'
    },
    {
      key: 'poisonRate', label: 'Poisoning Rate', type: 'select', section: 'attack',
      options: ['0.04', '0.08', '0.12', '0.16', '0.2'].map(v => ({ value: v, label: `${Math.round(+v * 100)}%` })),
      tooltip: 'p/(n+p), varied from 4% to 20% in §V-B'
    },
    {
      key: 'regInit', label: 'Initialisation', type: 'select', section: 'attack',
      options: [
        { value: 'bflip', label: 'BFlip: y = round(1−y)' },
        { value: 'invflip', label: 'InvFlip: y = 1−y' },
      ],
      tooltip: '§III-A. BFlip is preferred in 5 of the 6 best attacks (Table I)'
    },
    {
      key: 'optimizeY', label: 'Optimise', type: 'select', section: 'attack',
      options: [
        { value: 'xy', label: '(x, y) jointly (OptP)' },
        { value: 'x', label: 'x only (BGD baseline)' },
      ],
      tooltip: 'Joint response-variable optimisation is the paper\'s contribution (Eq. 13–14)'
    },
    {
      key: 'objective', label: 'Outer Objective W', type: 'select', section: 'attack',
      options: [
        { value: 'wtr', label: 'W_tr: training MSE + λΩ (Eq. 8)' },
        { value: 'wval', label: 'W_val: validation MSE (Eq. 9)' },
      ],
      tooltip: 'Loss the attacker maximises'
    },
    {
      key: 'lineSearchEta', label: 'Line Search Step (η)', type: 'select', section: 'attack',
      options: ['0.01', '0.03', '0.05', '0.1', '0.3', '0.5', '1'].map(v => ({ value: v, label: v })),
      tooltip: 'Table VI grid; decays by β = 0.75 when an iteration makes no progress'
    },
    {
      key: 'outerIters', label: 'Max Iterations', type: 'range', section: 'attack',
      min: 5, max: 50, step: 1, tooltip: 'Outer iterations of Algorithm 1 (stops earlier when |Δw| < 1e-5)'
    },
  ],

  defaultConfig: {
    regType: 'ridge',
    ridgeLambda: 'cv',
    poisonRate: '0.08',
    regInit: 'bflip',
    optimizeY: 'xy',
    objective: 'wtr',
    lineSearchEta: '0.3',
    outerIters: 30,
  },

  trainClean(dataset, config) {
    const type = config.regType as RegType;
    let lambda = type === 'ols' ? 0 : Number(config.ridgeLambda);
    if (type !== 'ols' && config.ridgeLambda === 'cv') {
      // Pick λ by validation MSE on the clean split
      let best = Infinity;
      for (const lam of LAMBDA_GRID) {
        const m = mse(fitLinear(dataset.train.X, dataset.train.y, type, lam), dataset.valid.X, dataset.valid.y);
        if (m < best) { best = m; lambda = lam; }
      }
    }
    const model = fitLinear(dataset.train.X, dataset.train.y, type, lambda);
    const state = getRidgeModelState(model, dataset.train.X, dataset.train.y, dataset.test.X, dataset.test.y);
    return { modelState: state, rawModel: model };
  },

  runAttack(dataset, cleanModelState, config, onProgress, onComplete, onError, signal) {
    try {
      const frames = regressionAttack({
        train: dataset.train, valid: dataset.valid, test: dataset.test,
        type: config.regType,
        lambda: cleanModelState.lambda,              // the λ the clean model was trained with
        poisonRate: Number(config.poisonRate),
        init: config.regInit,
        optimizeY: config.optimizeY === 'xy',
        objective: config.objective,
        eta: Number(config.lineSearchEta),
        beta: 0.75,
        eps: 1e-5,
        maxIter: config.outerIters,
        bounds: dataset.bounds ?? [0, 1],
        onehotGroups: dataset.onehotGroups,
        cleanMSE: cleanModelState.testMSE,
        cleanTheta: cleanModelState.theta,
      });
      const step = () => {
        try {
          if (signal?.aborted) { frames.return(undefined); return; }   // runs the generator's finally
          const started = performance.now();
          while (performance.now() - started < 30) {
            const next = frames.next();
            if (next.done) { onComplete(); return; }
            if (next.value) onProgress(next.value as TraceFrame);
          }
          setTimeout(step, 0);                       // yield so React can paint
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
    return ridgePredict(rawModel, x);
  },
};

registerAlgorithm(jagielski2018);
export default jagielski2018;
