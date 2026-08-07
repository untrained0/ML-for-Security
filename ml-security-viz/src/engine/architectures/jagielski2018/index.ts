/**
 * Jagielski et al. 2018 — "Manipulating Machine Learning: Poisoning Attacks
 * and Countermeasures for Regression Learning" (arXiv:1804.00308)
 *
 * Implements bilevel optimization poisoning against Ridge/LASSO/OLS Regression.
 * The attacker maximizes validation MSE by injecting poison points,
 * computing gradients via implicit differentiation of the normal equation.
 */

import { AlgorithmModule, TraceFrame, registerAlgorithm } from '../registry';
import { zeros, eye, dot, norm, scale, vadd, vsub, transpose, mmul, invert, mvmul } from '../../linalg';

// ─── Ridge Regression ───────────────────────────────────────────────

/** Train Ridge Regression: θ* = (X^T X + λI)^-1 X^T y */
function trainRidge(X: number[][], y: number[], lambda: number) {
  const n = X.length;
  const d = X[0].length;

  // Augment X with bias column of 1s
  const Xaug = X.map(row => [...row, 1]);
  const dAug = d + 1;

  // X^T X
  const XtX = zeros(dAug, dAug);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < dAug; j++)
      for (let k = 0; k < dAug; k++)
        XtX[j][k] += Xaug[i][j] * Xaug[i][k];

  // Add regularization (don't regularize bias)
  for (let j = 0; j < d; j++) XtX[j][j] += lambda;

  // X^T y
  const Xty = new Array(dAug).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < dAug; j++)
      Xty[j] += Xaug[i][j] * y[i];

  // θ = (X^T X + λI)^-1 X^T y
  const invXtX = invert(XtX);
  const theta = mvmul(invXtX, Xty);

  return { theta, Xaug, invXtX, lambda, d, dAug, type: 'ridge' };
}

// ─── LASSO Regression (Coordinate Descent) ──────────────────────────

/** Train LASSO Regression with L1 Penalty */
function trainLasso(X: number[][], y: number[], lambda: number, maxIter=500, tol=1e-4) {
  const n = X.length;
  const d = X[0].length;
  const Xaug = X.map(row => [...row, 1]);
  const dAug = d + 1;
  let theta = new Array(dAug).fill(0);
  
  // Precompute z_j = sum(X_ij^2)
  const z = new Array(dAug).fill(0);
  for(let j=0; j<dAug; j++) {
    for(let i=0; i<n; i++) z[j] += Xaug[i][j] * Xaug[i][j];
  }

  for(let iter=0; iter<maxIter; iter++) {
    let maxDiff = 0;
    for(let j=0; j<dAug; j++) {
      if (z[j] === 0) continue;
      
      let rho = 0;
      for(let i=0; i<n; i++) {
        let predExcludingJ = 0;
        for(let k=0; k<dAug; k++) {
          if(k !== j) predExcludingJ += Xaug[i][k] * theta[k];
        }
        rho += Xaug[i][j] * (y[i] - predExcludingJ);
      }
      
      let newTheta = 0;
      if (j === d) { 
        // Bias is not regularized
        newTheta = rho / z[j];
      } else {
        // Soft thresholding: S(rho, lambda/2) / z
        const threshold = lambda / 2;
        if (rho > threshold) newTheta = (rho - threshold) / z[j];
        else if (rho < -threshold) newTheta = (rho + threshold) / z[j];
        else newTheta = 0;
      }
      
      maxDiff = Math.max(maxDiff, Math.abs(newTheta - theta[j]));
      theta[j] = newTheta;
    }
    if (maxDiff < tol) break;
  }
  
  // For implicit differentiation, gradients are 0 for inactive weights.
  // We compute the active-set inverse Hessian and pad it with zeros.
  const XtX = zeros(dAug, dAug);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < dAug; j++)
      for (let k = 0; k < dAug; k++)
        XtX[j][k] += Xaug[i][j] * Xaug[i][k];
        
  const activeCols = [];
  for(let j=0; j<dAug; j++) {
    if(Math.abs(theta[j]) > 1e-6 || j === d) activeCols.push(j);
  }
  
  const smallXtX = zeros(activeCols.length, activeCols.length);
  for(let i=0; i<activeCols.length; i++)
    for(let j=0; j<activeCols.length; j++)
      smallXtX[i][j] = XtX[activeCols[i]][activeCols[j]];
      
  const smallInv = invert(smallXtX);
  
  const invXtX = zeros(dAug, dAug);
  for(let i=0; i<activeCols.length; i++)
    for(let j=0; j<activeCols.length; j++)
      invXtX[activeCols[i]][activeCols[j]] = smallInv[i][j];

  return { theta, Xaug, invXtX, lambda, d, dAug, type: 'lasso' };
}

/** Predict with model */
function ridgePredict(model: any, x: number[]): number {
  const xAug = [...x, 1];
  return dot(model.theta, xAug);
}

/** MSE on a dataset */
function mse(model: any, X: number[][], y: number[]): number {
  let sum = 0;
  for (let i = 0; i < X.length; i++) {
    const err = ridgePredict(model, X[i]) - y[i];
    sum += err * err;
  }
  return sum / X.length;
}

/** R² score */
function r2Score(model: any, X: number[][], y: number[]): number {
  const yMean = y.reduce((s, v) => s + v, 0) / y.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < X.length; i++) {
    const err = y[i] - ridgePredict(model, X[i]);
    ssRes += err * err;
    ssTot += (y[i] - yMean) * (y[i] - yMean);
  }
  return 1 - ssRes / (ssTot + 1e-12);
}

/** Get model state for display */
function getRidgeModelState(model: any, X: number[][], y: number[], testX?: number[][], testY?: number[]) {
  const mseVal = mse(model, X, y);
  const r2 = r2Score(model, X, y);
  const testMSE = testX && testY ? mse(model, testX, testY) : mseVal;
  const testR2 = testX && testY ? r2Score(model, testX, testY) : r2;
  const wNorm = Math.sqrt(model.theta.slice(0, -1).reduce((s: number, v: number) => s + v * v, 0));

  return {
    theta: [...model.theta],
    w: model.theta.slice(0, -1),
    b: model.theta[model.theta.length - 1],
    lambda: model.lambda,
    mse: mseVal,
    r2,
    testMSE,
    testR2,
    wNorm,
    regType: model.type
  };
}

// ─── Attack: Bilevel Optimization via KKT / Implicit Differentiation ────

/**
 * Compute gradient of validation MSE w.r.t. poison point (x_c, y_c).
 *
 * For Ridge regression, θ* = (X^T X + λI)^{-1} X^T y.
 * By implicit differentiation of the normal equation stationarity condition:
 *   ∂θ/∂x_c = M^{-1} * (∂(X^T y)/∂x_c - ∂(X^T X)/∂x_c * θ*)
 * where M = X^T X + λI.
 *
 * Then: ∂L_val/∂x_c = (∂θ/∂x_c)^T * ∇_θ L_val
 */
function computePoisonGradient(
  trainX: number[][], trainY: number[],
  poisonX: number[][], poisonY: number[],
  validX: number[][], validY: number[],
  lambda: number,
  poisonIdx: number,
  trainFn: (X: number[][], y: number[], lambda: number) => any
): { gradX: number[]; gradY: number } {
  const allX = [...trainX, ...poisonX];
  const allY = [...trainY, ...poisonY];
  const model = trainFn(allX, allY, lambda);
  const { theta, invXtX, dAug } = model;
  const d = trainX[0].length;

  // ∇_θ L_val = (2/n_val) * Σ (θ^T x_i - y_i) * x_i_aug
  const nVal = validX.length;
  const gradTheta = new Array(dAug).fill(0);
  for (let i = 0; i < nVal; i++) {
    const xAug = [...validX[i], 1];
    const residual = dot(theta, xAug) - validY[i];
    for (let j = 0; j < dAug; j++) gradTheta[j] += 2 * residual * xAug[j] / nVal;
  }

  // Now compute ∂θ*/∂x_c_j for each feature j of poison point poisonIdx
  // The poison point index in the augmented data is trainX.length + poisonIdx
  const cIdx = trainX.length + poisonIdx;
  const xc = allX[cIdx];
  const yc = allY[cIdx];
  const xcAug = [...xc, 1];
  const n = allX.length;

  const gradX = new Array(d).fill(0);

  for (let j = 0; j < d; j++) {
    // ∂(X^T y)/∂x_c_j: only row cIdx contributes
    const dXty = new Array(dAug).fill(0);
    dXty[j] = yc;

    // ∂(X^T X)/∂x_c_j * θ
    const dotXcTheta = dot(xcAug, theta);
    const dXtXTheta = new Array(dAug).fill(0);
    for (let k = 0; k < dAug; k++) {
      dXtXTheta[k] = xcAug[k] * theta[j];
      if (k === j) dXtXTheta[k] += dotXcTheta;
    }

    // ∂θ*/∂x_c_j = M^{-1} * (dXty - dXtXTheta)
    const rhs = new Array(dAug).fill(0);
    for (let k = 0; k < dAug; k++) rhs[k] = dXty[k] - dXtXTheta[k];
    const dThetaDxj = mvmul(invXtX, rhs);

    // ∂L_val/∂x_c_j = dot(dThetaDxj, gradTheta)
    gradX[j] = dot(dThetaDxj, gradTheta);
  }

  // Gradient w.r.t. y_c:
  const dThetaDyc = mvmul(invXtX, xcAug);
  const gradY = dot(dThetaDyc, gradTheta);

  return { gradX, gradY };
}

// ─── AlgorithmModule Implementation ──────────────────────────────────

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

  runAttack(dataset, cleanModelState, config, onProgress, onComplete, onError) {
    try {
      const { train, valid, test } = dataset;
      const lambda = config.regType === 'ols' ? 0 : config.ridgeLambda;
      const trainFn = config.regType === 'lasso' ? trainLasso : trainRidge;

      // Initialize poison points: random points in the data range with flipped y
      let poisonX: number[][] = [];
      let poisonY: number[] = [];
      const n = train.X.length;
      const d = train.X[0].length;

      for (let i = 0; i < config.numPoison; i++) {
        // Pick a random training point and place poison nearby
        const idx = Math.floor(Math.random() * n);
        const px = train.X[idx].map((v: number) => v + (Math.random() - 0.5) * 0.5);
        // Flip the y value to cause maximum damage
        const yRange = Math.max(...train.y) - Math.min(...train.y);
        const yMean = train.y.reduce((s: number, v: number) => s + v, 0) / n;
        const py = train.y[idx] > yMean ? yMean - yRange * 0.5 : yMean + yRange * 0.5;
        poisonX.push(px);
        poisonY.push(py);
      }

      // Compute initial poisoned model
      const initModel = trainFn([...train.X, ...poisonX], [...train.y, ...poisonY], lambda);
      const initState = getRidgeModelState(initModel, train.X, train.y, test.X, test.y);
      const initMSE = mse(initModel, valid.X, valid.y);

      const cleanTheta = cleanModelState.theta;

      onProgress({
        iteration: 0,
        poisonX: poisonX.map(p => [...p]),
        poisonY: [...poisonY],
        poisonedModel: initState,
        poisonedRawModel: initModel,
        poisonedMSE: initState.testMSE,
        objectiveValue: initMSE,
        gradients: [],
        gradientNorms: [],
        cleanMSE: cleanModelState.testMSE,
        deltaW: norm(vsub(initState.theta, cleanTheta)),
      });

      let lastObj = initMSE;
      let noProgress = 0;
      let iter = 1;

      const step = () => {
        if (iter > config.attackMaxIter) { onComplete(); return; }

        try {
          const gradients: number[][] = [];
          const gradientNorms: number[] = [];
          const yGradients: number[] = [];

          // Compute gradient for each poison point
          for (let i = 0; i < config.numPoison; i++) {
            const { gradX, gradY } = computePoisonGradient(
              train.X, train.y, poisonX, poisonY,
              valid.X, valid.y, lambda, i, trainFn
            );
            gradients.push(gradX);
            gradientNorms.push(norm(gradX));
            yGradients.push(gradY);
          }

          // Gradient ascent (we want to MAXIMIZE validation MSE)
          const newPoisonX = poisonX.map((p, i) => {
            const gN = gradientNorms[i];
            if (gN < 1e-10) return [...p];
            const gNorm = scale(gradients[i], 1 / gN);
            return vadd(p, scale(gNorm, config.attackEta)).map((v: number) => Math.max(-3.5, Math.min(3.5, v)));
          });
          const newPoisonY = poisonY.map((y, i) => {
            const absGy = Math.abs(yGradients[i]);
            if (absGy < 1e-10) return y;
            return y + Math.sign(yGradients[i]) * config.attackEta * 0.5;
          });

          // Compute new model with updated poison
          const newModel = trainFn([...train.X, ...newPoisonX], [...train.y, ...newPoisonY], lambda);
          const newMSE = mse(newModel, valid.X, valid.y);
          const newState = getRidgeModelState(newModel, train.X, train.y, test.X, test.y);

          if (newMSE >= lastObj) {
            poisonX = newPoisonX;
            poisonY = newPoisonY;
          } else {
            // Backtrack with smaller step
            const reducedPoisonX = poisonX.map((p, i) => {
              const gN = gradientNorms[i];
              if (gN < 1e-10) return [...p];
              const gNorm = scale(gradients[i], 1 / gN);
              return vadd(p, scale(gNorm, config.attackEta * config.attackBeta)).map((v: number) => Math.max(-3.5, Math.min(3.5, v)));
            });
            const reducedPoisonY = poisonY.map((y, i) => {
              const absGy = Math.abs(yGradients[i]);
              if (absGy < 1e-10) return y;
              return y + Math.sign(yGradients[i]) * config.attackEta * config.attackBeta * 0.5;
            });
            const reducedModel = trainFn([...train.X, ...reducedPoisonX], [...train.y, ...reducedPoisonY], lambda);
            const reducedMSE = mse(reducedModel, valid.X, valid.y);

            if (reducedMSE > lastObj) {
              poisonX = reducedPoisonX;
              poisonY = reducedPoisonY;
            }
            // Otherwise keep current position
          }

          // Re-compute final state
          const finalModel = trainFn([...train.X, ...poisonX], [...train.y, ...poisonY], lambda);
          const finalMSE = mse(finalModel, valid.X, valid.y);
          const finalState = getRidgeModelState(finalModel, train.X, train.y, test.X, test.y);

          onProgress({
            iteration: iter,
            poisonX: poisonX.map(p => [...p]),
            poisonY: [...poisonY],
            poisonedModel: finalState,
            poisonedRawModel: finalModel,
            poisonedMSE: finalState.testMSE,
            objectiveValue: finalMSE,
            gradients,
            gradientNorms,
            cleanMSE: cleanModelState.testMSE,
            deltaW: norm(vsub(finalState.theta, cleanTheta)),
          });

          const diff = Math.abs(finalMSE - lastObj);
          if (diff < 1e-6) {
            noProgress++;
            if (noProgress >= 3) { onComplete(); return; }
          } else {
            noProgress = 0;
          }
          lastObj = finalMSE;
          iter++;

          setTimeout(step, 0);
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
