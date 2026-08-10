/**
 * Jagielski et al. 2018 — Regression model implementations
 * Ridge, LASSO, and OLS regression with implicit differentiation support.
 */

import { zeros, dot, invert, mvmul } from '../../linalg';

// ─── Ridge Regression ───────────────────────────────────────────────

/** Train Ridge Regression: θ* = (X^T X + λI)^-1 X^T y */
export function trainRidge(X: number[][], y: number[], lambda: number) {
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
export function trainLasso(X: number[][], y: number[], lambda: number, maxIter=500, tol=1e-4) {
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
export function ridgePredict(model: any, x: number[]): number {
  const xAug = [...x, 1];
  return dot(model.theta, xAug);
}

/** MSE on a dataset */
export function mse(model: any, X: number[][], y: number[]): number {
  let sum = 0;
  for (let i = 0; i < X.length; i++) {
    const err = ridgePredict(model, X[i]) - y[i];
    sum += err * err;
  }
  return sum / X.length;
}

/** R² score */
export function r2Score(model: any, X: number[][], y: number[]): number {
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
export function getRidgeModelState(model: any, X: number[][], y: number[], testX?: number[][], testY?: number[]) {
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
