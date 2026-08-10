/**
 * Jagielski et al. 2018 — Attack: Bilevel optimization via KKT / Implicit Differentiation
 * 
 * Computes gradient of validation MSE w.r.t. poison point (x_c, y_c).
 * For Ridge regression, θ* = (X^T X + λI)^{-1} X^T y.
 * By implicit differentiation of the normal equation stationarity condition:
 *   ∂θ/∂x_c = M^{-1} * (∂(X^T y)/∂x_c - ∂(X^T X)/∂x_c * θ*)
 * where M = X^T X + λI.
 * Then: ∂L_val/∂x_c = (∂θ/∂x_c)^T * ∇_θ L_val
 */

import { dot, norm, scale, vadd, vsub } from '../../linalg';
import { mvmul } from '../../linalg';
import { trainRidge, trainLasso, ridgePredict, mse, getRidgeModelState } from './model';
import type { TraceFrame } from '../registry';

export function computePoisonGradient(
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

/** Run the full poisoning attack loop */
export function runRegressionAttack(
  dataset: any,
  cleanModelState: any,
  config: Record<string, any>,
  onProgress: (frame: TraceFrame) => void,
  onComplete: () => void,
  onError: (msg: string) => void
) {
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
      const idx = Math.floor(Math.random() * n);
      const px = train.X[idx].map((v: number) => v + (Math.random() - 0.5) * 0.5);
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
}
