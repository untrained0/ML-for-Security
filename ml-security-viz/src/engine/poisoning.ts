/**
 * poisoning.js — Gradient-Ascent Poisoning Attack on SVMs
 * Biggio et al., "Poisoning Attacks against Support Vector Machines" (arXiv:1206.6389)
 */

import { trainSVM, predict, accuracy, hingeLoss, getModelState } from './svm';
import { norm, scale, vadd } from './linalg';

/**
 * Initialize poison points using the chosen strategy
 */
export function initializePoisonPoints(X, y, numPoison, strategy, kernelFn, C) {
  const n = X.length;
  let poisonX = [], poisonY = [];

  switch (strategy) {
    case 'random': {
      const used = new Set();
      for (let i = 0; i < numPoison; i++) {
        let idx;
        do { idx = Math.floor(Math.random() * n); } while (used.has(idx));
        used.add(idx);
        poisonX.push([...X[idx]]);
        poisonY.push(-y[idx]);
      }
      break;
    }
    case 'furthest': {
      const model = trainSVM(X, y, kernelFn, C);
      const margins = X.map((x, i) => ({ idx: i, margin: Math.abs(predict(model, x)) }));
      margins.sort((a, b) => b.margin - a.margin);
      for (let i = 0; i < numPoison; i++) {
        const idx = margins[i].idx;
        poisonX.push([...X[idx]]);
        poisonY.push(-y[idx]);
      }
      break;
    }
    case 'influential': {
      const model = trainSVM(X, y, kernelFn, C);
      const preds = X.map(x => predict(model, x));
      const scores = X.map((_, i) => ({
        idx: i,
        score: (1 / (Math.abs(preds[i]) + 0.01)) * (1 + Math.abs(y[i] - Math.sign(preds[i])))
      }));
      scores.sort((a, b) => b.score - a.score);
      for (let i = 0; i < numPoison; i++) {
        const idx = scores[i].idx;
        poisonX.push([...X[idx]]);
        poisonY.push(-y[idx]);
      }
      break;
    }
    default: throw new Error(`Unknown init strategy: ${strategy}`);
  }
  return { poisonX, poisonY };
}

/**
 * Compute attacker objective (validation hinge loss of poisoned model)
 */
export function attackerObjective(trainX, trainY, poisonX, poisonY, validX, validY, kernelFn, C) {
  const augX = [...trainX, ...poisonX];
  const augY = [...trainY, ...poisonY];
  const model = trainSVM(augX, augY, kernelFn, C);
  return {
    value: hingeLoss(model, validX, validY),
    accuracy: accuracy(model, validX, validY),
    model: getModelState(model),
    rawModel: model
  };
}

/**
 * Compute gradient of attacker objective w.r.t. a single poison point (finite differences)
 */
export function computeGradient(poisonIdx, trainX, trainY, poisonX, poisonY, validX, validY, kernelFn, C) {
  const d = trainX[0].length;
  const delta = 0.01;
  const gradX = new Array(d);
  for (let j = 0; j < d; j++) {
    const fwd = poisonX.map((p, i) => i === poisonIdx ? [...p.slice(0, j), p[j] + delta, ...p.slice(j + 1)] : p);
    const bwd = poisonX.map((p, i) => i === poisonIdx ? [...p.slice(0, j), p[j] - delta, ...p.slice(j + 1)] : p);
    const fObj = attackerObjective(trainX, trainY, fwd, poisonY, validX, validY, kernelFn, C);
    const bObj = attackerObjective(trainX, trainY, bwd, poisonY, validX, validY, kernelFn, C);
    gradX[j] = (fObj.value - bObj.value) / (2 * delta);
  }
  return { gradX };
}

/**
 * One iteration of gradient-ascent poisoning
 */
export function poisonIteration(trainX, trainY, poisonX, poisonY, validX, validY, kernelFn, C, eta, beta) {
  const numPoison = poisonX.length;
  const currentObj = attackerObjective(trainX, trainY, poisonX, poisonY, validX, validY, kernelFn, C);
  const gradients = [], gradientNorms = [];

  for (let i = 0; i < numPoison; i++) {
    const { gradX } = computeGradient(i, trainX, trainY, poisonX, poisonY, validX, validY, kernelFn, C);
    gradients.push(gradX);
    gradientNorms.push(norm(gradX));
  }

  const newPoisonX = poisonX.map((p, i) => {
    const gN = gradientNorms[i];
    if (gN < 1e-10) return [...p];
    const gNorm = scale(gradients[i], 1 / gN);
    return vadd(p, scale(gNorm, eta)).map(v => Math.max(-3, Math.min(3, v)));
  });

  const newObj = attackerObjective(trainX, trainY, newPoisonX, poisonY, validX, validY, kernelFn, C);

  if (newObj.value < currentObj.value) {
    // Backtrack
    const reduced = poisonX.map((p, i) => {
      const gN = gradientNorms[i];
      if (gN < 1e-10) return [...p];
      return vadd(p, scale(scale(gradients[i], 1 / gN), eta * beta)).map(v => Math.max(-3, Math.min(3, v)));
    });
    const redObj = attackerObjective(trainX, trainY, reduced, poisonY, validX, validY, kernelFn, C);
    if (redObj.value > currentObj.value) {
      return { poisonX: reduced, poisonY, objectiveValue: redObj.value, poisonedAccuracy: redObj.accuracy, poisonedModel: redObj.model, rawModel: redObj.rawModel, gradients, gradientNorms, improved: true };
    }
    return { poisonX: poisonX.map(p => [...p]), poisonY, objectiveValue: currentObj.value, poisonedAccuracy: currentObj.accuracy, poisonedModel: currentObj.model, rawModel: currentObj.rawModel, gradients, gradientNorms, improved: false };
  }

  return { poisonX: newPoisonX, poisonY, objectiveValue: newObj.value, poisonedAccuracy: newObj.accuracy, poisonedModel: newObj.model, rawModel: newObj.rawModel, gradients, gradientNorms, improved: true };
}
