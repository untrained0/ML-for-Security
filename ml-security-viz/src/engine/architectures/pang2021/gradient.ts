/**
 * Pang et al. 2021 — Gradient computation for accumulative perturbation
 * 
 * Computes gradients for crafting perturbations during the accumulative phase
 * and for generating the trigger batch. Uses PGD (Projected Gradient Descent)
 * with L∞ constraints.
 */

import { sigmoid, predictProb, computeLoss, cloneModel, sgdStep } from './model';

/**
 * Compute the gradient of the trigger batch loss w.r.t. perturbation δ
 * on the current accumulative batch.
 * 
 * The goal: find δ for the current batch S_t such that after the SGD update
 * θ_{t+1} = θ_t − β ∇L(S_t + δ; θ_t), the model becomes more vulnerable
 * to the trigger batch.
 * 
 * We approximate this by computing ∇_δ L(triggerBatch; θ_{t+1}(δ))
 * using the chain rule through one SGD step.
 */
export function computeAccumulativeGradient(
  model: { w: number[]; b: number },
  batchX: number[][], batchY: number[],
  triggerX: number[][], triggerY: number[],
  lr: number
): number[][] {
  const n = batchX.length;
  const d = model.w.length;
  const gradients: number[][] = [];

  for (let i = 0; i < n; i++) {
    const grad = new Array(d).fill(0);

    // Numerical gradient: ∂L_trigger(θ_{t+1}(x_i + δ_i))/∂δ_i
    const eps = 1e-4;
    for (let j = 0; j < d; j++) {
      // Perturb x_i[j] positively
      const batchPlus = batchX.map((x, k) => {
        if (k !== i) return x;
        const xp = [...x];
        xp[j] += eps;
        return xp;
      });
      const modelPlus = sgdStep(model, batchPlus, batchY, lr);
      const lossPlus = computeLoss(modelPlus, triggerX, triggerY);

      // Perturb x_i[j] negatively
      const batchMinus = batchX.map((x, k) => {
        if (k !== i) return x;
        const xm = [...x];
        xm[j] -= eps;
        return xm;
      });
      const modelMinus = sgdStep(model, batchMinus, batchY, lr);
      const lossMinus = computeLoss(modelMinus, triggerX, triggerY);

      // Central difference
      grad[j] = (lossPlus - lossMinus) / (2 * eps);
    }
    gradients.push(grad);
  }

  return gradients;
}

/**
 * PGD step: project perturbation onto L∞ ball of radius ε
 * x_perturbed = clip(x + α * sign(∇L), x - ε, x + ε)
 */
export function pgdStep(
  cleanX: number[][],
  perturbedX: number[][],
  gradients: number[][],
  epsilon: number,
  stepSize: number
): number[][] {
  return cleanX.map((xClean, i) => {
    const xNew = perturbedX[i].map((val, j) => {
      // FGSM-style signed gradient step
      const step = stepSize * Math.sign(gradients[i][j]);
      let newVal = val + step;
      // Project onto L∞ ball centered at xClean
      newVal = Math.max(xClean[j] - epsilon, Math.min(xClean[j] + epsilon, newVal));
      return newVal;
    });
    return xNew;
  });
}

/**
 * Craft the trigger batch: maximize loss after one SGD step on the trigger.
 * Uses PGD to find the worst-case perturbation of a clean batch.
 */
export function craftTriggerBatch(
  model: { w: number[]; b: number },
  cleanTriggerX: number[][], triggerY: number[],
  validX: number[][], validY: number[],
  epsilon: number,
  lr: number,
  pgdSteps: number = 10,
  pgdStepSize: number = 0.01
): number[][] {
  let perturbedX = cleanTriggerX.map(x => [...x]);

  for (let step = 0; step < pgdSteps; step++) {
    // Compute gradient of validation loss after one SGD step on trigger
    const gradients = computeTriggerGradient(model, perturbedX, triggerY, validX, validY, lr);
    // PGD step (maximize loss, so ascend)
    perturbedX = pgdStep(cleanTriggerX, perturbedX, gradients, epsilon, pgdStepSize);
  }

  return perturbedX;
}

/**
 * Gradient of validation loss w.r.t. trigger batch perturbation.
 * ∇_δ L_val(θ'(δ)) where θ' = θ - β ∇L(trigger + δ; θ)
 */
function computeTriggerGradient(
  model: { w: number[]; b: number },
  triggerX: number[][], triggerY: number[],
  validX: number[][], validY: number[],
  lr: number
): number[][] {
  const n = triggerX.length;
  const d = model.w.length;
  const gradients: number[][] = [];

  const eps = 1e-4;
  for (let i = 0; i < n; i++) {
    const grad = new Array(d).fill(0);
    for (let j = 0; j < d; j++) {
      const batchPlus = triggerX.map((x, k) => {
        if (k !== i) return x;
        const xp = [...x];
        xp[j] += eps;
        return xp;
      });
      const modelPlus = sgdStep(model, batchPlus, triggerY, lr);
      const lossPlus = computeLoss(modelPlus, validX, validY);

      const batchMinus = triggerX.map((x, k) => {
        if (k !== i) return x;
        const xm = [...x];
        xm[j] -= eps;
        return xm;
      });
      const modelMinus = sgdStep(model, batchMinus, triggerY, lr);
      const lossMinus = computeLoss(modelMinus, validX, validY);

      grad[j] = (lossPlus - lossMinus) / (2 * eps);
    }
    gradients.push(grad);
  }
  return gradients;
}
