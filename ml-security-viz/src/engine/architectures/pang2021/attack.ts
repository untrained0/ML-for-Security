/**
 * Pang et al. 2021 — Accumulative Poisoning Attack
 * 
 * Two-phase attack on online learning:
 * 1. Accumulative Phase: subtly perturb sequential batches to drift θ
 *    while maintaining accuracy (secrecy constraint)
 * 2. Trigger Phase: inject a single crafted trigger batch that causes
 *    catastrophic accuracy drop
 */

import type { TraceFrame } from '../registry';
import {
  cloneModel, sgdStep, computeAccuracy, computeLoss,
  getModelState, trainFull
} from './model';
import { computeAccumulativeGradient, pgdStep, craftTriggerBatch } from './gradient';

/** Split data into sequential mini-batches */
function createBatches(X: number[][], y: number[], batchSize: number): { X: number[][]; y: number[] }[] {
  const batches: { X: number[][]; y: number[] }[] = [];
  for (let i = 0; i < X.length; i += batchSize) {
    const end = Math.min(i + batchSize, X.length);
    batches.push({
      X: X.slice(i, end),
      y: y.slice(i, end),
    });
  }
  return batches;
}

/** Compute L2 norm distance between two weight vectors */
function modelDist(m1: { w: number[]; b: number }, m2: { w: number[]; b: number }): number {
  let sum = 0;
  for (let i = 0; i < m1.w.length; i++) {
    sum += (m1.w[i] - m2.w[i]) ** 2;
  }
  sum += (m1.b - m2.b) ** 2;
  return Math.sqrt(sum);
}

/** L∞ norm of perturbation */
function linfNorm(cleanX: number[][], perturbedX: number[][]): number {
  let maxDiff = 0;
  for (let i = 0; i < cleanX.length; i++) {
    for (let j = 0; j < cleanX[i].length; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(perturbedX[i][j] - cleanX[i][j]));
    }
  }
  return maxDiff;
}

export function runAccumulativeAttack(
  dataset: any,
  cleanModelState: any,
  config: Record<string, any>,
  onProgress: (frame: TraceFrame) => void,
  onComplete: () => void,
  onError: (msg: string) => void
) {
  try {
    const { train, test, valid } = dataset;
    const lr = config.learningRate || 0.5;
    const epsilon = config.epsilon || 0.3;
    const numAccumBatches = config.numAccumBatches || 8;
    const batchSize = config.batchSize || Math.max(4, Math.floor(train.X.length / (numAccumBatches + 1)));
    const secrecyThreshold = config.secrecyThreshold || 0.7;
    const pgdSteps = config.pgdSteps || 5;
    const pgdStepSize = config.pgdStepSize || 0.05;

    // Step 1: Train clean model as baseline
    const cleanModel = trainFull(train.X, train.y, lr, 100);
    const cleanAcc = computeAccuracy(cleanModel, test.X, test.Y || test.y);
    const testY = test.Y || test.y;
    const validX = valid?.X || test.X;
    const validY = valid?.Y || valid?.y || testY;

    // Create sequential batches from training data (simulating online learning)
    const batches = createBatches(train.X, train.y, batchSize);

    // Reserve the last batch as the trigger batch base
    const triggerBatchIdx = Math.min(numAccumBatches, batches.length - 1);
    const accumBatches = batches.slice(0, triggerBatchIdx);
    const triggerCleanX = batches[triggerBatchIdx]?.X || batches[batches.length - 1].X;
    const triggerY = batches[triggerBatchIdx]?.y || batches[batches.length - 1].y;

    // Start from the clean model state
    let currentModel = cloneModel(cleanModel);
    let iter = 0;

    // Emit baseline frame
    const baselineState = getModelState(currentModel, train.X, train.y, test.X, testY);
    onProgress({
      iteration: iter,
      poisonX: [],
      poisonY: [],
      poisonedModel: baselineState,
      poisonedRawModel: cloneModel(currentModel),
      poisonedAccuracy: baselineState.testAccuracy,
      objectiveValue: computeLoss(currentModel, validX, validY),
      gradients: [],
      gradientNorms: [],
      cleanAccuracy: cleanAcc,
      deltaW: 0,
      phase: 'baseline',
      batchIndex: 0,
      perturbationNorm: 0,
      secretAccuracy: baselineState.testAccuracy,
      triggerLoss: computeLoss(currentModel, validX, validY),
      accumulatedDrift: 0,
    });

    iter = 1;
    let accumIdx = 0;

    const runStep = () => {
      try {
        if (accumIdx < accumBatches.length) {
          // ── ACCUMULATIVE PHASE ──
          const batch = accumBatches[accumIdx];
          const cleanBatchX = batch.X.map(x => [...x]);

          // Compute gradient of trigger loss w.r.t. accumulative perturbation
          const grads = computeAccumulativeGradient(
            currentModel, batch.X, batch.y,
            triggerCleanX, triggerY, lr
          );

          // PGD: perturb the batch to maximize future trigger damage
          let perturbedX = batch.X.map(x => [...x]);
          for (let s = 0; s < pgdSteps; s++) {
            const stepGrads = computeAccumulativeGradient(
              currentModel, perturbedX, batch.y,
              triggerCleanX, triggerY, lr
            );
            perturbedX = pgdStep(cleanBatchX, perturbedX, stepGrads, epsilon, pgdStepSize);
          }

          // Check secrecy: if accuracy drops too much, skip perturbation
          const testModelPerturbed = sgdStep(currentModel, perturbedX, batch.y, lr);
          const perturbedAcc = computeAccuracy(testModelPerturbed, test.X, testY);

          if (perturbedAcc >= secrecyThreshold) {
            // Accept the perturbation
            currentModel = testModelPerturbed;
          } else {
            // Secrecy violated — use clean batch instead (early stopping of this batch)
            currentModel = sgdStep(currentModel, batch.X, batch.y, lr);
            perturbedX = cleanBatchX;
          }

          const pertNorm = linfNorm(cleanBatchX, perturbedX);
          const modelState = getModelState(currentModel, train.X, train.y, test.X, testY);
          const drift = modelDist(currentModel, cleanModel);

          // Collect all perturbed points as "poison points" for visualization
          onProgress({
            iteration: iter,
            poisonX: perturbedX,
            poisonY: [...batch.y],
            poisonedModel: modelState,
            poisonedRawModel: cloneModel(currentModel),
            poisonedAccuracy: modelState.testAccuracy,
            objectiveValue: computeLoss(currentModel, validX, validY),
            gradients: grads,
            gradientNorms: grads.map(g => Math.sqrt(g.reduce((s, v) => s + v * v, 0))),
            cleanAccuracy: cleanAcc,
            deltaW: drift,
            phase: 'accumulative',
            batchIndex: accumIdx,
            perturbationNorm: pertNorm,
            secretAccuracy: modelState.testAccuracy,
            triggerLoss: computeLoss(currentModel, triggerCleanX, triggerY),
            accumulatedDrift: drift,
          });

          iter++;
          accumIdx++;
          setTimeout(runStep, 0);

        } else {
          // ── TRIGGER PHASE ──
          // Craft the trigger batch using PGD
          const craftedTriggerX = craftTriggerBatch(
            currentModel, triggerCleanX, triggerY,
            validX, validY,
            epsilon, lr, pgdSteps * 2, pgdStepSize
          );

          // Apply the trigger: one SGD step on the crafted trigger batch
          const postTriggerModel = sgdStep(currentModel, craftedTriggerX, triggerY, lr);

          const triggerState = getModelState(postTriggerModel, train.X, train.y, test.X, testY);
          const drift = modelDist(postTriggerModel, cleanModel);

          onProgress({
            iteration: iter,
            poisonX: craftedTriggerX,
            poisonY: [...triggerY],
            poisonedModel: triggerState,
            poisonedRawModel: cloneModel(postTriggerModel),
            poisonedAccuracy: triggerState.testAccuracy,
            objectiveValue: computeLoss(postTriggerModel, validX, validY),
            gradients: [],
            gradientNorms: [],
            cleanAccuracy: cleanAcc,
            deltaW: drift,
            phase: 'trigger',
            batchIndex: accumIdx,
            perturbationNorm: linfNorm(triggerCleanX, craftedTriggerX),
            secretAccuracy: triggerState.testAccuracy,
            triggerLoss: computeLoss(postTriggerModel, validX, validY),
            accumulatedDrift: drift,
          });

          onComplete();
        }
      } catch (e: any) {
        onError(e.message);
      }
    };

    setTimeout(runStep, 0);
  } catch (e: any) {
    onError(e.message);
  }
}
