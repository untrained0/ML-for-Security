/**
 * Pang et al. 2021 — Accumulative Poisoning Attack (Algorithm 1, online setting)
 *
 * Two phases:
 * 1. Accumulative — perturb each arriving batch so the update both keeps
 *    learning normally and steers θ toward a state where the trigger's
 *    gradient is anti-aligned with the validation gradient (Eq. 7).
 * 2. Trigger — release one crafted batch; a single SGD step on it collapses
 *    accuracy.
 *
 * The poisoned trigger P(S_T) is refined *during* accumulation, because Eq. (7)
 * is a joint minimisation over both the accumulative operator A and P.
 */

import type { TraceFrame } from '../registry';
import {
  cloneModel, sgdStep, applyGradient, lossGradient,
  computeAccuracy, computeLoss, getModelState, trainFull,
} from './model';
import {
  alignment, alignmentParamGradient, accumulativeTarget,
  refineAccumulativeBatch, refineTriggerBatch, gnorm,
} from './gradient';

/** Split data into sequential mini-batches */
function createBatches(X: number[][], y: number[], batchSize: number): { X: number[][]; y: number[] }[] {
  const batches: { X: number[][]; y: number[] }[] = [];
  for (let i = 0; i < X.length; i += batchSize) {
    const end = Math.min(i + batchSize, X.length);
    batches.push({ X: X.slice(i, end), y: y.slice(i, end) });
  }
  return batches;
}

/** L2 distance between two parameter vectors */
function modelDist(m1: { w: number[]; b: number }, m2: { w: number[]; b: number }): number {
  let sum = (m1.b - m2.b) ** 2;
  for (let i = 0; i < m1.w.length; i++) sum += (m1.w[i] - m2.w[i]) ** 2;
  return Math.sqrt(sum);
}

/** L∞ norm of a perturbation */
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
  onError: (msg: string) => void,
  signal?: AbortSignal
) {
  try {
    const { train, test, valid } = dataset;
    const lr = config.learningRate ?? 0.5;
    const epsilon = config.epsilon ?? 0.3;
    const numAccumBatches = config.numAccumBatches ?? 8;
    const lambda = config.lambda ?? 1.0;
    const gamma = config.gamma ?? 0.15;
    const secrecyThreshold = config.secrecyThreshold ?? 0.7;
    const pgdSteps = config.pgdSteps ?? 5;
    const pgdStepSize = config.pgdStepSize ?? 0.05;
    // Size batches so the requested number of accumulative rounds actually
    // exists. Asking for 40 rounds on 90 training points used to silently give
    // you 22, because the batch size was derived and then floored at 4.
    const batchSize = Math.max(2, Math.floor(train.X.length / (numAccumBatches + 1)));

    const testY = test.Y || test.y;
    // The attacker's own held-out split S_val. Falls back to the test split
    // only if the dataset has no validation split of its own.
    const validX = valid?.X || test.X;
    const validY = valid?.y || valid?.Y || testY;

    // θ_0 — reuse the model the app already trained and drew as "clean".
    // Retraining here would silently start the attack from different weights
    // than the clean boundary shown on the canvas.
    const startModel: { w: number[]; b: number } =
      cleanModelState?.rawModel?.w
        ? cloneModel(cleanModelState.rawModel)
        : cleanModelState?.w
        ? cloneModel(cleanModelState)
        : trainFull(train.X, train.y, lr, 100);

    const cleanAcc = computeAccuracy(startModel, test.X, testY);

    const batches = createBatches(train.X, train.y, batchSize);
    const triggerBatchIdx = Math.min(numAccumBatches, batches.length - 1);
    const accumBatches = batches.slice(0, triggerBatchIdx);
    const triggerCleanX = batches[triggerBatchIdx].X.map((x: number[]) => [...x]);
    const triggerY = batches[triggerBatchIdx].y;

    let currentModel = cloneModel(startModel);
    // The honest reference trajectory: what θ would have been on clean batches.
    // Eq. (7)'s stealth budget is measured against this, not against a fixed number.
    let referenceModel = cloneModel(startModel);
    // P(S_T) starts as the clean trigger batch (Algorithm 1, initialisation).
    let triggerX = triggerCleanX.map(x => [...x]);

    let iter = 0;
    let accumIdx = 0;
    let stopped = false;

    const baselineState = getModelState(currentModel, train.X, train.y, test.X, testY);
    const baselineAlign = alignment(currentModel, validX, validY, triggerX, triggerY);

    onProgress({
      iteration: iter,
      poisonX: [],
      poisonY: [],
      poisonedModel: baselineState,
      poisonedRawModel: cloneModel(currentModel),
      poisonedAccuracy: baselineState.testAccuracy,
      objectiveValue: baselineAlign,
      gradients: [],
      gradientNorms: [],
      cleanAccuracy: cleanAcc,
      deltaW: 0,
      phase: 'baseline',
      batchIndex: 0,
      perturbationNorm: 0,
      secretAccuracy: computeAccuracy(currentModel, validX, validY),
      triggerLoss: computeLoss(currentModel, validX, validY),
      accumulatedDrift: 0,
      gradientAlignment: baselineAlign,
      stealthBudgetUsed: 0,
    });

    iter = 1;

    const runStep = () => {
      if (signal?.aborted) return;
      try {
        if (!stopped && accumIdx < accumBatches.length) {
          // ── ACCUMULATIVE PHASE (Algorithm 1, inner loop) ──
          const batch = accumBatches[accumIdx];
          const cleanBatchX = batch.X.map((x: number[]) => [...x]);

          // G_t = ∇_θ ⟨∇L(S_val;θ_t), ∇L(P(S_T);θ_t)⟩
          const alignGrad = alignmentParamGradient(
            currentModel, validX, validY, triggerX, triggerY
          );

          // d_t = ∇L(S_t;θ_t) + λ G_t — learn normally *and* accumulate
          const target = accumulativeTarget(
            currentModel, cleanBatchX, batch.y, alignGrad, lambda
          );

          // A_t(S_t): maximise H_t = ⟨∇L(A_t(S_t);θ_t), d_t⟩ inside ‖δ‖∞ ≤ ε
          const { X: perturbedX, gradients } = refineAccumulativeBatch(
            currentModel, cleanBatchX, batch.y, target,
            epsilon, pgdSteps, pgdStepSize
          );

          // P(S_T): jointly refined against the *current* θ_t
          triggerX = refineTriggerBatch(
            currentModel, triggerCleanX, triggerY, triggerX,
            validX, validY, epsilon, pgdSteps, pgdStepSize
          );

          // θ_{t+1} = θ_t − β ∇L(A_t(S_t); θ_t)
          const nextModel = sgdStep(currentModel, perturbedX, batch.y, lr);
          // …and the clean counterfactual for the stealth budget.
          referenceModel = sgdStep(referenceModel, cleanBatchX, batch.y, lr);

          // Stealth check — Eq. (7)'s constraint, plus the paper's early stop:
          //   L(S_val; θ_{t+1}) ≤ L(S_val; θ̃_{t+1}) + γ
          const poisonedValLoss = computeLoss(nextModel, validX, validY);
          const referenceValLoss = computeLoss(referenceModel, validX, validY);
          const budgetUsed = poisonedValLoss - referenceValLoss;
          const valAcc = computeAccuracy(nextModel, validX, validY);

          let appliedX = perturbedX;
          if (budgetUsed > gamma || valAcc < secrecyThreshold) {
            // Over budget: fall back to the clean update for this round and
            // stop accumulating, matching the paper's early-stopping rule.
            currentModel = sgdStep(currentModel, cleanBatchX, batch.y, lr);
            appliedX = cleanBatchX;
            stopped = true;
          } else {
            currentModel = nextModel;
          }

          const modelState = getModelState(currentModel, train.X, train.y, test.X, testY);
          const drift = modelDist(currentModel, startModel);
          const align = alignment(currentModel, validX, validY, triggerX, triggerY);
          const gVal = lossGradient(currentModel, validX, validY);
          const gTrig = lossGradient(currentModel, triggerX, triggerY);
          const gBatch = lossGradient(currentModel, cleanBatchX, batch.y);

          onProgress({
            iteration: iter,
            poisonX: appliedX,
            poisonY: [...batch.y],
            poisonedModel: modelState,
            poisonedRawModel: cloneModel(currentModel),
            poisonedAccuracy: modelState.testAccuracy,
            // The attacker's real objective is the alignment, so that is what
            // the timeline plots — lower (more negative) is a better attack.
            objectiveValue: align,
            gradients,
            gradientNorms: gradients.map(g => Math.sqrt(g.reduce((s, v) => s + v * v, 0))),
            // The honest counterfactual, re-measured each round: this is what
            // accuracy would be had every batch been clean. Comparing against a
            // single frozen number hides the fact that the victim is still
            // learning while the attacker accumulates.
            cleanAccuracy: computeAccuracy(referenceModel, test.X, testY),
            deltaW: drift,
            phase: 'accumulative',
            batchIndex: accumIdx,
            perturbationNorm: linfNorm(cleanBatchX, appliedX),
            secretAccuracy: computeAccuracy(currentModel, validX, validY),
            triggerLoss: computeLoss(currentModel, validX, validY),
            accumulatedDrift: drift,
            gradientAlignment: align,
            alignmentGradNorm: gnorm(alignGrad),
            stealthBudgetUsed: budgetUsed,
            stealthBudget: gamma,
            earlyStopped: stopped,
            gradVal: [...gVal.gw],
            gradTrigger: [...gTrig.gw],
            gradBatch: [...gBatch.gw],
            gradAccum: [...alignGrad.gw],
            gradTarget: [...target.gw],
          });

          iter++;
          accumIdx++;
          setTimeout(runStep, 0);

        } else {
          // ── TRIGGER PHASE ──
          // A final polish of P(S_T) at θ_T, then one online step on it.
          triggerX = refineTriggerBatch(
            currentModel, triggerCleanX, triggerY, triggerX,
            validX, validY, epsilon, pgdSteps * 2, pgdStepSize
          );

          const preTriggerAlign = alignment(currentModel, validX, validY, triggerX, triggerY);
          const preTriggerValLoss = computeLoss(currentModel, validX, validY);

          // θ_{T+1} = θ_T − β ∇L(P(S_T); θ_T)
          const triggerGrad = lossGradient(currentModel, triggerX, triggerY);
          const postTriggerModel = applyGradient(currentModel, triggerGrad, lr);

          const triggerState = getModelState(postTriggerModel, train.X, train.y, test.X, testY);
          const drift = modelDist(postTriggerModel, startModel);
          const postValLoss = computeLoss(postTriggerModel, validX, validY);

          onProgress({
            iteration: iter,
            poisonX: triggerX,
            poisonY: [...triggerY],
            poisonedModel: triggerState,
            poisonedRawModel: cloneModel(postTriggerModel),
            poisonedAccuracy: triggerState.testAccuracy,
            objectiveValue: preTriggerAlign,
            gradients: [],
            gradientNorms: [],
            cleanAccuracy: computeAccuracy(referenceModel, test.X, testY),
            deltaW: drift,
            phase: 'trigger',
            batchIndex: accumIdx,
            perturbationNorm: linfNorm(triggerCleanX, triggerX),
            secretAccuracy: computeAccuracy(postTriggerModel, validX, validY),
            triggerLoss: postValLoss,
            accumulatedDrift: drift,
            gradientAlignment: preTriggerAlign,
            // First-order prediction from Eq. (7) vs. what actually happened —
            // the two should agree closely for a single small step.
            predictedLossJump: -lr * preTriggerAlign,
            actualLossJump: postValLoss - preTriggerValLoss,
            earlyStopped: stopped,
            gradVal: [...lossGradient(currentModel, validX, validY).gw],
            gradTrigger: [...triggerGrad.gw],
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
