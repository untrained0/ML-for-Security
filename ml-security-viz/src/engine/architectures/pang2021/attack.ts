/**
 * Pang et al. 2021 — Algorithm 1, accumulative poisoning attacks in online learning.
 *
 *   θ_0  ← burn-in (trainClean)
 *   P(S_T) ← S_T, or Eq. (6) crafted at θ_0 for a poisoned trigger
 *   for t = 0 … T−1:                       (the accumulative phase, one frame per round)
 *     G_t   = ∇_θ cos(∇L(S_val; θ_t), ∇L(P(S_T); θ_t))
 *     A_t(S_t) = argmax_{‖δ‖∞≤ε}  ĝ(A_t(S_t))ᵀ [ĝ(S_t) + λĜ_t]         (Eq. 9, C PGD steps)
 *     θ_{t+1} = θ_t − β v_{t+1},  v_{t+1} = μ' v_t + ∇L(A_t(S_t); θ_t)     (Eq. 8, with momentum)
 *     early stop if the monitor would notice (Eq. 7's γ, or §4's accuracy floor)
 *     optionally re-optimise P(S_T) at θ_{t+1}                            ("optimizing P", Table 1)
 *   θ_{T+1} = θ_T − β (μ v_T + ∇L(P(S_T); θ_T))   (the trigger, final frame)
 *
 * The victim's optimiser is the reference code's: SGD with momentum μ (0.9), a fresh optimiser at
 * θ_0 whose buffer v runs through the accumulative batches and into the trigger step, so the
 * trigger also carries the velocity the poisoned batches built up. μ' = μ, or 1.1 with the
 * paper's "weight momentum" trick (§4.1, Table 1). μ = 0 is plain SGD, Eq. 3.
 * The order of operations follows the reference implementation (online_accu_train.py). Batches
 * are drawn from a shuffled stream of the training pool; the trigger batch is held out of it.
 * Alongside the attacked model the loop keeps the clean counterfactual θ̃_t (same stream, no
 * perturbations) — the reference trajectory for Eq. (7)'s constraint, and the "vanilla poisoned
 * trigger" baseline of Table 1 is evaluated on it at the end.
 */

import type { TraceFrame } from '../registry';
import {
  type Net, type Grad, type Optim, cloneModel, learnerStep, stepDirection, applyGradient, lossGradient, batchStats, gradFromStats,
  computeAccuracy, computeLoss, evaluate, getModelState, packNet, paramDistance, statsEval, unpackNet,
  shuffled, toBatches,
} from './model';
import {
  alignValue, alignmentParamGradient, accumulativeTarget, craftAccumulativeBatch, craftTrigger,
  gnorm, gdot, planeCoords,
} from './gradient';

function linfNorm(a: number[][], b: number[][]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
  return m;
}

/**
 * Batches enter the trace for drawing and export only. 3072-d CIFAR batches are large, so they
 * are rounded to 1e-5 (0.3% of one 8-bit grey level) — far below anything visible or measured.
 */
const compact = (X: number[][]) => X.map(r => r.map(v => Math.round(v * 1e5) / 1e5));
/** Per-pixel PGD gradients are only drawn as directions: 4 significant digits. */
const compactGrad = (G: number[][]) => G.map(r => r.map(v => +v.toPrecision(4)));

/** The raw model for the frame, packed as float32 (an MLP on CIFAR-10 has ~200k parameters). */
const raw = (m: Net) => packNet(m, 32);

export function runAccumulativeAttack(
  dataset: any,
  cleanModelState: any,
  config: Record<string, any>,
  onProgress: (frame: TraceFrame) => void,
  onComplete: () => void,
  onError: (msg: string) => void,
  signal?: AbortSignal,
) {
  try {
    const { train, test } = dataset;
    const valid = dataset.valid?.X?.length ? dataset.valid : test;
    const testY: number[] = test.Y || test.y;

    const lr = config.learningRate ?? 1;
    const B = Math.max(2, Math.round(config.batchSize ?? 25));
    const T = Math.max(1, Math.round(config.numAccumBatches ?? 30));
    const epsilon = (config.epsilon255 ?? 16) / 255;
    const C = Math.max(1, Math.round(config.pgdSteps ?? 10));
    const lambda = config.lambda ?? 1;
    const gamma = config.gamma ?? 0.1;
    const maxAccDrop = config.maxAccDrop ?? 0.03;
    const poisonedTrigger = (config.triggerType ?? 'poisoned') === 'poisoned';
    const optimizeP = poisonedTrigger && (config.optimizeTrigger ?? 'yes') === 'yes';
    const box: [number, number] = dataset.bounds ?? [0, 1];
    const normalized = (config.normalizeGrads ?? 'yes') === 'yes';
    // Online phase: a fresh optimiser without weight decay, as online_accu_train.py builds it
    const opt: Optim = { lr, momentum: config.momentum ?? 0.9, weightDecay: 0 };
    const accumMomentum = (config.weightMomentum ?? 'no') === 'yes' ? 1.1 : opt.momentum;
    const vnorm = (v: Grad | null) => (v ? gnorm(v) : 0);

    if (!cleanModelState?.rawModel) throw new Error('Train the clean (burn-in) model first');
    const start: Net = unpackNet(cleanModelState.rawModel);
    if (start.sizes[0] !== train.X[0]?.length) throw new Error('The clean model was trained on another dataset — retrain it');

    // The stream after burn-in: the trigger batch S_T, then S_0 … S_{T−1} (cycling if T·B > N)
    const order = shuffled(train.X.length);
    const trigIdx = order.slice(0, B);
    const rest = order.slice(B);
    if (!rest.length) throw new Error('Training pool too small for this batch size');
    const streamIdx = Array.from({ length: T * B }, (_, k) => rest[k % rest.length]);
    const stream = toBatches(train.X, train.y, B, streamIdx);
    const trigCleanX: number[][] = trigIdx.map(i => train.X[i]);
    const trigY: number[] = trigIdx.map(i => train.y[i]);

    let model = cloneModel(start);
    let reference = cloneModel(start);        // θ̃_t: the same stream, unperturbed
    let vel: Grad | null = null;              // momentum buffers of the two optimisers
    let refVel: Grad | null = null;
    const test0 = evaluate(start, test.X, testY);
    const acc0 = test0.acc;
    const floor = acc0 - maxAccDrop;

    // S_val's forward/backward pass at the current θ_t, shared by G_t, the trigger and the readouts
    let val = batchStats(model, valid.X, valid.y);
    const val0 = statsEval(val, valid.y);

    // P(S_T): the clean batch, or Eq. (6) at θ_0 (the reference's --use_advtrigger)
    let trigX = poisonedTrigger
      ? craftTrigger(model, trigCleanX, trigCleanX, trigY, val.grad, epsilon, C, box, normalized)
      : trigCleanX.map(r => [...r]);

    const a0 = alignValue(val.grad, lossGradient(model, trigX, trigY), normalized);
    const s0 = getModelState(model, train.X, train.y, test.X, testY, test0);
    onProgress({
      iteration: 0,
      poisonX: [], poisonY: [],
      poisonedModel: s0,
      poisonedRawModel: raw(model),
      poisonedAccuracy: s0.testAccuracy,
      cleanAccuracy: acc0,
      objectiveValue: a0,
      gradients: [], gradientNorms: [],
      deltaW: 0,
      phase: 'baseline',
      batchIndex: 0,
      perturbationNorm: 0,
      secretAccuracy: val0.acc,
      triggerLoss: val0.loss,
      accumulatedDrift: 0,
      gradientAlignment: a0,
      stealthBudgetUsed: 0,
      stealthBudget: gamma,
      accuracyFloor: floor,
    });

    let t = 0;
    let stopped = false;

    const accumulate = () => {
      const batch = stream[t];
      const trig = batchStats(model, trigX, trigY);
      const clean = batchStats(model, batch.X, batch.y);

      // G_t and the Eq. (9) target d_t = ĝ(S_t) + λĜ_t
      const { G, u: gVal, v: gTrig } = alignmentParamGradient(val, trig, normalized);
      const { target, honest, accum } = accumulativeTarget(clean, G, lambda);

      // A_t(S_t) within ‖δ‖∞ ≤ ε, pixels kept in the box
      const { X: advX, gradients, objective } = craftAccumulativeBatch(model, batch.X, batch.y, target, epsilon, C, box);

      // Eq. (8) on the poisoned batch, and the clean counterfactual on S_t
      const cand = learnerStep({ net: model, v: vel }, advX, batch.y, opt, accumMomentum);
      const candidate = cand.net;
      ({ net: reference, v: refVel } = learnerStep({ net: reference, v: refVel }, batch.X, batch.y, opt));

      // Secrecy. Eq. (7): L(S_val; A(θ)) ≤ L(S_val; θ̃) + γ, plus §4's early stop on accuracy.
      // If this round would give the attacker away, feed the clean batch instead and stop.
      const candVal = batchStats(candidate, valid.X, valid.y);
      const budgetUsed = statsEval(candVal, valid.y).loss - computeLoss(reference, valid.X, valid.y);
      const candTest = evaluate(candidate, test.X, testY);
      let fedX = advX;
      let testEval = candTest;
      if (budgetUsed > gamma || candTest.acc < floor) {
        ({ net: model, v: vel } = learnerStep({ net: model, v: vel }, batch.X, batch.y, opt));
        fedX = batch.X;
        stopped = true;
        val = batchStats(model, valid.X, valid.y);
        testEval = undefined;
      } else {
        model = candidate;
        vel = cand.v;
        val = candVal;
      }
      const valEval = statsEval(val, valid.y);

      // "Optimizing P": re-craft the trigger against the new model
      if (optimizeP && !stopped) {
        trigX = craftTrigger(model, trigCleanX, trigX, trigY, val.grad, epsilon, C, box, normalized);
      }

      const state = getModelState(model, train.X, train.y, test.X, testY, testEval);
      const align = alignValue(val.grad, lossGradient(model, trigX, trigY), normalized);
      const drift = paramDistance(model, start);
      const [pVal, pTrig] = planeCoords(gVal, gTrig);
      const [pHonest, pAccum, pTarget] = planeCoords(honest, accum, target);

      onProgress({
        iteration: t + 1,
        poisonX: compact(fedX),
        poisonY: [...batch.y],
        poisonSource: batch.idx,
        poisonedModel: state,
        poisonedRawModel: raw(model),
        poisonedAccuracy: state.testAccuracy,
        // The honest counterfactual: accuracy had every batch been clean
        cleanAccuracy: computeAccuracy(reference, test.X, testY),
        // What the attacker minimises (Eq. 7) — lower is a stronger attack
        objectiveValue: align,
        gradients: stopped ? [] : compactGrad(gradients),
        gradientNorms: stopped ? [] : gradients.map(g => Math.sqrt(g.reduce((s, v) => s + v * v, 0))),
        deltaW: drift,
        phase: 'accumulative',
        batchIndex: t,
        perturbationNorm: linfNorm(batch.X, fedX),
        secretAccuracy: valEval.acc,
        triggerLoss: valEval.loss,
        accumulatedDrift: drift,
        gradientAlignment: align,
        alignmentGradNorm: gnorm(G),
        accumulativeObjective: objective,
        stealthBudgetUsed: budgetUsed,
        stealthBudget: gamma,
        accuracyFloor: floor,
        earlyStopped: stopped,
        velocityNorm: vnorm(vel),
        gradVal: pVal,
        gradTrigger: pTrig,
        gradBatch: pHonest,
        gradAccum: pAccum,
        gradTarget: pTarget,
      });
      t++;
    };

    const fireTrigger = () => {
      const preAcc = computeAccuracy(model, test.X, testY);
      const preLoss = statsEval(val, valid.y).loss;
      const gVal = val.grad;
      const gTrig = gradFromStats(batchStats(model, trigX, trigY));

      // θ_{T+1} = θ_T − β (μ v_T + ∇L(P(S_T); θ_T)) — the trigger step carries the momentum
      const dir = stepDirection({ net: model, v: vel }, gTrig, opt);
      const after = applyGradient(model, dir, lr);
      const state = getModelState(after, train.X, train.y, test.X, testY);
      const post = evaluate(after, valid.X, valid.y);
      const align = alignValue(gVal, gTrig, normalized);

      // Table 1's baseline on the same stream without accumulation: the clean counterfactual θ̃_T
      // fed the same kind of trigger (crafted against θ̃_T when poisoned)
      const vanillaX = poisonedTrigger
        ? craftTrigger(reference, trigCleanX, trigCleanX, trigY, lossGradient(reference, valid.X, valid.y), epsilon, C, box, normalized)
        : trigCleanX;
      const refPre = computeAccuracy(reference, test.X, testY);
      const refAfter = learnerStep({ net: reference, v: refVel }, vanillaX, trigY, opt).net;
      const [pVal, pTrig] = planeCoords(gVal, gTrig);
      const drift = paramDistance(after, start);

      onProgress({
        iteration: t + 1,
        poisonX: compact(trigX),
        poisonY: [...trigY],
        poisonSource: trigIdx,
        poisonedModel: state,
        poisonedRawModel: raw(after),
        poisonedAccuracy: state.testAccuracy,
        cleanAccuracy: refPre,
        objectiveValue: align,
        gradients: [], gradientNorms: [],
        deltaW: drift,
        phase: 'trigger',
        batchIndex: t,
        perturbationNorm: linfNorm(trigCleanX, trigX),
        secretAccuracy: post.acc,
        triggerLoss: post.loss,
        accumulatedDrift: drift,
        gradientAlignment: align,
        // First-order forecast of Eq. (6) for the step actually taken (−β∇L(S_val)ᵀ(μv_T + ∇L(P)))
        // vs. what it did
        predictedLossJump: -lr * gdot(gVal, dir),
        velocityNorm: vnorm(vel),
        actualLossJump: post.loss - preLoss,
        preTriggerAccuracy: preAcc,
        vanillaTriggerAccuracy: computeAccuracy(refAfter, test.X, testY),
        accumulatedRounds: t,
        earlyStopped: stopped,
        accuracyFloor: floor,
        gradVal: pVal,
        gradTrigger: pTrig,
      });
      onComplete();
    };

    const step = () => {
      if (signal?.aborted) return;
      try {
        if (!stopped && t < T) {
          accumulate();
          setTimeout(step, 0);
        } else {
          fireTrigger();
        }
      } catch (e: any) {
        onError(e?.message ?? String(e));
      }
    };
    setTimeout(step, 0);
  } catch (e: any) {
    onError(e?.message ?? String(e));
  }
}
