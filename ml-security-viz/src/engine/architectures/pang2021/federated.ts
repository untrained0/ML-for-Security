/**
 * Pang et al. 2021 — accumulative poisoning in federated learning (§3.3, Algorithm 2, §4.2).
 *
 * The server aggregates the gradients its clients submit and takes one SGD step per round. The
 * poisoners do not perturb images here — they submit gradients, and only the server's clipping
 * bounds what they send (the paper's η; Tables 3–5 sweep ℓ2 and ℓ∞ clip bounds).
 *
 *   θ_0 ← burn-in (trainClean, shared with the online setting)
 *   for t = 0 … T−1:                                     (the accumulative phase)
 *     Σ_n A_t(G_t^n) = w·∇L(S_t; θ_t) + λ ∇_θ cos(σ∇L(S_T; θ_t), ∇L(S_val; θ_t))      (Eq. 12)
 *     θ_{t+1} = θ_t − β·SGD-momentum(clip(Σ_n A_t(G_t^n)))                             (Eq. 11)
 *   θ_{T+1} = θ_T − β·SGD-momentum(clip(P))              (the trigger, Eq. 11 on the trigger batch)
 *
 * With the recovered offset of Eq. 14 the poisoners set the whole aggregate, so it is modelled as a
 * single update. The alignment is the cosine as in the reference code (feder_accu_train.py), or
 * Eq. 10's raw inner product, which also rewards a larger trigger gradient; the server's optimiser is SGD with momentum, whose buffer carries
 * into the trigger step. Two choices the reference makes explicit:
 *   • w — Algorithm 2's H_t keeps the honest gradients Σ_n G_t^n (w = 1); the reference code's
 *     craft_federated_NEW / _PoisonTri_FIX send only the λ-term (w = 0);
 *   • σ — a clean trigger S_T (σ = +1, Table 3 "accumulative phase + clean trigger") or the
 *     reversed poisoned trigger of Eq. 13, P = −s∇L(S_T) (σ = −1, Table 6; adapt_tensor_reverse).
 * Alongside, the server's honest trajectory θ̃_t (same stream, honest gradients, same clipping) is
 * the no-accumulation baseline: the same trigger fed at θ̃_T, and Table 3's direct poisoner — one
 * update −s_d∇L(S_val) at θ̃_T (craft_direct_NEW).
 *
 * Batch norm (ablation option): as in the reference, every forward pass in train mode updates the
 * running statistics — for the attacked model the S_T and S_val passes of each round's crafting,
 * for the honest server its batch, and for every trigger / direct update the batch it forwards.
 * S_val is a held-out training split, or (`federatedValSource` 'test') a 500-image batch of the test
 * split like the reference's teloader batch, which then overlaps the test set accuracy is read on.
 *
 * The monitor (§4's early stop) is simulated by the white-box attacker before each submission, as
 * in the online setting: a round whose update would take test accuracy more than Δacc below θ_0
 * is replaced by the honest aggregate, and the attacker stops accumulating.
 */

import type { TraceFrame } from '../registry';
import {
  type Net, type Grad, type Optim, type Learner, type BatchStats, cloneModel, batchStats, lossGradient,
  statsEval, evaluate, computeAccuracy, getModelState, optimStep, stepDirection, packNet, paramDistance,
  updateRunningStats,
  unpackNet, shuffled, toBatches,
} from './model';
import { alignmentParamGradient, alignValue, gaxpy, gdot, gnorm, gscale, planeCoords } from './gradient';

export type ClipNorm = 'none' | 'l2' | 'linf';

/**
 * The server's defence, torch.nn.utils.clip_grad_norm_: if ‖g‖_p > c, scale g by c/(‖g‖_p + 1e-6).
 * p = 2 or ∞ over all parameters at once (not an element-wise clamp).
 */
export function clipUpdate(g: Grad, norm: ClipNorm, c: number): { g: Grad; factor: number; before: number } {
  let total = 0;
  if (norm === 'linf') for (let j = 0; j < g.length; j++) total = Math.max(total, Math.abs(g[j]));
  else total = gnorm(g);
  if (norm === 'none') return { g, factor: 1, before: total };
  const coef = c / (total + 1e-6);
  return coef < 1 ? { g: gscale(g, coef), factor: coef, before: total } : { g, factor: 1, before: total };
}

/**
 * The aggregate the poisoners submit at θ (Eq. 12 with the recovered offset):
 *   U(θ) = w·∇L(S_t; θ) + λ ∇_θ a(σ∇L(S_T; θ), ∇L(S_val; θ)),
 * a = cos (normalised, the reference code) or the raw inner product of Eq. 10. U is the gradient
 * of Φ(θ) = w·L(S_t; θ) + λ·a(σ∇L(S_T), ∇L(S_val)); the λ-term is two Hessian-vector products
 * (gradient.ts), and a(u, −v) = −a(u, v) handles σ.
 */
export function federatedUpdate(
  val: BatchStats, trig: BatchStats, honest: Grad | null, lambda: number, sigma: 1 | -1, w: number,
  normalized = true,
): { U: Grad; G: Grad; u: Grad; v: Grad; align: number } {
  const { G, u, v } = alignmentParamGradient(val, trig, normalized);   // ∇_θ a(∇L(S_val), ∇L(S_T))
  let U = gscale(G, sigma * lambda);
  if (honest && w) U = gaxpy(U, honest, w);
  return { U, G: gscale(G, sigma), u, v, align: sigma * alignValue(u, v, normalized) };
}

export function runFederatedAttack(
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
    // S_val: the held-out training split, or a test batch as in the reference (test_batch_size 500)
    const valid = (config.federatedValSource ?? 'train') === 'test' || !dataset.valid?.X?.length
      ? (() => { const idx = shuffled(test.X.length).slice(0, 500); return { X: idx.map(i => test.X[i]), y: idx.map(i => (test.Y || test.y)[i]) }; })()
      : dataset.valid;
    const testY: number[] = test.Y || test.y;

    const B = Math.max(2, Math.round(config.batchSize ?? 25));
    const T = Math.max(1, Math.round(config.federatedRounds ?? 300));
    const lambda = config.federatedLambda ?? 0.3;
    const w = (config.federatedHonest ?? 'no') === 'yes' ? 1 : 0;
    const reversed = (config.federatedTrigger ?? 'clean') === 'poisoned';
    const sigma: 1 | -1 = reversed ? -1 : 1;
    const trigScale = config.triggerScale ?? 1;
    const directScale = config.directLossScale ?? 10;
    const clipNorm: ClipNorm = config.clipNorm ?? 'none';
    const clipValue = config.clipValue ?? 1;
    const maxAccDrop = config.federatedMaxAccDrop ?? 1;     // 1 = no monitor (the reference's federated runs)
    // The server's optimiser: SGD with momentum, no weight decay (feder_accu_train.py)
    const opt: Optim = { lr: config.serverLearningRate ?? 1, momentum: config.momentum ?? 0.9, weightDecay: 0 };
    const clip = (g: Grad) => clipUpdate(g, clipNorm, clipValue);
    const normalized = (config.federatedAlign ?? 'inner') === 'cosine';
    const align = (u: Grad, v: Grad) => sigma * alignValue(u, v, normalized);

    if (!cleanModelState?.rawModel) throw new Error('Train the clean (burn-in) model first');
    const start: Net = unpackNet(cleanModelState.rawModel);
    if (start.sizes[0] !== train.X[0]?.length) throw new Error('The clean model was trained on another dataset — retrain it');

    // The trigger batch S_T, and the honest clients' stream S_0 … S_{T−1} (cycling)
    const order = shuffled(train.X.length);
    const trigIdx = order.slice(0, B);
    const rest = order.slice(B);
    if (!rest.length) throw new Error('Training pool too small for this batch size');
    const stream = toBatches(train.X, train.y, B, Array.from({ length: T * B }, (_, k) => rest[k % rest.length]));
    const trigX: number[][] = trigIdx.map(i => train.X[i]);
    const trigY: number[] = trigIdx.map(i => train.y[i]);

    let L: Learner = { net: cloneModel(start), v: null };      // the attacked server model
    let R: Learner = { net: cloneModel(start), v: null };      // θ̃_t, honest aggregates only
    const acc0 = computeAccuracy(start, test.X, testY);
    const floor = acc0 - maxAccDrop;
    const frameEvery = Math.max(1, Math.ceil(T / 50));          // ≤ ~50 frames, however long T is

    let val = batchStats(L.net, valid.X, valid.y);
    let trig = batchStats(L.net, trigX, trigY);
    let frame = 0;
    let lastUpdate = { norm: 0, factor: 1, G: 0 };

    const emit = (t: number, stopped: boolean) => {
      const m = L.net;
      const state = getModelState(m, train.X, train.y, test.X, testY);
      const u = val.grad, v = trig.grad;
      const alignNow = align(u, v);
      const [pVal, pTrig] = planeCoords(u, gscale(v, sigma));
      const drift = paramDistance(m, start);
      const ve = m.bn ? evaluate(m, valid.X, valid.y) : statsEval(val, valid.y);
      onProgress({
        iteration: frame++,
        poisonX: [], poisonY: [],
        poisonedModel: state,
        poisonedRawModel: packNet(m, 32),
        poisonedAccuracy: state.testAccuracy,
        cleanAccuracy: t === 0 ? acc0 : computeAccuracy(R.net, test.X, testY),
        objectiveValue: alignNow,
        gradients: [], gradientNorms: [],
        deltaW: drift,
        phase: t === 0 ? 'baseline' : 'accumulative',
        batchIndex: t,
        perturbationNorm: 0,
        secretAccuracy: ve.acc,
        triggerLoss: ve.loss,
        accumulatedDrift: drift,
        gradientAlignment: alignNow,
        alignmentGradNorm: lastUpdate.G,
        accuracyFloor: floor,
        earlyStopped: stopped,
        velocityNorm: L.v ? gnorm(L.v) : 0,
        updateNorm: lastUpdate.norm,
        clipFactor: lastUpdate.factor,
        gradVal: pVal,
        gradTrigger: pTrig,
      });
    };

    emit(0, false);

    let t = 0;
    let stopped = false;

    /** A server step on the batch the model forwards (train mode), clipped: the forward also moves BN's running statistics. */
    const serverStep = (l: Learner, X: number[][], y: number[], scale = 1): Learner => {
      const s = batchStats(l.net, X, y);
      return optimStep({ net: updateRunningStats(l.net, s), v: l.v }, clip(scale === 1 ? s.grad : gscale(s.grad, scale)).g, opt);
    };

    const round = () => {
      const batch = stream[t];
      // Crafting forwards S_T then S_val in train mode (craft_federated_NEW): with BN they move the running statistics
      if (L.net.bn) L = { net: updateRunningStats(updateRunningStats(L.net, trig), val), v: L.v };
      let honest: Grad | null = null;
      if (w) {
        const sb = batchStats(L.net, batch.X, batch.y);
        L = { net: updateRunningStats(L.net, sb), v: L.v };
        honest = sb.grad;
      }
      const { U, G } = federatedUpdate(val, trig, honest, lambda, sigma, w, normalized);
      const c = clip(U);
      const candidate = optimStep(L, c.g, opt);
      R = serverStep(R, batch.X, batch.y);
      lastUpdate = { norm: c.before, factor: c.factor, G: gnorm(G) };
      // The monitor would notice: submit the honest aggregate this round instead, and stop
      if (maxAccDrop < 1 && computeAccuracy(candidate.net, test.X, testY) < floor) {
        L = serverStep(L, batch.X, batch.y);
        stopped = true;
      } else {
        L = candidate;
      }
      t++;
      val = batchStats(L.net, valid.X, valid.y);
      trig = batchStats(L.net, trigX, trigY);
      if (stopped || t % frameEvery === 0 || t === T) emit(t, stopped);
    };

    const fireTrigger = () => {
      const pre = evaluate(L.net, test.X, testY);
      const preLoss = L.net.bn ? evaluate(L.net, valid.X, valid.y).loss : statsEval(val, valid.y).loss;
      // The trigger batch's update: ∇L(S_T), or the reversed −s∇L(S_T) (Eq. 13, adapt_tensor_reverse)
      const tScale = reversed ? -trigScale : 1;
      const sT = batchStats(L.net, trigX, trigY);
      const g = clip(tScale === 1 ? sT.grad : gscale(sT.grad, tScale)).g;
      const dir = stepDirection(L, g, opt);
      const after = optimStep({ net: updateRunningStats(L.net, sT), v: L.v }, g, opt).net;
      const post = evaluate(after, valid.X, valid.y);
      const state = getModelState(after, train.X, train.y, test.X, testY);

      // No accumulation: the same trigger on the honest trajectory, and Table 3's direct poisoner
      const refPre = computeAccuracy(R.net, test.X, testY);
      const refAfter = serverStep(R, trigX, trigY, tScale).net;
      const direct = serverStep(R, valid.X, valid.y, -directScale).net;

      const u = val.grad;
      const [pVal, pTrig] = planeCoords(u, dir);
      const drift = paramDistance(after, start);
      onProgress({
        iteration: frame++,
        poisonX: trigX.map(r => r.map(v => Math.round(v * 1e5) / 1e5)),
        poisonY: [...trigY],
        poisonSource: trigIdx,
        poisonedModel: state,
        poisonedRawModel: packNet(after, 32),
        poisonedAccuracy: state.testAccuracy,
        cleanAccuracy: refPre,
        objectiveValue: align(u, trig.grad),
        gradients: [], gradientNorms: [],
        deltaW: drift,
        phase: 'trigger',
        batchIndex: t,
        perturbationNorm: 0,
        secretAccuracy: post.acc,
        triggerLoss: post.loss,
        accumulatedDrift: drift,
        gradientAlignment: align(u, trig.grad),
        // First-order forecast for the step actually taken, −β∇L(S_val)ᵀ(μv_T + clip(P)), vs. reality
        predictedLossJump: -opt.lr * gdot(u, dir),
        actualLossJump: post.loss - preLoss,
        preTriggerAccuracy: pre.acc,
        vanillaTriggerAccuracy: computeAccuracy(refAfter, test.X, testY),
        directAttackAccuracy: computeAccuracy(direct, test.X, testY),
        accumulatedRounds: t,
        earlyStopped: stopped,
        accuracyFloor: floor,
        velocityNorm: L.v ? gnorm(L.v) : 0,
        updateNorm: gnorm(g),
        gradVal: pVal,
        gradTrigger: pTrig,
      });
      onComplete();
    };

    const step = () => {
      if (signal?.aborted) return;
      try {
        if (!stopped && t < T) {
          round();
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
