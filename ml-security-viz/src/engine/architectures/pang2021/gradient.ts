/**
 * Pang et al. 2021 — Accumulative poisoning objectives (Algorithm 1, online setting)
 *
 * The attack is built on the *gradient alignment* of Eq. (7):
 *
 *   min_{P, A}  ∇_θ L(S_val; A(θ_T))ᵀ ∇_θ L(P(S_T); A(θ_T))
 *   s.t.        L(S_val; A(θ_T)) ≤ L(S_val; θ_T) + γ
 *
 * Why an inner product? The trigger update is θ_{T+1} = θ_T − β ∇L(P(S_T); θ_T),
 * so to first order
 *
 *   L(S_val; θ_{T+1}) ≈ L(S_val; θ_T) − β ⟨∇L(S_val; θ_T), ∇L(P(S_T); θ_T)⟩.
 *
 * Driving that inner product negative is exactly what makes one trigger step
 * *raise* validation loss. The accumulative phase exists to steer θ into a
 * region where such an anti-aligned trigger gradient is available.
 *
 * Algorithm 1 realises this with a first-order surrogate per round:
 *   G_t = ∇_θ [ ⟨∇L(S_val; θ_t), ∇L(P(S_T); θ_t)⟩ ]
 *   H_t = ⟨ ∇L(A_t(S_t); θ_t), ∇L(S_t; θ_t) + λ G_t ⟩      ← maximised over δ
 * so that the resulting update −β∇L(A_t(S_t)) both keeps learning normally
 * (the ∇L(S_t) term) and descends the alignment (the λG_t term).
 */

import { lossGradient, cloneModel, type Grad } from './model';

type Model = { w: number[]; b: number };

// ── small parameter-space vector helpers ──

export function gdot(a: Grad, b: Grad): number {
  let s = a.gb * b.gb;
  for (let j = 0; j < a.gw.length; j++) s += a.gw[j] * b.gw[j];
  return s;
}

export function gnorm(g: Grad): number {
  return Math.sqrt(gdot(g, g));
}

/** a + s·b */
export function gaxpy(a: Grad, b: Grad, s: number): Grad {
  return {
    gw: a.gw.map((v, j) => v + s * b.gw[j]),
    gb: a.gb + s * b.gb,
  };
}

/**
 * The alignment scalar from Eq. (7), evaluated at θ:
 *   a(θ) = ⟨∇L(S_val; θ), ∇L(S_trigger; θ)⟩
 * Negative values mean a trigger step would increase validation loss.
 */
export function alignment(
  model: Model,
  validX: number[][], validY: number[],
  trigX: number[][], trigY: number[],
  l2: number = 0.01
): number {
  return gdot(
    lossGradient(model, validX, validY, l2),
    lossGradient(model, trigX, trigY, l2)
  );
}

/**
 * The same alignment, normalised to a cosine.
 *
 * The reference implementation normalises its gradients before taking dot
 * products, and it matters here for the same reason: the raw inner product is
 * dominated by ‖∇L‖, which collapses as the model converges. Using the cosine
 * for the *optimisation signal* keeps λ meaning the same thing at every round.
 * The raw inner product is still what gets reported, because that is the
 * quantity in the first-order loss expansion.
 */
export function cosineAlignment(
  model: Model,
  validX: number[][], validY: number[],
  trigX: number[][], trigY: number[],
  l2: number = 0.01
): number {
  const a = lossGradient(model, validX, validY, l2);
  const b = lossGradient(model, trigX, trigY, l2);
  const na = gnorm(a), nb = gnorm(b);
  if (na < 1e-12 || nb < 1e-12) return 0;
  return gdot(a, b) / (na * nb);
}

/**
 * G_t = ∇_θ a(θ) — how the alignment changes as the parameters move.
 *
 * This is a second-order quantity (it differentiates a product of gradients),
 * computed here by central differences over the d+1 parameters. That is cheap
 * for a 2-D visualiser and avoids hand-rolling Hessian-vector products.
 */
export function alignmentParamGradient(
  model: Model,
  validX: number[][], validY: number[],
  trigX: number[][], trigY: number[],
  l2: number = 0.01,
  h: number = 1e-4
): Grad {
  const score = (m: Model) => cosineAlignment(m, validX, validY, trigX, trigY, l2);
  const gw = new Array(model.w.length).fill(0);

  for (let j = 0; j < model.w.length; j++) {
    const plus = cloneModel(model);
    plus.w[j] += h;
    const minus = cloneModel(model);
    minus.w[j] -= h;
    gw[j] = (score(plus) - score(minus)) / (2 * h);
  }

  const bPlus = cloneModel(model);
  bPlus.b += h;
  const bMinus = cloneModel(model);
  bMinus.b -= h;
  const gb = (score(bPlus) - score(bMinus)) / (2 * h);

  return { gw, gb };
}

/** g / ‖g‖ — the zero vector maps to itself. */
export function gnormalize(g: Grad): Grad {
  const n = gnorm(g);
  if (n < 1e-12) return { gw: g.gw.map(() => 0), gb: 0 };
  return { gw: g.gw.map(v => v / n), gb: g.gb / n };
}

/**
 * The direction the attacker wants this round's update to take:
 *   d_t = ĝ(S_t; θ_t) + λ Ĝ_t          (both terms unit-length)
 *
 * Normalising both parts before mixing is what makes λ a genuine dial. With
 * raw gradients the honest term and the accumulation term differ by orders of
 * magnitude — whichever happens to be larger wins outright, and λ barely
 * registers. The reference implementation combines L2-normalised gradients for
 * the same reason.
 */
export function accumulativeTarget(
  model: Model,
  batchX: number[][], batchY: number[],
  alignGrad: Grad,
  lambda: number,
  l2: number = 0.01
): Grad {
  const honest = gnormalize(lossGradient(model, batchX, batchY, l2));
  const accumulate = gnormalize(alignGrad);
  return gaxpy(honest, accumulate, lambda);
}

/**
 * H_t = ⟨ĝ(A_t(S_t); θ_t), d_t⟩ — maximised over the batch perturbation.
 * Normalised on the left too, so the search optimises *direction* rather than
 * simply inflating the gradient's magnitude.
 */
export function accumulativeObjective(
  model: Model,
  batchX: number[][], batchY: number[],
  target: Grad,
  l2: number = 0.01
): number {
  return gdot(gnormalize(lossGradient(model, batchX, batchY, l2)), target);
}

/**
 * ∇_δ of a scalar objective that depends on a batch's feature matrix,
 * by central differences — one entry per (sample, feature).
 */
export function batchGradient(
  objective: (X: number[][]) => number,
  X: number[][],
  h: number = 1e-4
): number[][] {
  return X.map((row, i) =>
    row.map((_, j) => {
      const plus = X.map((r, k) => (k === i ? r.map((v, m) => (m === j ? v + h : v)) : r));
      const minus = X.map((r, k) => (k === i ? r.map((v, m) => (m === j ? v - h : v)) : r));
      return (objective(plus) - objective(minus)) / (2 * h);
    })
  );
}

/**
 * One PGD step with an L∞ trust region:
 *   δ ← Π_{‖δ‖∞ ≤ ε} ( δ + direction · α · sign(∇) )
 * `direction` is +1 to ascend the objective, −1 to descend it.
 */
export function pgdStep(
  cleanX: number[][],
  currentX: number[][],
  gradients: number[][],
  epsilon: number,
  stepSize: number,
  direction: 1 | -1 = 1
): number[][] {
  return currentX.map((row, i) =>
    row.map((val, j) => {
      const stepped = val + direction * stepSize * Math.sign(gradients[i][j]);
      return Math.max(cleanX[i][j] - epsilon, Math.min(cleanX[i][j] + epsilon, stepped));
    })
  );
}

/**
 * Refine the accumulative batch A_t(S_t): maximise H_t inside the ε-ball.
 */
export function refineAccumulativeBatch(
  model: Model,
  cleanBatchX: number[][], batchY: number[],
  target: Grad,
  epsilon: number, steps: number, stepSize: number,
  l2: number = 0.01
): { X: number[][]; gradients: number[][] } {
  let current = cleanBatchX.map(x => [...x]);
  let gradients: number[][] = current.map(r => r.map(() => 0));

  for (let s = 0; s < steps; s++) {
    gradients = batchGradient(
      (X) => accumulativeObjective(model, X, batchY, target, l2),
      current
    );
    current = pgdStep(cleanBatchX, current, gradients, epsilon, stepSize, 1);
  }

  return { X: current, gradients };
}

/**
 * Refine the poisoned trigger P(S_T): drive the alignment at θ_t as negative
 * as possible inside the ε-ball. Updated every round alongside A_t, because
 * Eq. (7) is a *joint* minimisation over P and A — accumulating against a
 * trigger you then throw away is the common way to get this wrong.
 */
export function refineTriggerBatch(
  model: Model,
  cleanTriggerX: number[][], triggerY: number[],
  currentTriggerX: number[][],
  validX: number[][], validY: number[],
  epsilon: number, steps: number, stepSize: number,
  l2: number = 0.01
): number[][] {
  let current = currentTriggerX.map(x => [...x]);
  const validGrad = lossGradient(model, validX, validY, l2);

  for (let s = 0; s < steps; s++) {
    const grads = batchGradient(
      (X) => gdot(validGrad, lossGradient(model, X, triggerY, l2)),
      current
    );
    current = pgdStep(cleanTriggerX, current, grads, epsilon, stepSize, -1);
  }

  return current;
}
