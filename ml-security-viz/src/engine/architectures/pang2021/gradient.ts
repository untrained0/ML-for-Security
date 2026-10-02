/**
 * Pang et al. 2021 — the attack objectives and their exact gradients (online setting, §3.2).
 *
 * Eq. (6), the vanilla poisoner at round T: to first order in β,
 *   L(S_val; θ_{T+1}) ≈ L(S_val; θ_T) − β ∇L(S_val; θ_T)ᵀ ∇L(P(S_T); θ_T),
 * so raising the validation loss means making that inner product as negative as possible.
 *
 * Eq. (7) adds the accumulative phase A, which moves θ_T itself somewhere the trigger does more
 * damage:  min_{P,A} ∇L(S_val; A(θ_T))ᵀ ∇L(P(S_T); A(θ_T)).
 *
 * Eq. (9) / Algorithm 1 craft each accumulative batch by (A(θ_T) greedily replaced by θ_t)
 *   max_{A_t}  ∇L(A_t(S_t); θ_t)ᵀ [ ∇L(S_t; θ_t) + λ G_t ],
 *   G_t = ∇_θ [ ∇L(S_val; θ_t)ᵀ ∇L(P(S_T); θ_t) ],
 * the bracket being a stop-gradient target. Algorithm 1 optionally normalises the three gradients
 * "to concentrate on angular distances" (the reference code does); `normalized` switches between
 * that cosine and Eq. 7's raw inner product.
 *
 * Every derivative here is exact for either victim (model.ts): ∇_θ L from backprop, Hessian-vector
 * products and pixel gradients of directional derivatives from the R-operator — no finite
 * differences, even with the MLP's ~200k parameters.
 */

import { type Net, type Grad, type BatchStats, batchStats, gradFromStats, hessVec, pixelGradient } from './model';

// ── parameter-space vector helpers ──

export function gdot(a: Grad, b: Grad): number {
  let s = 0;
  for (let j = 0; j < a.length; j++) s += a[j] * b[j];
  return s;
}

export function gnorm(g: Grad): number {
  return Math.sqrt(gdot(g, g));
}

/** a + s·b */
export function gaxpy(a: Grad, b: Grad, s: number): Grad {
  const out = new Float64Array(a.length);
  for (let j = 0; j < a.length; j++) out[j] = a[j] + s * b[j];
  return out;
}

export function gscale(a: Grad, s: number): Grad {
  const out = new Float64Array(a.length);
  for (let j = 0; j < a.length; j++) out[j] = a[j] * s;
  return out;
}

/** g / ‖g‖ — the zero vector maps to itself. */
export function gnormalize(g: Grad): Grad {
  const n = gnorm(g);
  return n < 1e-12 ? gscale(g, 0) : gscale(g, 1 / n);
}

export function cosine(a: Grad, b: Grad): number {
  const na = gnorm(a), nb = gnorm(b);
  return na < 1e-12 || nb < 1e-12 ? 0 : gdot(a, b) / (na * nb);
}

/**
 * ∂cos(u, v)/∂v = (û − cos·v̂)/‖v‖ — how the cosine responds to a change in v. Used both for the
 * parameter gradient G_t and for backpropagating a cosine into the pixels of the batch behind v.
 */
function cosineGradWrtSecond(u: Grad, v: Grad): Grad {
  const nu = gnorm(u), nv = gnorm(v);
  if (nu < 1e-12 || nv < 1e-12) return gscale(v, 0);
  const c = gdot(u, v) / (nu * nv);
  return gaxpy(gscale(u, 1 / (nu * nv)), v, -c / (nv * nv));
}

/**
 * The alignment of Eq. (7) at θ between u = ∇L(S_val; θ) and v = ∇L(P(S_T); θ): the cosine when
 * the gradients are normalised (Algorithm 1's option, and the reference code), else the raw inner
 * product uᵀv of Eq. (7) itself. Negative means a step on the trigger raises the validation loss.
 */
export function alignValue(u: Grad, v: Grad, normalized: boolean): number {
  return normalized ? cosine(u, v) : gdot(u, v);
}

/** ∂alignment/∂v (and, by symmetry, ∂/∂u with the arguments swapped). */
function alignGradWrtSecond(u: Grad, v: Grad, normalized: boolean): Grad {
  return normalized ? cosineGradWrtSecond(u, v) : u;
}

export function alignment(
  m: Net, valX: number[][], valY: number[], trigX: number[][], trigY: number[], normalized = true,
): number {
  return alignValue(gradFromStats(batchStats(m, valX, valY)), gradFromStats(batchStats(m, trigX, trigY)), normalized);
}

/**
 * G_t = ∇_θ a(u(θ), v(θ)) with u = ∇L(S_val), v = ∇L(P(S_T)). By the chain rule through the
 * two gradients (whose Jacobians are the Hessians):
 *   G_t = H_val · ∂a/∂u + H_P · ∂a/∂v        (raw inner product: H_val v + H_P u)
 * — two Hessian-vector products, the second-order term Algorithm 1 differentiates.
 */
export function alignmentParamGradient(val: BatchStats, trig: BatchStats, normalized = true): { G: Grad; u: Grad; v: Grad } {
  const u = gradFromStats(val), v = gradFromStats(trig);
  const G = gaxpy(
    hessVec(val, alignGradWrtSecond(v, u, normalized)),
    hessVec(trig, alignGradWrtSecond(u, v, normalized)), 1,
  );
  return { G, u, v };
}

/**
 * The per-round target of Eq. (9):  d_t = ĝ(S_t; θ_t) + λ Ĝ_t — the honest direction plus the
 * direction that lowers the alignment. Both unit length, per Algorithm 1's normalisation.
 */
export function accumulativeTarget(clean: BatchStats, G: Grad, lambda: number): { target: Grad; honest: Grad; accum: Grad } {
  const honest = gnormalize(gradFromStats(clean));
  const accum = gnormalize(G);
  return { target: gaxpy(honest, accum, lambda), honest, accum };
}

/**
 * One L∞ PGD step with the pixel box:  x ← clip_[lo,hi]( Π_{‖x − x⁰‖∞ ≤ ε}( x + dir·α·sign(∇) ) ).
 * `dir` = +1 ascends the objective, −1 descends it.
 */
export function pgdStep(
  clean: number[][], current: number[][], grads: number[][],
  epsilon: number, alpha: number, dir: 1 | -1, box: [number, number],
): number[][] {
  const [lo, hi] = box;
  return current.map((row, i) => row.map((v, j) => {
    const stepped = v + dir * alpha * Math.sign(grads[i][j]);
    const inBall = Math.max(clean[i][j] - epsilon, Math.min(clean[i][j] + epsilon, stepped));
    return Math.max(lo, Math.min(hi, inBall));
  }));
}

/**
 * A_t(S_t): C steps of PGD ascending  H_t = ĝ(A_t(S_t); θ_t)ᵀ d_t  (Algorithm 1, "update A_t").
 * Only the left gradient depends on the pixels; d_t is the stop-gradient target. Starts from the
 * clean batch (Algorithm 1 initialises A_t(S_t) = S_t).
 */
export function craftAccumulativeBatch(
  m: Net, cleanX: number[][], y: number[], target: Grad,
  epsilon: number, steps: number, box: [number, number],
): { X: number[][]; gradients: number[][]; objective: number } {
  const alpha = (2 * epsilon) / steps;       // α = 2ε/C (§4; the reference's step_size)
  let X = cleanX.map(r => [...r]);
  let gradients: number[][] = [];
  for (let s = 0; s < steps; s++) {
    const st = batchStats(m, X, y);
    const g = gradFromStats(st);
    const ng = gnorm(g);
    if (ng < 1e-12) break;
    // ∂(ĝᵀd)/∂g = (d − (ĝᵀd) ĝ)/‖g‖
    const gh = gscale(g, 1 / ng);
    const q = gscale(gaxpy(target, gh, -gdot(gh, target)), 1 / ng);
    gradients = pixelGradient(st, q);
    X = pgdStep(cleanX, X, gradients, epsilon, alpha, 1, box);
  }
  const objective = gdot(gnormalize(gradFromStats(batchStats(m, X, y))), target);
  return { X, gradients, objective };
}

/**
 * P(S_T): C steps of PGD descending the alignment between u = ∇L(S_val; θ) and ∇L(P(S_T); θ) — the
 * reference code's craft_tri, i.e. Eq. (6) solved for the current model. Warm-starts from
 * `startX` but always projects onto the ε-ball around the CLEAN trigger S_T.
 */
export function craftTrigger(
  m: Net, cleanX: number[][], startX: number[][], y: number[], u: Grad,
  epsilon: number, steps: number, box: [number, number], normalized = true,
): number[][] {
  const alpha = (2 * epsilon) / steps;
  let X = startX.map(r => [...r]);
  for (let s = 0; s < steps; s++) {
    const st = batchStats(m, X, y);
    const q = alignGradWrtSecond(u, gradFromStats(st), normalized);
    X = pgdStep(cleanX, X, pixelGradient(st, q), epsilon, alpha, -1, box);
  }
  return X;
}

/**
 * Coordinates of parameter-space vectors in the plane spanned by the first two, in an orthonormal
 * basis (e₁ ∥ a). Angles and relative lengths are exact, so the explainer can draw gradients with
 * thousands of coordinates as honest 2-D arrows.
 */
export function planeCoords(a: Grad, b: Grad, ...rest: Grad[]): number[][] {
  const e1 = gnormalize(a);
  const perp = gaxpy(b, e1, -gdot(b, e1));
  const e2 = gnormalize(perp);
  return [a, b, ...rest].map(v => [gdot(v, e1), gdot(v, e2)]);
}
