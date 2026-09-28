/**
 * Jagielski et al. 2018 — "Manipulating Machine Learning: Poisoning Attacks and Countermeasures
 * for Regression Learning" (IEEE S&P 2018, arXiv:1804.00308). Optimization-based attack (§III-A).
 *
 *   max_{D_p} W(D′, θ*_p)   s.t.   θ*_p ∈ argmin_θ L(D_tr ∪ D_p, θ)                 (Eq. 2–3)
 *
 * Algorithm 1: poisoning points are updated ONE AT A TIME, each by a line search along
 * ∇_{z_c}W with z_c = (x_c, y_c) (Eq. 13–14, the paper's joint response-variable optimisation),
 * projected onto the feasible domain [0,1]^{d+1}, and the model is retrained after every point.
 */

import { dot, norm, vsub } from '../../linalg';
import { RegressionFit, regularizer, mse, getRidgeModelState, type RegType } from './model';
import { compute } from '../../compute';
import type { TraceFrame } from '../registry';

type Split = { X: number[][]; y: number[] };

export interface RegressionAttackParams {
  train: Split; valid: Split; test: Split;
  type: RegType;
  lambda: number;                     // per-sample λ of Eq. (1)
  poisonRate: number;                 // p / (n + p), §II-A
  init: 'invflip' | 'bflip';
  optimizeY: boolean;                 // (x, y) vs x only
  objective: 'wtr' | 'wval';
  eta: number;                        // line-search step (Table VI grid)
  beta: number;                       // decay, 0.75 (Table VI)
  eps: number;                        // stop when |w(i) − w(i−1)| < ε, 1e-5 (Table VI)
  maxIter: number;
  bounds: [number, number];           // feasible domain for every feature and the response
  onehotGroups?: number[][];
  cleanMSE: number;
  cleanTheta: number[];
}

/** Number of poisoning points for a poisoning rate α/(1+α) = p/(n+p). */
export const poisonCount = (n: number, rate: number) => Math.max(1, Math.round((rate * n) / (1 - rate)));

/**
 * §III-A initialisation: clone random training points, then set the response by
 *   InvFlip  y_c = 1 − y          BFlip  y_c = round(1 − y)
 * (responses live in [0,1]). The paper found smarter choices of x "do not have significant
 * improvement over a simple uniform random choice".
 */
export function initPoison(train: Split, p: number, init: 'invflip' | 'bflip') {
  const idx = train.X.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const X: number[][] = [], Y: number[] = [];
  for (let c = 0; c < p; c++) {
    const i = idx[c % idx.length];
    X.push([...train.X[i]]);
    Y.push(init === 'invflip' ? 1 - train.y[i] : Math.round(1 - train.y[i]));
  }
  return { X, Y };
}

/**
 * Outer objective W (Eq. 8 or 9) and ∇_θW (Eq. 11 or 12) on a fixed data set (D_tr for W_tr,
 * D_val for W_val). The residuals r = Xw + b − y are one product with a matrix kept by the compute
 * backend (on the GPU on the attack server) and ∇_w = (2/m)·Xᵀr another; the line search only
 * needs the value, so `value` skips the gradient. Call `dispose()` when done.
 */
function outerObjective(p: RegressionAttackParams) {
  const data = p.objective === 'wtr' ? p.train : p.valid;
  const m = data.X.length, d = data.X[0].length;
  const flat = new Float64Array(m * d);
  for (let i = 0; i < m; i++) flat.set(data.X[i], i * d);
  const X = compute().denseMatrix(flat, m, d);
  const w = new Float64Array(d);

  const residuals = (theta: number[]) => {
    for (let j = 0; j < d; j++) w[j] = theta[j];
    const r = X.mul(w), b = theta[d];
    for (let i = 0; i < m; i++) r[i] += b - data.y[i];
    return r;
  };
  const value = (theta: number[], r = residuals(theta)) => {
    let v = 0;
    for (let i = 0; i < m; i++) v += r[i] * r[i] / m;
    // W_tr includes the regulariser λΩ(w) (Eq. 8)
    if (p.objective === 'wtr') v += regularizer(p.type, p.lambda, theta.slice(0, -1)).value;
    return v;
  };

  return {
    value: (theta: number[]) => value(theta),
    valueAndGrad(theta: number[]) {
      const r = residuals(theta);
      const gw = X.mulT(r);
      const grad = new Array(d + 1);
      let gb = 0;
      for (let i = 0; i < m; i++) gb += r[i];
      for (let j = 0; j < d; j++) grad[j] = (2 / m) * gw[j];
      grad[d] = (2 / m) * gb;
      if (p.objective === 'wtr') {
        // …and its gradient adds λ ∂Ω/∂w (Eq. 11)
        const reg = regularizer(p.type, p.lambda, theta.slice(0, -1));
        for (let j = 0; j < d; j++) grad[j] += reg.grad[j];
      }
      return { value: value(theta, r), grad };
    },
    dispose: () => X.dispose(),
  };
}

/**
 * ∇_{z_c}W for poisoning point c (Eq. 4 with Eq. 14).
 *
 * Differentiating the (halved, N-scaled) KKT condition Xᵀ(Xθ − y) + reg(θ) = 0:
 *   ∂θ/∂x_cj = −H⁻¹(e_j r_c + x̃_c w_j),     ∂θ/∂y_c = H⁻¹ x̃_c,     r_c = f(x_c) − y_c
 * which is Eq. (14)'s −(2/n)[M w; −x_cᵀ −1][Σ+λg μ; μᵀ 1]⁻¹ up to a positive scale
 * (M = x_c wᵀ + r_c I). With v = H⁻¹∇_θW (H symmetric):
 *   ∇_{x_c}W = −(r_c v_w + (x̃_c·v) w),       ∂W/∂y_c = x̃_c·v
 * so the whole gradient costs one H⁻¹-vector product instead of d linear solves.
 */
export function poisonGradient(fit: RegressionFit, row: number, gradTheta: number[]) {
  const D = fit.dAug, th = fit.theta, xc = fit.rows[row];
  const v = fit.applyHessianInverse(gradTheta);
  const rc = dot(xc, th) - fit.y[row];
  const xv = dot(xc, v);
  const gradX = new Array(D - 1);
  for (let j = 0; j < D - 1; j++) gradX[j] = -(rc * v[j] + xv * th[j]);
  return { gradX, gradY: xv };
}

/**
 * Categorical feasibility, from the authors' implementation (gd_poisoners.py, `colmap`): after the
 * line search each one-hot group is snapped back to a valid one-hot vector — the largest entry
 * becomes 1 if it exceeds 1/(1+k) (k = group size), every other entry 0.
 */
function snapOneHot(x: number[], groups: number[][]) {
  for (const g of groups) {
    let top = g[0];
    for (const j of g) if (x[j] > x[top]) top = j;
    const on = x[top] > 1 / (1 + g.length);
    for (const j of g) x[j] = 0;
    if (on) x[top] = 1;
  }
  return x;
}

/**
 * Algorithm 1. Yields a TraceFrame after initialisation and after every outer iteration, and
 * `null` between single-point updates so a caller can yield to the UI.
 */
export function* regressionAttack(p: RegressionAttackParams): Generator<TraceFrame | null> {
  const { train, test } = p;
  const [lo, hi] = p.bounds;
  const proj = (v: number) => Math.min(hi, Math.max(lo, v));
  const n = train.X.length;
  const P = poisonCount(n, p.poisonRate);

  const { X: poisonX, Y: poisonY } = initPoison(train, P, p.init);
  const fit = new RegressionFit([...train.X, ...poisonX], [...train.y, ...poisonY], p.type, p.lambda);
  const objective = outerObjective(p);
  // The fit's inverse and the objective's matrix may live in GPU memory: release them however
  // the generator ends (completion, an error, or the caller cancelling via return()).
  try {
    const W = () => objective.value(fit.theta);

    let eta = p.eta;
    let gradients: number[][] = poisonX.map(x => new Array(x.length).fill(0));
    let gradientNorms: number[] = poisonX.map(() => 0);

    const frame = (iteration: number, value: number): TraceFrame => {
      const model = { theta: [...fit.theta], type: p.type, lambda: fit.lambda };
      const state = getRidgeModelState(model, train.X, train.y, test.X, test.y);
      return {
        iteration,
        poisonX: poisonX.map(x => [...x]),
        poisonY: [...poisonY],
        poisonedModel: state,
        poisonedRawModel: model,
        poisonedMSE: state.testMSE,
        objectiveValue: value,
        gradients,
        gradientNorms,
        cleanMSE: p.cleanMSE,
        deltaW: norm(vsub(state.theta, p.cleanTheta)),
      };
    };

    let wPrev = W();                                            // Alg. 1 line 4
    yield frame(0, wPrev);

    for (let it = 1; it <= p.maxIter; it++) {
      gradients = [];
      gradientNorms = [];

      for (let c = 0; c < P; c++) {                           // line 6: one point at a time
        const row = n + c;
        const { gradX, gradY } = poisonGradient(fit, row, objective.valueAndGrad(fit.theta).grad);
        gradients.push(gradX);

        // Normalise over the optimised variables (x, or x and y together)
        const gn = Math.sqrt(dot(gradX, gradX) + (p.optimizeY ? gradY * gradY : 0));
        gradientNorms.push(gn);
        if (gn < 1e-12) { yield null; continue; }

        // line 7: line search along ∇_{z_c}W, projected onto [0,1]. As in the authors' code, step
        // η, then ηβ, ηβ², … while W keeps increasing; the step that stops increasing is undone.
        let w0 = W(), step = eta;
        let x = poisonX[c], yv = poisonY[c];
        for (let k = 0; k < 50; k++) {
          const nx = x.map((v, j) => proj(v + step * gradX[j] / gn));
          const ny = p.optimizeY ? proj(yv + step * gradY / gn) : yv;
          fit.setRow(row, nx, ny);                              // line 8: retrain
          const w1 = W();
          if (!(w1 > w0 + 1e-12)) { fit.setRow(row, x, yv); break; }
          x = nx; yv = ny; w0 = w1; step *= p.beta;
        }
        if (p.onehotGroups?.length) {
          x = snapOneHot([...x], p.onehotGroups);
          fit.setRow(row, x, yv);
        }
        poisonX[c] = x;
        poisonY[c] = yv;
        yield null;
      }

      fit.refit();                                              // clear rank-one update drift
      const w = W();                                            // line 9
      yield frame(it, w);

      if (Math.abs(w - wPrev) < p.eps) break;                   // line 11
      if (w <= wPrev) eta *= p.beta;                            // App. C: decay η on no progress
      wPrev = w;
    }
  } finally {
    fit.dispose();
    objective.dispose();
  }
}
