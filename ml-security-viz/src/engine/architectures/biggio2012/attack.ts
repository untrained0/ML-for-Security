/**
 * Biggio, Nelson & Laskov 2012 — "Poisoning Attacks against Support Vector Machines"
 * (ICML 2012, arXiv:1206.6389).
 *
 * The attacker adds a point (x_c, y_c) with a FIXED label y_c (the "attacking class") and moves
 * x_c by gradient ascent on the validation hinge loss of the SVM retrained on D_tr ∪ {(x_c, y_c)}:
 *
 *   max_{x_c} L(x_c) = Σ_k (1 − y_k f_{x_c}(x_k))₊ = Σ_k (−g_k)₊                        (Eq. 1)
 *
 * The gradient is the closed form of Eq. (10), obtained by keeping the KKT conditions of the SVM
 * (Eq. 4–5) at equilibrium while x_c moves, i.e. assuming the margin/error/reserve sets S, E, R
 * do not change during an infinitesimal step. Multi-point attacks are "sequential single-point
 * attacks" (§4, Fig. 3): each point is optimised to convergence, frozen, and the next is added.
 */

import { trainSVM, predict, accuracy, hingeLoss, getModelState, compactModel } from './model';
import { invert, norm, scale, vadd, vsub } from '../../linalg';
import type { TraceFrame } from '../registry';

type KernelFn = (a: number[], b: number[]) => number;
type KernelGrad = (xi: number[], xc: number[]) => number[];

/**
 * ∂L/∂x_c — Eq. (3), (7), (10).
 *
 * `model` is the SVM trained on D_tr ∪ {(x_c, y_c)} with the attack point at index `c`.
 *
 *   [∂b/∂u; ∂α_s/∂u] = −[0 y_sᵀ; y_s Q_ss]⁻¹ [0; ∂Q_sc/∂u] α_c                       (Eq. 7)
 *   ∂g_k/∂u = Q_ks ∂α_s/∂u + ∂Q_kc/∂u α_c + y_k ∂b/∂u                                 (Eq. 3)
 *
 * Since L = Σ(−g_k)₊, only validation points with g_k < 0 contribute and
 *   ∂L/∂u = −Σ_{k: g_k<0} ∂g_k/∂u.
 * Eq. (10) as printed is Σ_k ∂g_k/∂u without that leading minus; ascending it would DEcrease the
 * hinge loss.
 *
 * Two departures from the printed equations, both needed for the gradient to be correct:
 *
 * 1. Eq. (8)–(9) invert the bordered matrix via Sherman–Morrison, which needs Q_ss⁻¹. That fails
 *    whenever Q_ss is singular — e.g. a linear kernel with more than d margin SVs (3 in 2-D) —
 *    while the bordered matrix itself usually stays invertible. We solve (7) directly, which is
 *    identical when Q_ss is invertible.
 *
 * 2. Eq. (7) holds α_c fixed, i.e. assumes x_c ∈ E (α_c = C). As the attack point is pulled
 *    toward the attacking class it often becomes a margin SV (0 < α_c < C); its own KKT condition
 *    g_c = 0 must then be kept at equilibrium too, with α_c free — the same adiabatic argument
 *    (Cauwenberghs & Poggio 2001) applied to one more point. Keeping α_c fixed there gives an
 *    ascent direction with the WRONG sign (checked against finite differences on MNIST).
 *    With c ∈ S the right-hand side row for c is ∂g_c/∂u|explicit = Σ_j α_j ∂Q_cj/∂u, which
 *    includes ∂Q_cc/∂u = 2 ∂K(x, x_c)/∂x_c at x = x_c.
 *
 * If x_c is a reserve point (α_c = 0) the gradient vanishes and the attack stalls — the paper's
 * warning in §2.3. With no margin SVs at all nothing can absorb the move (∂α = 0) and b is not
 * pinned by any margin condition; we take ∂b = 0, leaving the direct α_c ∂Q_kc/∂u term.
 */
export function biggioGradient(
  model: any, c: number,
  Xval: number[][], yval: number[],
  kernelFn: KernelFn, dK: KernelGrad,
): number[] {
  const { alpha, y, X, K, C } = model;
  const d = X[c].length;
  const zero = () => new Array(d).fill(0);
  const tol = 1e-6 * C;
  const ac = alpha[c], yc = y[c];
  if (ac <= tol) return zero();

  // S: margin support vectors, 0 < α < C — including x_c itself when it is one
  const S: number[] = [];
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] > tol && alpha[i] < C - tol && (i !== c || ac < C - tol)) S.push(i);
  }
  const ns = S.length;

  // Explicit ∂g_i/∂u for i ∈ S (the right-hand side of Eq. 7, before the minus sign)
  const rhs = S.map(i => {
    if (i !== c) return scale(dK(X[i], X[c]), ac * y[i] * yc);         // α_c ∂Q_ic/∂u
    const r = scale(dK(X[c], X[c]), 2 * ac);                           // α_c ∂Q_cc/∂u
    for (let j = 0; j < alpha.length; j++) {
      if (j === c || alpha[j] <= tol) continue;
      const g = dK(X[j], X[c]);
      for (let l = 0; l < d; l++) r[l] += alpha[j] * yc * y[j] * g[l];  // α_j ∂Q_cj/∂u
    }
    return r;
  });

  // Eq. (7), solved on the bordered matrix
  const A: number[][] = Array.from({ length: ns + 1 }, () => new Array(ns + 1).fill(0));
  for (let r = 0; r < ns; r++) {
    A[0][r + 1] = A[r + 1][0] = y[S[r]];
    for (let q = 0; q < ns; q++) A[r + 1][q + 1] = y[S[r]] * y[S[q]] * K[S[r]][S[q]];
  }
  const Ainv = ns > 0 ? invert(A) : [[0]];
  const solveRow = (row: number) => {
    const out = new Array(d).fill(0);
    for (let q = 0; q < ns; q++) {
      const a = -Ainv[row][q + 1];
      if (a === 0) continue;
      for (let l = 0; l < d; l++) out[l] += a * rhs[q][l];
    }
    return out;
  };
  const dB = solveRow(0);                                     // ∂b/∂u
  const dAlpha = S.map((_, r) => solveRow(r + 1));            // ∂α_s/∂u (and ∂α_c/∂u if c ∈ S)

  // Eq. (10), summed over validation points inside the hinge. ∂b/∂u and ∂α_s/∂u do not depend on
  // k, so their coefficients are summed over k first:
  //   Σ_k ∂g_k/∂u = α_c y_c Σ_k y_k ∂K_kc/∂u + (Σ_k y_k) ∂b/∂u + Σ_r (Σ_k y_k y_r K_kr) ∂α_r/∂u
  // (m·n_s·d → n_s·d for the last term; 784-d MNIST spent ~75% of its run time there).
  const grad = new Array(d).fill(0);
  const coefAlpha = new Array(ns).fill(0);
  let coefB = 0;
  for (let k = 0; k < Xval.length; k++) {
    const yk = yval[k];
    const gk = yk * predict(model, Xval[k]) - 1;              // Eq. (2)
    if (gk >= 0) continue;

    const dQkc = dK(Xval[k], X[c]);
    const a = ac * yk * yc;
    for (let l = 0; l < d; l++) grad[l] -= a * dQkc[l];
    coefB += yk;
    for (let r = 0; r < ns; r++) coefAlpha[r] += yk * y[S[r]] * kernelFn(Xval[k], X[S[r]]);
  }
  for (let l = 0; l < d; l++) grad[l] -= coefB * dB[l];
  for (let r = 0; r < ns; r++) {
    const q = coefAlpha[r];
    if (q === 0) continue;
    const da = dAlpha[r];
    for (let l = 0; l < d; l++) grad[l] -= q * da[l];
  }
  return grad.every(Number.isFinite) ? grad : zero();
}

/**
 * Initial attack point, §2.3: "cloning an arbitrary point from the attacked class and flipping
 * its label" (the attacked class is −y_c).
 *
 * 'furthest' is NOT from the paper: it clones the attacked-class point lying deepest on its own
 * side of the current boundary. The paper notes that a start too close to the boundary "may
 * become a reserve point, which halts further progress"; this heuristic avoids that.
 */
export function initialAttackPoint(
  X: number[][], y: number[], yc: number, strategy: string, model: any,
): number[] {
  const pool = X.map((_, i) => i).filter(i => y[i] === -yc);
  if (pool.length === 0) throw new Error('No points in the attacked class to clone');
  if (strategy === 'furthest') {
    let best = pool[0], bestM = -Infinity;
    for (const i of pool) {
      const m = y[i] * predict(model, X[i]);
      if (m > bestM) { bestM = m; best = i; }
    }
    return [...X[best]];
  }
  return [...X[pool[Math.floor(Math.random() * pool.length)]]];
}

export interface BiggioAttackParams {
  train: { X: number[][]; y: number[] };
  valid: { X: number[][]; y: number[] };
  test: { X: number[][]; y: number[] };
  kernelFn: KernelFn;
  dK: KernelGrad;
  C: number;
  numPoison: number;
  yc: number;                    // attacking class label (fixed)
  stepSize: number;              // t
  maxIter: number;               // per attack point
  /**
   * 'holdout' — early stopping on a held-out part of the attacker's validation data (default)
   * 'paper'   — Alg. 1 line 8: stop when L(x^(p)) − L(x^(p−1)) < ε
   * 'none'    — always run maxIter steps
   */
  stopRule: 'holdout' | 'paper' | 'none';
  eps: number;
  holdoutFraction?: number;      // share of D_val kept out of the gradient (default 0.2)
  patience?: number;             // steps without held-out improvement before stopping (default 10)
  bounds: [number, number];      // feasible box for every feature of x_c
  initStrategy: string;
  cleanAccuracy: number;
  maxFrames?: number;
}

/**
 * Split D_val into the part the gradient ascends (D_opt) and a held-out part (D_mon) that only
 * judges whether the attack still generalises. Shuffled, since generated sets are class-ordered.
 */
function splitValidation(valid: BiggioAttackParams['valid'], fraction: number) {
  const idx = valid.X.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const cut = Math.round(idx.length * (1 - fraction));
  const pick = (ids: number[]) => ({ X: ids.map(i => valid.X[i]), y: ids.map(i => valid.y[i]) });
  return { opt: pick(idx.slice(0, cut)), mon: pick(idx.slice(cut)) };
}

/**
 * Secant ascent direction for low-dimensional data when Eq. (10) is exactly zero although x_c
 * is a support vector. That happens when d + 1 margin SVs pin (w, b) — e.g. a linear kernel in
 * 2-D, the setting of Fig. 1 — so no infinitesimal move changes f, and progress needs a move
 * large enough to change the SV sets. Differencing at the step size t measures exactly that.
 */
function secantDirection(
  base: { X: number[][]; y: number[] }, xc: number[], yc: number, h: number,
  opt: BiggioAttackParams['valid'], kernelFn: KernelFn, C: number,
  project: (x: number[]) => number[],
) {
  const L = (x: number[]) =>
    hingeLoss(trainSVM([...base.X, project(x)], [...base.y, yc], kernelFn, C), opt.X, opt.y);
  return xc.map((_, j) => {
    const a = [...xc], b = [...xc];
    a[j] += h; b[j] -= h;
    return (L(a) - L(b)) / (2 * h);
  });
}

/**
 * Algorithm 1, run once per attack point (sequential multi-point attack, §4).
 *
 * With a fixed step and a finite D_val, the ascent keeps finding validation-specific gains long
 * after the attack stops generalising, and near a maximum it oscillates around it: in both cases
 * later iterations change the loss on D_val but not the damage to unseen data. With the default
 * 'holdout' rule the attacker therefore ascends Eq. (10) on 80% of its validation data, tracks
 * the hinge loss on the other 20%, stops after `patience` steps without held-out improvement,
 * and keeps the iterate that was best on the held-out part. All of this uses only the
 * attacker's own data — never D_ts. (Halving t on every loss dip was tried and rejected: dips
 * from SV-set changes are frequent, so t collapsed and the attack stopped early and weaker.)
 *
 * Yields TraceFrames; long runs are thinned to ~maxFrames, but the last step of every point is
 * always emitted.
 */
export function* biggioAttack(p: BiggioAttackParams): Generator<TraceFrame, void, unknown> {
  const { train, test, kernelFn, dK, C, yc, bounds: [lo, hi] } = p;
  const project = (x: number[]) => x.map(v => Math.min(hi, Math.max(lo, v)));
  const frameEvery = Math.max(1, Math.ceil((p.numPoison * p.maxIter) / (p.maxFrames ?? 150)));
  const holdout = p.stopRule === 'holdout';
  const patience = p.patience ?? 10;
  const { opt, mon } = holdout
    ? splitValidation(p.valid, p.holdoutFraction ?? 0.2)
    : { opt: p.valid, mon: null };
  const d = train.X[0].length;

  const fixedX: number[][] = [];
  const fixedY: number[] = [];
  let frameNo = 0;
  let lastW: number[] | null = null;
  let lastSVs = new Map<number, number[]>();

  const frame = (c: number, xc: number[], model: any, L: number, heldOut: number | undefined,
    grad: number[], gn: number, step: number): TraceFrame => {
    const state: any = getModelState(model);
    const svs = new Map<number, number[]>(state.supportVectors.map((sv: any) => [sv.index, sv.point]));
    const gainedSVs = [...svs].filter(([i]) => !lastSVs.has(i)).map(([, pt]) => pt);
    const lostSVs = [...lastSVs].filter(([i]) => !svs.has(i)).map(([, pt]) => pt);
    lastSVs = svs;
    const deltaW = lastW ? norm(vsub(state.w, lastW)) : 0;
    lastW = state.w;
    return {
      iteration: frameNo++,
      activePoison: c,
      poisonX: [...fixedX.map(x => [...x]), [...xc]],
      poisonY: [...fixedY, yc],
      poisonedModel: state,
      poisonedRawModel: compactModel(model),
      poisonedAccuracy: accuracy(model, test.X, test.y),
      objectiveValue: L,
      heldOutObjective: heldOut,
      attackStep: step,
      gradients: [...fixedX.map(x => new Array(x.length).fill(0)), grad],
      gradientNorms: [...fixedX.map(() => 0), gn],
      cleanAccuracy: p.cleanAccuracy,
      deltaW,
      gainedSVs,
      lostSVs,
    };
  };

  for (let c = 0; c < p.numPoison; c++) {
    // Alg. 1 line 1: SVM on the data the attacker is poisoning (clean + already-fixed points)
    const base = { X: [...train.X, ...fixedX], y: [...train.y, ...fixedY] };
    const baseModel = trainSVM(base.X, base.y, kernelFn, C);
    let xc = project(initialAttackPoint(train.X, train.y, yc, p.initStrategy, baseModel));
    let prevL = -Infinity;
    const step = p.stepSize;
    let best = { x: xc, heldOut: -Infinity, it: 0 };
    let sinceBest = 0;

    for (let it = 0; it <= p.maxIter; it++) {
      // line 4: SVM on D_tr ∪ {x_c^(p), y_c}. (The paper warm-starts an incremental SVM; a
      // from-scratch solve reaches the same optimum.)
      const augX = [...base.X, xc];
      const augY = [...base.y, yc];
      const model = trainSVM(augX, augY, kernelFn, C);
      const L = hingeLoss(model, opt.X, opt.y);                   // mean of Eq. (1)
      const heldOut = mon ? hingeLoss(model, mon.X, mon.y) : undefined;

      // line 5: ∂L/∂u on D_val (or D_opt), Eq. (10)
      let grad = biggioGradient(model, augX.length - 1, opt.X, opt.y, kernelFn, dK);
      let gn = norm(grad);
      if (gn < 1e-12 && model.alpha[augX.length - 1] > 1e-6 * C && d <= 10) {
        grad = secantDirection(base, xc, yc, step, opt, kernelFn, C, project);
        gn = norm(grad);
      }

      let stop = it === p.maxIter || gn < 1e-12;
      if (p.stopRule === 'paper') {
        stop ||= it > 0 && L - prevL < p.eps;                     // line 8
      } else if (holdout) {
        if (heldOut! > best.heldOut + 1e-9) { best = { x: xc, heldOut: heldOut!, it }; sinceBest = 0; }
        else sinceBest++;
        stop ||= sinceBest >= patience;
      }

      if (it % frameEvery === 0 || stop) yield frame(c, xc, model, L, heldOut, grad, gn, step);
      if (stop) break;

      // lines 6–7: u = ∇L/‖∇L‖, x_c ← x_c + t u, projected onto the feasible box ("we bound the
      // size of our attack points", §2.3; [−4,4]² in Fig. 1, pixel range [0,1] for MNIST)
      prevL = L;
      xc = project(vadd(xc, scale(grad, step / gn)));
    }

    // Keep the iterate that generalised best, and show the model it produces
    if (holdout && best.x !== xc) {
      xc = best.x;
      const model = trainSVM([...base.X, xc], [...base.y, yc], kernelFn, C);
      const zero = new Array(xc.length).fill(0);
      yield frame(c, xc, model, hingeLoss(model, opt.X, opt.y), best.heldOut, zero, 0, step);
    }

    fixedX.push(xc);
    fixedY.push(yc);
  }
}

/** Default feasible box when a dataset does not declare one: the data range padded by 25%. */
export function defaultBounds(dataset: any): [number, number] {
  if (dataset.bounds) return dataset.bounds;
  let lo = Infinity, hi = -Infinity;
  for (const x of dataset.train.X) for (const v of x) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const pad = 0.25 * (hi - lo);
  return [lo - pad, hi + pad];
}

/**
 * Objective surface of Fig. 1: validation hinge loss of the SVM trained on D_tr ∪ {(x, y_c)}
 * for every x on a grid, plus the Eq. (10) gradient there. Only meaningful for 2-D data.
 */
export function objectiveSurface(
  train: any, valid: any, kernelFn: KernelFn, dK: KernelGrad, C: number, yc: number,
  range: { xMin: number; xMax: number; yMin: number; yMax: number }, resolution: number,
) {
  const data: { x: number; y: number; value: number; gradX: number; gradY: number }[] = [];
  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const x = range.xMin + j * (range.xMax - range.xMin) / (resolution - 1);
      const yv = range.yMax - i * (range.yMax - range.yMin) / (resolution - 1);
      const X = [...train.X, [x, yv]], Y = [...train.y, yc];
      const model = trainSVM(X, Y, kernelFn, C);
      const g = biggioGradient(model, X.length - 1, valid.X, valid.y, kernelFn, dK);
      data.push({ x, y: yv, value: hingeLoss(model, valid.X, valid.y), gradX: g[0], gradY: g[1] });
    }
  }
  return data;
}
