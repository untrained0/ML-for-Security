/**
 * Jagielski et al. 2018 — the linear regression learners of §II (Eq. 1):
 *
 *   L(D, θ) = (1/N) Σᵢ (wᵀxᵢ + b − yᵢ)² + λ Ω(w)
 *
 *   OLS          Ω = 0
 *   ridge        Ω = ½‖w‖²
 *   LASSO        Ω = ‖w‖₁
 *   elastic net  Ω = ρ‖w‖₁ + (1−ρ)½‖w‖²,  ρ = 0.5 ("as we do in this work")
 *
 * λ is PER SAMPLE, as in the paper, so its meaning does not drift when poisoning points are added.
 * Internally we minimise the N-scaled objective Σ r² + NλΩ(w), whose stationarity condition
 * (halved) is
 *
 *   Xᵀ(Xθ − y) + (Nλ₁/2) sign(w) + (Nλ₂/2) w = 0,      λ₁ = λ·[ℓ1 weight], λ₂ = λ·[ℓ2 weight]
 *
 * and H = XᵀX + (Nλ₂/2)·I_w (on the active set when λ₁ > 0) is the matrix the implicit
 * derivative of Eq. (7)/(14) inverts. The bias is never regularised.
 */

import { dot } from '../../linalg';
import { compute, type DenseMatrix, type RankOneInverse } from '../../compute';

export type RegType = 'ols' | 'ridge' | 'lasso' | 'elasticnet';

const RHO = 0.5;
/** The authors fit "OLS" as sklearn Ridge(alpha=1e-5) (gd_poisoners.py): one-hot columns are
 *  collinear with the bias, so XᵀX is singular without a tiny ridge. */
const OLS_RIDGE = 1e-5;
/** OLS iterative refinement (RegressionFit.refined), in relative energy norm ‖δ‖_A / ‖x‖_A:
 *  converged below REFINE_TOL; a stall is accepted as the rounding floor below REFINE_FLOOR_THETA
 *  (θ = A⁻¹Xᵀy) or REFINE_FLOOR_GRAD (H⁻¹∇W, where 1e-6 relative is ample for a search
 *  direction), else the inverse is refactored. */
const REFINE_TOL = 1e-12;
const REFINE_FLOOR_THETA = 2e-9;
const REFINE_FLOOR_GRAD = 1e-6;
const REFINE_STEPS = 8;

/** λΩ(w) = l1‖w‖₁ + l2·½‖w‖² */
export function penalty(type: RegType, lambda: number) {
  switch (type) {
    case 'ridge': return { l1: 0, l2: lambda };
    case 'lasso': return { l1: lambda, l2: 0 };
    case 'elasticnet': return { l1: lambda * RHO, l2: lambda * (1 - RHO) };
    default: return { l1: 0, l2: 0 };
  }
}

/** λΩ(w) and its (sub)gradient ∂(λΩ)/∂w, used by the W_tr objective (Eq. 8, 11). */
export function regularizer(type: RegType, lambda: number, w: number[]) {
  const { l1, l2 } = penalty(type, lambda);
  let value = 0;
  const grad = w.map(v => {
    value += l1 * Math.abs(v) + 0.5 * l2 * v * v;
    return l1 * Math.sign(v) + l2 * v;
  });
  return { value, grad };
}

const soft = (z: number, t: number) => (z > t ? z - t : z < -t ? z + t : 0);

/**
 * A regression fit on N rows whose last rows (the poisoning points) can be replaced one at a
 * time, as Algorithm 1 does. Xᵀy is kept current under row swaps, and so is the inverse the
 * implicit derivative needs, via Sherman–Morrison rank-one updates (`compute().rankOneInverse`,
 * resident on the GPU when the attack server has one):
 *   ridge / OLS   (XᵀX + l2s·I_w)⁻¹, θ = that · Xᵀy. XᵀX is only read to rebuild that inverse,
 *                 which happens right after a refit, so a row swap merely marks it stale.
 *   ℓ1 models     (X_AᵀX_A + l2s·I_w)⁻¹ on the active set A, θ_A from the KKT system when the sign
 *                 pattern holds, warm-started coordinate descent when it does not. The KKT checks
 *                 read XᵀX, so it is updated on every swap (O(d²)).
 *   OLS           as ridge, but l2s is only OLS_RIDGE and the one-hot columns are collinear with
 *                 the bias, so cond(XᵀX + l2s·I) ≈ 1e10: an explicitly updated inverse drifted by
 *                 up to 1e-2 in test predictions between refits (a fresh inverse is only good to
 *                 ~1e-7). So the inverse is used as a preconditioner only: every solve, θ and
 *                 H⁻¹g alike, is finished by iterative refinement against the normal-equation
 *                 residual computed from the rows (`refined`), and the inverse is refactored
 *                 whenever refinement stops converging.
 * Call `dispose()` when done: a GPU-resident inverse holds device memory.
 */
export class RegressionFit {
  type: RegType;
  lambda: number;
  N: number;
  dAug: number;
  rows: number[][];          // augmented rows [x, 1]
  y: number[];
  theta: number[];
  private l1s: number;       // Nλ₁/2  (soft-threshold level)
  private l2s: number;       // Nλ₂/2  (diagonal ridge)
  private XtX!: Float64Array[];            // row views into one D×D buffer
  private xtxStale = false;
  private Xty!: number[];
  private inv: RankOneInverse | null = null;   // inverse for `act` (all columns when closed-form)
  private act: number[] | null = null;
  private updates = 0;
  private readonly refine: boolean;           // OLS: polish every solve (see class comment)
  private Xd: DenseMatrix | null = null;      // OLS: the rows as of the last refit, for residuals
  private moved = new Map<number, number[]>(); // rows swapped since: index → the row Xd holds

  constructor(X: number[][], y: number[], type: RegType, lambda: number) {
    this.type = type;
    this.refine = type === 'ols';
    this.lambda = type === 'ols' ? 0 : lambda;
    this.N = X.length;
    this.rows = X.map(r => [...r, 1]);
    this.y = [...y];
    this.dAug = this.rows[0].length;
    const { l1, l2 } = penalty(type, this.lambda);
    this.l1s = (this.N * l1) / 2;
    this.l2s = type === 'ols' ? OLS_RIDGE : (this.N * l2) / 2;
    this.theta = new Array(this.dAug).fill(0);
    this.refit();
  }

  private get closedForm() { return this.l1s === 0; }

  /** XᵀX from the stored rows; returns them flattened (row-major N × D). */
  private buildXtX() {
    const D = this.dAug, N = this.N;
    const flat = new Float64Array(N * D);
    for (let i = 0; i < N; i++) flat.set(this.rows[i], i * D);
    const XtX = compute().gram(flat, N, D);
    this.XtX = Array.from({ length: D }, (_, j) => XtX.subarray(j * D, (j + 1) * D));
    this.xtxStale = false;
    return flat;
  }

  private ensureXtX() {
    if (this.xtxStale) this.buildXtX();
  }

  private replaceInverse(next: RankOneInverse | null) {
    if (this.inv && this.inv !== next) this.inv.dispose();
    this.inv = next;
  }

  /** Rebuild every statistic from the stored rows (also clears Sherman–Morrison drift). */
  refit() {
    const D = this.dAug;
    const flat = this.buildXtX();
    if (this.refine) {
      this.Xd?.dispose();
      this.Xd = compute().denseMatrix(flat, this.N, D);
      this.moved.clear();
    }
    this.Xty = new Array(D).fill(0);
    for (let i = 0; i < this.N; i++) {
      const r = this.rows[i], yi = this.y[i];
      for (let j = 0; j < D; j++) {
        const rj = r[j];
        if (rj !== 0) this.Xty[j] += rj * yi;
      }
    }
    this.replaceInverse(null);
    this.act = null;
    if (!this.closedForm && this.theta.every(v => v === 0)) this.coordinateDescent();
    this.solve();
    return this.theta;
  }

  /** Replace row i by (x, y) and refit. */
  setRow(i: number, x: number[], yv: number) {
    const D = this.dAug;
    const old = this.rows[i], oldY = this.y[i];
    const neu = [...x, 1];
    if (this.refine && !this.moved.has(i)) this.moved.set(i, old);
    this.rows[i] = neu;
    this.y[i] = yv;
    for (let j = 0; j < D; j++) this.Xty[j] += neu[j] * yv - old[j] * oldY;
    if (this.closedForm) {
      this.xtxStale = true;
    } else {
      for (let j = 0; j < D; j++) {
        const nj = neu[j], oj = old[j];
        if (nj === 0 && oj === 0) continue;
        const row = this.XtX[j];
        for (let k = 0; k < D; k++) row[k] += nj * neu[k] - oj * old[k];
      }
    }
    if (this.inv) {
      const sub = (v: number[]) => (this.act ? this.act.map(j => v[j]) : v);
      if (this.refine) {
        // Add before removing: A + nnᵀ is never closer to singular than A, whereas A − ooᵀ nearly
        // is when `old` is the only row carrying a rare one-hot category, and the tiny
        // Sherman–Morrison denominator 1 − oᵀA⁻¹o would then amplify rounding into the inverse
        this.inv.update(sub(neu), +1);
        this.inv.update(sub(old), -1);
      } else {
        this.inv.update(sub(old), -1);
        this.inv.update(sub(neu), +1);
      }
    }
    // Rank-one updates do not drift measurably under regularisation (~1e-13). They do on OLS's
    // ill-conditioned system, but there the inverse is only a preconditioner: `refined` corrects
    // every solve and refactors the inverse itself once it stops converging.
    if (++this.updates % 1024 === 0) return this.refit();
    this.solve();
    return this.theta;
  }

  /** (X_AᵀX_A + l2s·I_w)⁻¹ from scratch; `act` null means every column. */
  private freshInverse(act: number[] | null): RankOneInverse {
    this.ensureXtX();
    const D = this.dAug, idx = act ?? Array.from({ length: D }, (_, j) => j);
    const n = idx.length, H = new Float64Array(n * n);
    for (let a = 0; a < n; a++) {
      const row = this.XtX[idx[a]];
      for (let b = 0; b < n; b++) H[a * n + b] = row[idx[b]] + (a === b && idx[a] !== D - 1 ? this.l2s : 0);
    }
    const c = compute();
    return c.rankOneInverse(c.inverse(H, n), n);
  }

  private activeSet() {
    const D = this.dAug, act: number[] = [];
    for (let j = 0; j < D; j++) if (j === D - 1 || Math.abs(this.theta[j]) > 1e-10) act.push(j);
    return act;
  }

  /** Point `inv`/`act` at the current active set (all columns for closed form). */
  private syncInverse() {
    if (this.closedForm) {
      if (!this.inv) this.inv = this.freshInverse(null);
      return;
    }
    const act = this.activeSet();
    const same = this.act && this.act.length === act.length && this.act.every((j, a) => j === act[a]);
    if (!same || !this.inv) {
      this.act = act;
      this.replaceInverse(this.freshInverse(act));
    }
  }

  private solve() {
    this.syncInverse();
    if (this.refine) {
      this.theta = this.refined(this.Xty, REFINE_FLOOR_THETA, this.y);
    } else if (this.closedForm) {
      this.theta = Array.from(this.inv!.apply(this.Xty));
      if (!this.theta.every(Number.isFinite)) {
        this.replaceInverse(this.freshInverse(null));
        this.theta = Array.from(this.inv!.apply(this.Xty));
      }
    } else if (!this.activeSetSolve()) {
      this.coordinateDescent();
    }
  }

  /**
   * A⁻¹b, A = XᵀX + l2s·I_w, for OLS: iterative refinement with the maintained inverse M as the
   * preconditioner, x ← x + δ, δ = M(b − Ax).
   *
   * The residual is taken from the rows themselves, never from running totals: with `y`
   * (b = Xᵀy) r = Xᵀ(y − Xx) − l2s·x_w, which also avoids the cancellation in Xᵀy − XᵀXx;
   * otherwise r = b − Xᵀ(Xx) − l2s·x_w. An incrementally updated XᵀX / Xᵀy is off by ~1e-13
   * after a few hundred swaps, and at this conditioning that alone moved test predictions by
   * ~1e-6. The two O(N·d) products run on `Xd` (resident on the GPU when there is one), with the
   * rows swapped since it was built (`moved`; the poisoning points, during an attack) patched in.
   *
   * Progress is measured in the energy norm ‖δ‖_A ≈ √(δ·r) relative to ‖x‖_A. It bounds what
   * matters, ‖Xδ‖ ≤ ‖δ‖_A (the change of every fitted value), and ignores rounding noise along
   * the near-null direction of the one-hot columns and the bias, which dominates ‖δ‖ itself but
   * moves no prediction. Corrections are applied while they at least halve; once they stop (the
   * fp64 floor: ~1e-9 for θ and ~1e-7 for H⁻¹g on Warfarin/House) x is accepted if that is below
   * `floor`. Otherwise M has drifted too far to precondition (or went non-finite): it is
   * refactored and the solve restarted, and the result of that second attempt is kept.
   */
  private refined(b: ArrayLike<number>, floor: number, y?: number[]): number[] {
    const D = this.dAug, N = this.N, Xd = this.Xd!;
    for (let attempt = 0; ; attempt++) {
      const x = Array.from(this.inv!.apply(b));
      let prev = Infinity;
      for (let step = 0; step < REFINE_STEPS; step++) {
        const e = Xd.mul(x);                                  // Xx, stale rows fixed below
        for (const i of this.moved.keys()) e[i] = dot(this.rows[i], x);
        let xAx = 0;
        for (let i = 0; i < N; i++) {
          xAx += e[i] * e[i];
          e[i] = y ? y[i] - e[i] : -e[i];                     // y − Xx, or −Xx
        }
        const r = Xd.mulT(e);
        for (const [i, was] of this.moved) {
          const now = this.rows[i], ei = e[i];
          if (ei !== 0) for (let k = 0; k < D; k++) r[k] += (now[k] - was[k]) * ei;
        }
        for (let j = 0; j < D; j++) {
          if (!y) r[j] += b[j];
          if (j === D - 1) continue;
          r[j] -= this.l2s * x[j];
          xAx += this.l2s * x[j] * x[j];
        }
        const dx = this.inv!.apply(r);
        let dr = 0;
        for (let j = 0; j < D; j++) dr += dx[j] * r[j];
        if (xAx === 0 && dr === 0) return x;                  // b = 0
        const rel = Math.sqrt(Math.abs(dr) / xAx);            // NaN/∞ when M or x is not finite
        if (rel <= REFINE_TOL) return x;
        if (!(rel <= prev / 2)) {
          if (rel <= floor || attempt > 0) return x;
          break;                                              // refactor M and start over
        }
        for (let j = 0; j < D; j++) x[j] += dx[j];
        prev = rel;
      }
      if (attempt > 0) return x;
      this.replaceInverse(this.freshInverse(null));
    }
  }

  /**
   * Exact ℓ1 solve by an active-set iteration, warm-started from the current sign pattern (a
   * single-row replacement rarely changes more than a few signs). On the active set A,
   *   (X_AᵀX_A + l2s·I_w) θ_A = X_Aᵀy − l1s·sign(θ_A)
   * then coefficients that flipped sign leave A and inactive j violating |X_jᵀ(y − Xθ)| ≤ l1s
   * enter it with the sign of that correlation — until both KKT conditions hold. Returns false
   * (caller falls back to coordinate descent) if that does not settle quickly.
   */
  private activeSetSolve(): boolean {
    const D = this.dAug;
    let act = this.act!;
    const sign = new Map<number, number>(act.map(j => [j, j === D - 1 ? 0 : Math.sign(this.theta[j])]));
    let inv = this.inv!;
    const release = (m: RankOneInverse) => { if (m !== this.inv) m.dispose(); };

    for (let round = 0; round < 25; round++) {
      const na = act.length;
      const tA = inv.apply(act.map(j => this.Xty[j] - this.l1s * sign.get(j)!));

      const flipped = act.filter((j, a) => j !== D - 1 && Math.sign(tA[a]) !== sign.get(j));
      const entering: number[] = [];
      if (flipped.length === 0) {
        const isAct = new Array(D).fill(false);
        act.forEach(j => { isAct[j] = true; });
        for (let j = 0; j < D; j++) {
          if (isAct[j]) continue;
          let g = this.Xty[j];                                // X_jᵀ(y − X_A θ_A)
          const row = this.XtX[j];
          for (let a = 0; a < na; a++) g -= row[act[a]] * tA[a];
          if (Math.abs(g) > this.l1s * (1 + 1e-9) + 1e-12) { entering.push(j); sign.set(j, Math.sign(g)); }
        }
        if (entering.length === 0) {
          const next = new Array(D).fill(0);
          act.forEach((j, a) => { next[j] = tA[a]; });
          this.theta = next;
          this.act = act;
          this.replaceInverse(inv);
          return true;
        }
      }
      for (const j of flipped) sign.delete(j);
      act = [...act.filter(j => !flipped.includes(j)), ...entering].sort((a, b) => a - b);
      const nextInv = this.freshInverse(act);
      release(inv);
      inv = nextInv;
    }
    release(inv);
    return false;
  }

  /** Coordinate descent on Σ r² + 2·l1s‖w‖₁ + l2s‖w‖², warm-started from the current θ. */
  private coordinateDescent(maxSweeps = 500, tol = 1e-6) {
    this.ensureXtX();
    const D = this.dAug, N = this.N, th = [...this.theta];
    const res = this.y.map((yv, i) => dot(this.rows[i], th) - yv);     // r = Xθ − y
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let maxDelta = 0;
      for (let j = 0; j < D; j++) {
        const z = this.XtX[j][j];
        if (z === 0) { th[j] = 0; continue; }
        // ρ_j = Σ x_ij (y_i − Σ_{k≠j} x_ik θ_k) = −Σ x_ij r_i + z_j θ_j
        let rho = z * th[j];
        for (let i = 0; i < N; i++) rho -= this.rows[i][j] * res[i];
        const next = j === D - 1 ? rho / z : soft(rho, this.l1s) / (z + this.l2s);
        const delta = next - th[j];
        if (delta !== 0) {
          for (let i = 0; i < N; i++) res[i] += this.rows[i][j] * delta;
          th[j] = next;
          maxDelta = Math.max(maxDelta, Math.abs(delta));
        }
      }
      if (maxDelta < tol) break;
    }
    this.theta = th;
  }

  /**
   * H⁻¹g for the implicit derivative, as a (d+1)-vector. With an ℓ1 term the derivative only
   * exists on the active set (sign pattern fixed): inactive coefficients stay at 0 for an
   * infinitesimal move, so their rows/columns of H⁻¹ are zero. The paper's Eq. (7) instead uses
   * g = 0 on the full matrix, which lets inactive weights move; the active-set restriction is the
   * exact derivative of the LASSO solution.
   */
  applyHessianInverse(g: number[]): number[] {
    this.syncInverse();
    if (this.refine) return this.refined(g, REFINE_FLOOR_GRAD);
    if (this.closedForm) return Array.from(this.inv!.apply(g));
    const act = this.act!;
    const vA = this.inv!.apply(act.map(j => g[j]));
    const out = new Array(this.dAug).fill(0);
    act.forEach((j, a) => { out[j] = vA[a]; });
    return out;
  }

  /** Release the inverse (device memory on the GPU backend). The fit is unusable afterwards. */
  dispose() {
    this.replaceInverse(null);
    this.act = null;
    this.Xd?.dispose();
    this.Xd = null;
  }
}

/** Fit once and return the display-friendly raw model. */
export function fitLinear(X: number[][], y: number[], type: RegType, lambda: number) {
  const fit = new RegressionFit(X, y, type, lambda);
  const model = { theta: [...fit.theta], type, lambda: fit.lambda };
  fit.dispose();
  return model;
}

/** Predict with model */
export function ridgePredict(model: any, x: number[]): number {
  const t = model.theta;
  let s = t[t.length - 1];
  for (let j = 0; j < x.length; j++) s += t[j] * x[j];
  return s;
}

/** MSE on a dataset */
export function mse(model: any, X: number[][], y: number[]): number {
  let sum = 0;
  for (let i = 0; i < X.length; i++) {
    const err = ridgePredict(model, X[i]) - y[i];
    sum += err * err;
  }
  return sum / X.length;
}

/** R² score */
export function r2Score(model: any, X: number[][], y: number[]): number {
  const yMean = y.reduce((s, v) => s + v, 0) / y.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < X.length; i++) {
    const err = y[i] - ridgePredict(model, X[i]);
    ssRes += err * err;
    ssTot += (y[i] - yMean) * (y[i] - yMean);
  }
  return 1 - ssRes / (ssTot + 1e-12);
}

/** Get model state for display */
export function getRidgeModelState(model: any, X: number[][], y: number[], testX?: number[][], testY?: number[]) {
  const mseVal = mse(model, X, y);
  const r2 = r2Score(model, X, y);
  const testMSE = testX && testY ? mse(model, testX, testY) : mseVal;
  const testR2 = testX && testY ? r2Score(model, testX, testY) : r2;
  const w = model.theta.slice(0, -1);
  const wNorm = Math.sqrt(w.reduce((s: number, v: number) => s + v * v, 0));

  return {
    theta: [...model.theta],
    w,
    b: model.theta[model.theta.length - 1],
    lambda: model.lambda,
    mse: mseVal,
    r2,
    testMSE,
    testR2,
    wNorm,
    regType: model.type
  };
}
