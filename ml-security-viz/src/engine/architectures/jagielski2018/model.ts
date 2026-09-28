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

/** ℓ1 solves stop when every inactive |X_jᵀr| ≤ l1s(1 + KKT_TOL) and the face equations hold to
 *  KKT_TOL·l1s, i.e. the KKT conditions hold to ~1e-9 relative. */
const KKT_TOL = 1e-9;
/** A column whose Schur complement (squared distance to span(X_A), plus ridge) is below this
 *  fraction of H_jj counts as linearly dependent on the active columns. */
const DEP_TOL = 1e-10;

/**
 * A regression fit on N rows whose last rows (the poisoning points) can be replaced one at a
 * time, as Algorithm 1 does. Xᵀy is kept current under row swaps, and so is the inverse the
 * implicit derivative needs, via Sherman–Morrison rank-one updates (`compute().rankOneInverse`,
 * resident on the GPU when the attack server has one):
 *   ridge / OLS   (XᵀX + l2s·I_w)⁻¹, θ = that · Xᵀy. XᵀX is only read to rebuild that inverse,
 *                 which happens right after a refit, so a row swap merely marks it stale.
 *   ℓ1 models     (X_AᵀX_A + l2s·I_w)⁻¹ on the active set A, θ_A from the KKT system when the sign
 *                 pattern holds, warm-started feature-sign search when it does not. The KKT
 *                 checks read XᵀX, so it is updated on every swap (O(d²)).
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
    for (let j = 0; j < D; j++) if (j === D - 1 || this.theta[j] !== 0) act.push(j);
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
    } else {
      const outcome = this.signPatternSolve();
      if (outcome === 'optimal') return;
      if (!this.featureSignSearch(outcome === 'pattern')) this.featureSignSearch(false);
      this.syncInverse();
      // Verify with the inverse the search handed over; if that has drifted, with a fresh one.
      if (this.signPatternSolve() !== 'optimal') {
        this.replaceInverse(this.freshInverse(this.act));
        this.signPatternSolve();
      }
    }
  }

  /** H_jk = XᵀX + l2s on the weight diagonal (the bias is not regularised). */
  private H(j: number, k: number) {
    return this.XtX[j][k] + (j === k && j !== this.dAug - 1 ? this.l2s : 0);
  }

  /**
   * H_A x = c for the columns `idx`, from an approximate inverse `apply` plus iterative
   * refinement against the exact H_A (clears Sherman–Morrison / bordering drift). Returns x and
   * the final max-norm residual.
   */
  private faceSolve(apply: (v: ArrayLike<number>) => Float64Array, idx: number[], c: number[]) {
    const n = idx.length, x = apply(c), r = new Array(n);
    let res = Infinity;
    for (let it = 0; ; it++) {
      res = 0;
      for (let a = 0; a < n; a++) {
        const row = this.XtX[idx[a]];
        let s = c[a] - (idx[a] !== this.dAug - 1 ? this.l2s * x[a] : 0);
        for (let b = 0; b < n; b++) s -= row[idx[b]] * x[b];
        r[a] = s;
        res = Math.max(res, Math.abs(s));
      }
      if (!(res > KKT_TOL * this.l1s) || it === 3) return { x, res };
      const dx = apply(r);
      for (let a = 0; a < n; a++) x[a] += dx[a];
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
   * One Newton step on the current sign pattern (the active set of θ, whose inverse is kept
   * current across row swaps). Accepts it only if it is the exact ℓ1 optimum: the face equations
   *   (X_AᵀX_A + l2s·I_w) θ_A = X_Aᵀy − l1s·sign(θ_A)
   * hold, no sign flipped, and every inactive j has |X_jᵀ(y − Xθ)| ≤ l1s. This is the common
   * case after a single-row replacement. Otherwise reports whether the sign pattern changed
   * ('pattern') or the inverse is no longer accurate enough to tell ('inaccurate', e.g. a
   * row swap made H_A singular).
   */
  private signPatternSolve(): 'optimal' | 'pattern' | 'inaccurate' {
    const D = this.dAug, act = this.act!, na = act.length;
    const sign = act.map(j => (j === D - 1 ? 0 : Math.sign(this.theta[j])));
    const { x, res } = this.faceSolve(v => this.inv!.apply(v), act, act.map((j, a) => this.Xty[j] - this.l1s * sign[a]));
    if (!(res <= KKT_TOL * this.l1s)) return 'inaccurate';              // also catches NaN
    for (let a = 0; a < na; a++) if (act[a] !== D - 1 && Math.sign(x[a]) !== sign[a]) return 'pattern';
    const isAct = new Array(D).fill(false);
    act.forEach(j => { isAct[j] = true; });
    for (let j = 0; j < D; j++) {
      if (isAct[j]) continue;
      let g = this.Xty[j];                                               // X_jᵀ(y − X_A θ_A)
      const row = this.XtX[j];
      for (let a = 0; a < na; a++) g -= row[act[a]] * x[a];
      if (!(Math.abs(g) <= this.l1s * (1 + KKT_TOL))) return 'pattern';
    }
    const next = new Array(D).fill(0);
    act.forEach((j, a) => { next[j] = x[a]; });
    this.theta = next;
    return 'optimal';
  }

  /**
   * Exact solve of ½Σr² + l1s‖w‖₁ + ½l2s‖w‖² by feature-sign search (Lee et al., NIPS 2006),
   * warm-started from θ. Every step strictly decreases the objective, so it cannot cycle:
   *   - at a face optimum, the most violating inactive j (|X_jᵀr| > l1s) joins A with sign(X_jᵀr);
   *   - otherwise move from θ_A towards the face minimiser θ̂_A = H_A⁻¹(X_Aᵀy − l1s·s_A), stopping
   *     at the first coefficient that reaches zero, which leaves A.
   * One-hot groups are collinear with the bias and small samples contain duplicate columns, so
   * X_A would often be rank-deficient (LASSO has no ridge to fix that) and H_A singular. A is
   * therefore kept linearly independent: a column j in span(X_A), X_j = X_A a, is instead
   * pivoted in along s_j(e_j − a), which leaves the fit unchanged and lowers ‖w‖₁ at rate
   * |X_jᵀr| − l1s > 0, until an active coefficient reaches zero and swaps out.
   * H_A⁻¹ starts from the inverse kept for θ's active set when `reuseInverse` (it is still
   * accurate), else is rebuilt column by column, and is then kept by O(|A|²) bordering updates.
   * Every face solve is refined against the exact H_A (H_A⁻¹ is recomputed if that does not
   * converge), and the final inverse is handed over to `inv` for the implicit derivative.
   * Returns whether the KKT conditions were reached.
   */
  private featureSignSearch(reuseInverse: boolean): boolean {
    const D = this.dAug, bias = D - 1, l1s = this.l1s;
    const th = new Float64Array(D), sgn = new Float64Array(D);
    const A: number[] = [];
    const M = Array.from({ length: D }, () => new Float64Array(D));    // H_A⁻¹, rows/cols in A's order
    const isAct = new Uint8Array(D);

    const apply = (v: ArrayLike<number>) => {
      const n = A.length, out = new Float64Array(n);
      for (let p = 0; p < n; p++) {
        const Mp = M[p];
        let s = 0;
        for (let q = 0; q < n; q++) s += Mp[q] * v[q];
        out[p] = s;
      }
      return out;
    };
    /** a = H_A⁻¹h_A(j) and the Schur complement σ = H_jj − h_Aᵀa (> 0 iff j is independent of A). */
    const schur = (j: number) => {
      const h = A.map(k => this.H(k, j));
      const { x: a } = this.faceSolve(apply, A, h);
      let sigma = this.H(j, j);
      for (let p = 0; p < A.length; p++) sigma -= h[p] * a[p];
      if (sigma < 1e-6 * this.H(j, j)) {
        // cancellation-free form: σ = ‖X_j − X_A a‖² + l2s(1 + ‖a_w‖²)
        sigma = 0;
        for (let i = 0; i < this.N; i++) {
          const r = this.rows[i];
          let e = r[j];
          for (let p = 0; p < A.length; p++) e -= r[A[p]] * a[p];
          sigma += e * e;
        }
        if (j !== bias) sigma += this.l2s;
        for (let p = 0; p < A.length; p++) if (A[p] !== bias) sigma += this.l2s * a[p] * a[p];
      }
      return { a, sigma, independent: sigma > DEP_TOL * this.H(j, j) };
    };
    const add = (j: number, a: Float64Array, sigma: number) => {
      const n = A.length, Mn = M[n];
      for (let p = 0; p < n; p++) {
        const Mp = M[p], ap = a[p] / sigma;
        for (let q = 0; q < n; q++) Mp[q] += ap * a[q];
        Mp[n] = -ap;
        Mn[p] = -ap;
      }
      Mn[n] = 1 / sigma;
      A.push(j);
      isAct[j] = 1;
    };
    const remove = (p: number) => {
      const n = A.length, L = n - 1, Mp = M[p], piv = Mp[p];
      for (let i = 0; i < n; i++) {
        if (i === p) continue;
        const Mi = M[i], f = Mi[p] / piv;
        if (f !== 0) for (let k = 0; k < n; k++) Mi[k] -= f * Mp[k];
      }
      isAct[A[p]] = 0;
      th[A[p]] = 0;
      if (p !== L) {
        [M[p], M[L]] = [M[L], M[p]];
        for (let i = 0; i < L; i++) M[i][p] = M[i][L];
        A[p] = A[L];
      }
      A.pop();
    };

    // Warm start from θ: its active set with the inverse kept for it, or else its support added
    // column by column (bias first), dropping columns dependent on those already in.
    if (reuseInverse) {
      const act = this.act!, n = act.length, inv = this.inv!.read();
      act.forEach((j, p) => {
        M[p].set(inv.subarray(p * n, (p + 1) * n));
        A.push(j);
        isAct[j] = 1;
      });
    } else {
      for (const j of [bias, ...this.activeSet().filter(j => j !== bias)]) {
        const { a, sigma, independent } = schur(j);
        if (independent) add(j, a, sigma);
      }
    }
    for (const j of A) {
      th[j] = this.theta[j];
      sgn[j] = j === bias ? 0 : Math.sign(th[j]);
    }

    /** Face minimiser θ̂_A, recomputing H_A⁻¹ (A is independent) if the refinement stalls. */
    const faceMinimiser = () => {
      const c = A.map(j => this.Xty[j] - l1s * sgn[j]);
      const first = this.faceSolve(apply, A, c);
      if (first.res <= KKT_TOL * l1s) return first.x;
      const n = A.length, H = new Float64Array(n * n);
      for (let p = 0; p < n; p++) for (let q = 0; q < n; q++) H[p * n + q] = this.H(A[p], A[q]);
      const inv = compute().inverse(H, n);
      for (let p = 0; p < n; p++) M[p].set(inv.subarray(p * n, (p + 1) * n));
      return this.faceSolve(apply, A, c).x;
    };

    let converged = false;
    for (let iter = 0; iter < 20 * D; iter++) {
      const n = A.length;
      const tHat = faceMinimiser();
      let t = 1, hit = -1;
      for (let p = 0; p < n; p++) {
        const j = A[p];
        if (j === bias || Math.sign(tHat[p]) === sgn[j]) continue;
        const tp = th[j] / (th[j] - tHat[p]);
        if (tp < t) { t = tp; hit = p; }
      }
      if (hit >= 0) {                                   // first zero crossing: step there, drop it
        for (let p = 0; p < n; p++) th[A[p]] += t * (tHat[p] - th[A[p]]);
        remove(hit);
        continue;
      }
      for (let p = 0; p < n; p++) th[A[p]] = tHat[p];   // face optimum

      let best = -1, gBest = l1s * (1 + KKT_TOL);
      for (let j = 0; j < D; j++) {
        if (isAct[j]) continue;
        let g = this.Xty[j];
        const row = this.XtX[j];
        for (let p = 0; p < n; p++) g -= row[A[p]] * th[A[p]];
        if (Math.abs(g) > gBest) { gBest = Math.abs(g); best = j; sgn[j] = Math.sign(g); }
      }
      if (best < 0) { converged = true; break; }        // KKT hold: optimal

      const s = schur(best);
      if (s.independent) { add(best, s.a, s.sigma); continue; }
      // X_best ∈ span(X_A): pivot it in along sgn·(e_best − a)
      const sj = sgn[best];
      let tr = Infinity, out = -1;
      for (let p = 0; p < n; p++) {
        const j = A[p], dp = -sj * s.a[p];
        if (j === bias || dp * th[j] >= 0) continue;
        const tp = -th[j] / dp;
        if (tp < tr) { tr = tp; out = p; }
      }
      if (out < 0) break;                               // unbounded ray: impossible for this objective
      for (let p = 0; p < n; p++) th[A[p]] -= tr * sj * s.a[p];
      remove(out);
      const s2 = schur(best);
      if (!(s2.sigma > 0)) break;
      add(best, s2.a, s2.sigma);
      th[best] = sj * tr;
    }

    this.theta = Array.from(th);
    const order = A.map((_, p) => p).sort((p, q) => A[p] - A[q]), n = A.length;
    const flat = new Float64Array(n * n);
    order.forEach((p, a) => { order.forEach((q, b) => { flat[a * n + b] = M[p][q]; }); });
    this.act = order.map(p => A[p]);
    this.replaceInverse(compute().rankOneInverse(flat, n));
    return converged;
  }

  /**
   * H⁻¹g for the implicit derivative, as a (d+1)-vector. With an ℓ1 term the derivative only
   * exists on the active set (sign pattern fixed): inactive coefficients stay at 0 for an
   * infinitesimal move, so their rows/columns of H⁻¹ are zero. The paper's Eq. (7) instead uses
   * g = 0 on the full matrix, which lets inactive weights move; the active-set restriction is the
   * exact derivative of the LASSO solution.
   * Where that solution is degenerate — an inactive column in span(X_A) with |X_jᵀr| = l1s, as
   * duplicate or one-hot columns produce whenever the poisoning points are one-hot clones — no
   * two-sided derivative exists (for LASSO the optimum is then a flat segment, θ jumps between
   * its ends); this is the derivative along the current active set, a one-sided one.
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
