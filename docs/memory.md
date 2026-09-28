# Memory — ML Security Attack Visualizer

> **Last Updated**: 2026-08-08

---

## Current Status: 🟢 Active Development — 3 Attack Algorithms Implemented

### ✅ Completed

| Item | File(s) | Date |
|---|---|---|
| Research references & Biggio 2012 paper | — | 2025-07-30 |
| Analyzed PoisonRegression codebase | `poisoning/poison.py`, `gd_poisoners.py` | 2025-07-30 |
| Created all 6 docs (PRD, Architecture, Rules, Phases, Design, Memory) | `docs/*.md` | 2025-07-30 |
| Initialized Next.js project + Tailwind CSS v4 | `ml-security-viz/` | 2025-07-30 |
| Installed deps: zustand, d3, recharts, katex | `package.json` | 2025-07-30 |
| Created engine: linalg.ts, kernels.ts | `src/engine/` | 2025-07-30 |
| Created SVM solver (SMO) | `src/engine/architectures/biggio2012/model.ts` | 2025-07-30 |
| Created dataset generators + real-world datasets | `src/engine/data/datasets.ts` | 2025-07-30 |
| Created Zustand store | `src/store/useStore.ts` | 2025-07-30 |
| Created all UI components (Canvas, MathPanel, Timeline, etc.) | `src/components/` | 2025-07-30 |
| Built scalable multi-algorithm registry framework | `src/engine/architectures/registry.ts` | 2026-07-30 |
| Implemented Biggio 2012 SVM Poisoning | `src/engine/architectures/biggio2012/` | 2026-07-30 |
| Implemented Jagielski 2018 Regression Poisoning (KKT) | `src/engine/architectures/jagielski2018/` | 2026-07-30 |
| Added regression datasets (Linear, Quad, Sine, Warfarin, Lending, House) | `datasets.ts` | 2026-07-30 |
| LASSO (Coordinate Descent) and OLS regression | `jagielski2018/model.ts` | 2026-07-30 |
| GuidedTour + TutorialModal + Concept Explainer | `GuidedTour.tsx`, `TutorialModal.tsx`, `ExplainerCard.tsx` | 2026-07-30 |
| Full design system alignment with `design.md` | All components | 2026-08-07 |
| **Refactored jagielski2018 into model.ts + attack.ts + index.ts** | `jagielski2018/` | 2026-08-08 |
| **Extended TraceFrame with online learning fields** | `registry.ts` | 2026-08-08 |
| **Added explainerSteps to AlgorithmModule interface** | `registry.ts` | 2026-08-08 |
| **Implemented Pang 2021 Accumulative Poisoning (online learning)** | `pang2021/` | 2026-08-08 |
| **Created ExplainerOverlay (Transformer Explainer-style)** | `ExplainerOverlay.tsx` | 2026-08-08 |
| **Created DataFlowDiagram (animated SVG pipeline)** | `DataFlowDiagram.tsx` | 2026-08-08 |
| **Created CONTRIBUTING.md for collaborators** | `CONTRIBUTING.md` | 2026-08-08 |
| **Dual light/dark theme + typography overhaul** | `globals.css`, `layout.tsx`, `ThemeToggle.tsx`, `useThemeTokens.ts`, `docs/design.md` | 2026-08-11 |

---

## Theme System (added 2026-08-11)

The design system was dark-only; it now ships **light and dark**, both tuned rather than one being a tint of the other. Raw tokens are declared twice (`:root` = light, `.dark` = dark) and `@theme inline` maps them to utilities that re-resolve at runtime, so a class on `<html>` switches everything. Every token pair was measured — all clear WCAG AA 4.5:1 in both themes.

Four latent bugs surfaced and were fixed along the way:

1. **The reset block was unlayered.** Unlayered CSS beats every layered rule, so `button { background: none }` was overriding `bg-primary` (the play button had *no* fill), `h1 { font-size: 2xl }` was overriding `text-base`, and `a { color: accent }` was overriding link color utilities. Wrapping the reset in `@layer base` fixed all three.
2. **Web fonts were never actually applied.** `@theme` named the literal families (`"Inter"`), but `next/font` registers hashed family names exposed as CSS variables — so the whole UI silently fell back to system fonts. Font tokens now point at `var(--font-inter)` etc.
3. **Dead tokens.** `--bg-canvas`, `--accent-primary`, `--text-secondary`, `--text-tertiary`, `--color-clean-dim`, `--color-attack-dim`, `--border-default` survived the Tailwind migration in `Canvas.tsx`, `MathEq.tsx`, and `InverseImagePanel.tsx`, resolving to nothing (quiver arrows and residual lines were rendering as invalid fills).
4. **The concept-explainer UI was hardcoded light** (`bg-white`, `bg-gray-50`, `text-gray-500`) inside a dark app — unreadable before this change, correct in both themes now.

Also new: `useThemeTokens()` for Recharts/JS-computed SVG color (the only place CSS can't reach), and tabular/slashed-zero figures on `.data-value`/`.metric-value` so metrics stop jittering as they update.

---

## Pang 2021 correctness pass (2026-08-11)

Checked `pang2021/` against the paper (arXiv:2106.09993, Algorithm 1). The attack was optimising the wrong objective and has been rewritten.

**Eq. 7** is a joint minimisation over the poisoning operator 𝒫 and the accumulative operator 𝒜:

```
min_{P,A}  ∇L(S_val; A(θ_T))ᵀ ∇L(P(S_T); A(θ_T))
s.t.       L(S_val; A(θ_T)) ≤ L(S_val; θ_T) + γ
```

The inner product is the whole mechanism: to first order `L(S_val; θ_{T+1}) ≈ L(S_val; θ_T) − β⟨∇L(S_val), ∇L(P(S_T))⟩`, so driving the alignment negative is what makes one trigger step raise validation loss.

What was wrong, and what it is now:

| # | Was | Now |
|---|---|---|
| 1 | Accumulative phase maximised the **loss of the clean trigger batch** after the step — a proxy for gradient *magnitude* that ignores direction, which is the entire point | Follows Algorithm 1: `G_t = ∇_θ⟨∇L(S_val), ∇L(P(S_T))⟩`, `d_t = ∇L(S_t) + λG_t`, PGD maximises `H_t = ⟨∇L(A_t(S_t)), d_t⟩` |
| 2 | Trigger crafted **once at the end**, so accumulation steered θ against a batch that was then discarded — breaks the joint min | `P(S_T)` refined every round alongside `A_t` |
| 3 | No "keep learning" term, so perturbed updates had no pressure to look normal | The `∇L(S_t)` term in `d_t`; λ sets the mix |
| 4 | Stealth = absolute accuracy floor on the **test** set. If clean accuracy sat below τ, *every* batch was rejected and the phase silently no-opped | γ budget against a **counterfactual clean trajectory** θ̃ on the validation split, plus the paper's early stop |
| 5 | `runAttack` ignored the `cleanModelState` handed to it and retrained with a fresh random init — the attack started from different weights than the "clean" boundary drawn on the canvas | Starts from the handed-in model |
| 6 | Numerical gradient took a full SGD step + trigger-loss evaluation per coordinate | Analytic `lossGradient`; central differences only where genuinely needed. Faster *and* correct |

**Verified numerically** (headless harness, 5 seeds): the first-order forecast tracks reality (predicted Δloss 0.0279 vs actual 0.0314), and at the tuned defaults accumulation drops accuracy **22.7 pts vs 3.3 pts for the λ=0 control** — i.e. the accumulative phase, not the trigger, is what does the damage. That is the paper's central claim and it now reproduces.

**Honest caveat, surfaced in the UI**: the old defaults (ε=0.3, λ=1, β=0.5) produced *zero* measurable effect. Three-parameter logistic regression gives the attacker far less room than the paper's deep nets, so defaults are now ε=0.8, λ=25, β=1.5, γ=0.6 — larger than anything one would call imperceptible. The mechanism is faithful; the magnitudes are exaggerated so the effect is visible. The last explainer card says so explicitly.

## Pang 2021, second pass — why the attack still did nothing (2026-08-11)

Reported symptom: accuracy pinned at 100% across the whole run, trigger included. Checked against the
reference implementation ([ShawnXYang/AccumulativeAttack](https://github.com/ShawnXYang/AccumulativeAttack)).

Three causes, two of them ours:

1. **Gradients were combined un-normalised.** The reference builds its objective from L2-normalised
   gradients — `F.normalize(grad_tri) @ F.normalize(grad_val)` and
   `F.normalize(grad_train_adv) @ (gamma * F.normalize(grad_train))`. Ours mixed raw gradients, whose
   magnitudes differ by orders of magnitude, so whichever term happened to be larger won outright and λ
   barely registered. `d_t` is now built from unit vectors, making λ a genuine mixing weight
   ("λ≈1 = half honest learning, half accumulation"). Its range dropped from 0–50 to 0–5 accordingly.
2. **The victim was fully converged.** `trainClean` ran 100 epochs. The reference has a dedicated
   *burn-in* phase (`train_cifar.py`) and then keeps training online — the victim is mid-training when
   the attacker arrives. A converged 3-parameter model has ∇L → 0 and nothing to hijack. Added a
   **Burn-in Epochs** control (default 20).
3. **A separable dataset makes the attack impossible, and that is not a bug.** At ~100% accuracy with a
   wide margin there is no point close enough to the boundary for one bounded step to flip. The reported
   run used exactly that regime. The explainer now *diagnoses* this instead of leaving a flat line
   unexplained ("data too separable" / "raise λ / lower burn-in").

Also: batch sizing no longer silently truncates (asking for 40 accumulative rounds on 90 points used to
give 22), and `cleanAccuracy` is now re-measured each round from the counterfactual clean trajectory
rather than frozen at iteration 0 — the victim is still learning while the attacker accumulates.

Verified in-app on overlapping Gaussians: mid-accumulation the poisoned model reads **86.7% vs 83.3%
clean** — indistinguishable, which is the whole point — then collapses to **56.7% vs 80%** on the trigger.
Effect size is seed-dependent in 2-D; burn-in and dataset separability dominate everything else.

## Layout + explainer rework (2026-08-11)

- **`ControlPanel` could not scroll**: it carried both `overflow-y-auto` and `overflow-hidden`, and the later one won. ~250px of controls were unreachable. Fixed, plus `min-h-0` on the flex columns and a `min-h-[220px]` floor on the plot so the bottom strips can't squeeze it away. Strip padding unified to `p-4`.
- **Switching algorithms bricked the app**: `setActiveAlgorithm` nulls the dataset, but the regeneration effect was keyed only on `datasetKey`/`numPoints`, so it never re-fired — the app sat on "Generating dataset…" and every downstream action silently no-opped. `generateDataset` now depends on the active algorithm and falls back to a supported dataset key when the current one doesn't apply (e.g. `moons` → a regression set for jagielski2018).
- **Explainer rebuilt in the Transformer-Explainer mould**: `ExplainerStep` gained `liveEquation`, `readouts`, and `visual`, all evaluated against the frame the user is scrubbed to, so the maths shows the model's real numbers. New `ExplainerVisual.tsx` animates the geometry — the gradient-alignment wedge (turns red past 90°), the `d_t = ∇L(S_t) + λG_t` parallelogram, and predicted-vs-actual loss bars. The panel follows the attack's phase automatically and can be pinned to manual.

## Paper-fidelity pass: Biggio 2012 + Jagielski 2018 (2026-09-28)

Both modules re-checked line by line against the papers; every analytic gradient is now verified against central finite differences.

**Biggio 2012**
- **SMO did not converge** (KKT violations 1–2): the simplified SMO's fixed max|Eᵢ−Eⱼ| partner stalled and exited. Replaced with LIBSVM's maximal-violating-pair SMO (WSS1, `KKT_TOL = 1e-5`). Every SVM in the app had been under-trained.
- Gradient was finite differences through full retraining; now the closed form of Eq. (3)/(7)/(10). Three corrections to the printed maths: (i) ∂L/∂u carries a minus sign Eq. (10) omits; (ii) Eq. (7) is solved on the bordered matrix — Eq. (8)'s Q_ss⁻¹ is singular for a linear kernel with > d margin SVs; (iii) when x_c becomes a margin SV (0 < α_c < C) its own KKT row joins the system — holding α_c fixed flips the gradient's sign (seen on MNIST).
- Multi-point attack is sequential single-point (§4), fixed label y_c, fixed step t, projection onto the feasible box. Default runs the full iteration budget: the literal ΔL < ε rule stops at the first dip caused by an SV-set change.
- Real MNIST 7v1 / 9v8 / 4v0 in 784-d pixel space (100 train / 500 valid / full test), paper Gaussian set (µ± = ±1.5, Σ = 0.6I, 25/500 per class). One point: test error 2% → 16.5%; three points: → 30% (paper: 15–20% for one point).

**Jagielski 2018**
- Algorithm 1 as printed: points updated one at a time with retraining, joint (x_c, y_c) gradient (Eq. 14) in closed form via one H⁻¹-vector product, line search η·βᵏ, projection onto [0,1], InvFlip/BFlip, W_tr and W_val, elastic net, per-sample λ (Eq. 1) chosen by validation.
- `RegressionFit` keeps XᵀX and the needed inverse current under row swaps (Sherman–Morrison); ℓ1 models solve by an active-set iteration with KKT checks, coordinate descent only as fallback.
- Datasets are the authors' own preprocessed files (github.com/jagielski/manip-ml): Warfarin (204 features), Loan (88, 5000-record sample), Ames House (274). The previous "warfarin"/"lendingClub"/"housePricing" were the insurance, California-housing and CreditScoring datasets reduced to one feature.

**Data + UI**
- `scripts/fetch_datasets.mjs` (run by `predev`/`prebuild`) downloads MNIST and the regression CSVs into `public/data/` (git-ignored).
- Datasets may carry `view` (PCA window, `engine/data/view.ts`), `bounds`, `displayRange`, `imageShape`, `classNames`. The canvas draws high-dimensional data on its PCA plane/line and evaluates the algorithm's own raw models there (`alg.predict`) instead of retraining an SVM — this also fixes Pang's canvas, which used to show an SVM boundary.
- Pang still uses the synthetic `mnist`/`cifar` stand-ins; not yet reviewed.

## Biggio attack: early stopping + UI cleanup (2026-09-28)

- **Why the attack "kept going"**: with a fixed step, later iterations only fit the attacker's own validation sample (optimised loss rises, held-out loss flat) or oscillate around a maximum (RBF on the Gaussian set bounced between two positions forever). On the generic 2-D sets the attacker also had just 20% of ~150 points (30) as D_val.
- **Stopping rule `holdout` (default)**: ascend Eq. (10) on 80% of D_val, stop after 10 steps without improvement on the other 20%, commit the best held-out iterate. Seeded comparison: same or more damage than the full budget in ~1/5 of the steps (Moons RBF 5 pts: 95–127 vs 505 steps). Halving t on every loss dip was tried and rejected — t collapsed and the attack got weaker. `paper` (ΔL < 1e-5) and `none` remain selectable.
- `AlgorithmModule.evalPoints`: Biggio asks for 500-point validation and test sets on the synthetic 2-D generators (paper: 500/class); training size unchanged.
- Pinned-structure fallback: when Eq. (10) is exactly 0 because d+1 margin SVs fix (w, b) (linear kernel in 2-D), a secant direction at the step size is used for d ≤ 10.
- UI: DataFlowDiagram removed (component, page slot, store state). Timeline plots the held-out objective, marks where each poison point starts, and zooms the accuracy axis. Attack Details shows the held-out objective and the active point; the canvas badge names the point; trajectories mark their start. `AlgorithmModule.score` (optional continuous decision value) gives Pang a smooth boundary instead of a ±1 staircase.

---

## Architecture

### Algorithm Registry Pattern

Every attack paper is encapsulated as an `AlgorithmModule` registered in `src/engine/architectures/registry.ts`:

```
src/engine/architectures/
├── registry.ts          # AlgorithmModule interface + global registry
├── index.ts             # Barrel file — imports all algorithm modules
├── biggio2012/          # SVM Poisoning (Classification)
│   ├── index.ts         # Registration + configSchema
│   ├── model.ts         # SVM (SMO solver)
│   ├── attack.ts        # Gradient ascent poisoning loop
│   └── kernels.ts       # Kernel functions (linear, RBF, poly)
├── jagielski2018/       # Regression Poisoning (Ridge/LASSO/OLS)
│   ├── index.ts         # Registration + configSchema
│   ├── model.ts         # Ridge, LASSO, OLS training
│   └── attack.ts        # Bilevel optimization via KKT
└── pang2021/            # Accumulative Poisoning (Online Learning)
    ├── index.ts          # Registration + configSchema
    ├── model.ts          # Logistic regression + SGD
    ├── attack.ts         # Two-phase: accumulate + trigger
    ├── gradient.ts       # PGD perturbation computation
    └── explainer.ts      # Transformer Explainer-style step cards
```

### Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | Next.js 16 (App Router) | Modern React, optimized builds |
| State | Zustand | Lightweight, hook-based |
| Charts | Recharts | Native React, built on D3 |
| Canvas | Raw SVG + marching squares | React-friendly, no D3 DOM |
| Compute | setTimeout loops | Quick iteration; Web Worker ready |
| Styling | Tailwind CSS v4 | CSS-based configuration |
| Registry pattern | AlgorithmModule interface | Enables plug-and-play attack modules |

### Component Hierarchy

```
page.tsx
├── AppHeader
├── ControlPanel          # Left sidebar — dynamic config from AlgorithmModule
├── Canvas               # Center — 2D scatter + decision boundary/regression line
├── Timeline             # Bottom — loss/metric charts over iterations
├── DataFlowDiagram      # Bottom — animated clean vs poisoned pipeline
├── MathPanel            # Right sidebar — model state inspector
├── PlaybackBar          # Bottom bar — play/pause/step controls
├── PointInspector       # Floating — per-point math details
├── ExplainerOverlay     # Floating — step-by-step guided learning
├── GuidedTour           # Floating — algorithm walk-through
└── TutorialModal        # Modal — first-time onboarding
```

## RBF on MNIST sat at chance (2026-09-28)

- **Cause**: fixed γ (store default 0.8) on 784-pixel data. Squared distances between digits are ~80, so K = e^(−0.8·80) ≈ 1e-28 for every pair: the SVM predicted one class (52.5%) and every kernel gradient vanished, so the attack could not move anything.
- **Fix**: `gammaMode` — `scale` (default, sklearn's γ = 1/(d·Var X): 0.017 on MNIST → 98% clean, ≈0.29 on the paper Gaussian set) or `manual` (the old slider). One `kernelSpec(config, X)` feeds training, the attack, the heatmap worker and the re-attached kernel in page.tsx. Clean-model panel shows the γ used and warns when accuracy is near chance.
- **RBF poisoning is weak at C = 1 by construction**: K ≤ 1, so each point moves f by at most α_c ≤ C, locally (10 points: 98.6% → 97.4%). With C = 10, 5 points: 98.3% → 93.4%. Linear, where K = x·x′ ~ 20–100, is the paper's MNIST setting (3 points → ~30% error).
- `cachedKernel`: memoises kernel values by argument identity; only x_c changes per step, so the MNIST RBF attack runs ~6× faster.
- Canvas: clean boundary memoised separately; poisoned boundary throttled to one redraw per 300 ms on projected (high-d) data, grid 28². Each redraw was 2 × 140 ms per frame for RBF MNIST, which pinned the main thread during an attack.

## Exports + original datasets served by the app (2026-09-28)

- **Export dialog** (`ExportDialog.tsx`, opened by the SVG / JSON buttons). The old SVG export serialised the *first* `<svg>` on the page (not necessarily the canvas) and lost all class-based styling; the old JSON dumped raw store objects incl. the whole dataset.
  - Snapshots (`lib/exportCanvas.ts`): the canvas SVG (`data-export="attack-canvas"`) with computed paint inlined, background, caption (algorithm, dataset, frame, metrics) and legend. Current frame, every N-th, all, or a list like `0, 10, 25-30, last`; several frames are captured by scrubbing (`isExporting` bypasses the boundary throttle) and zipped with `manifest.csv` (`lib/zip.ts`, STORE, no dependency).
  - Data (`lib/exportTrace.ts`): schema `ml-security-viz/attack-trace` v1 — algorithm, dataset metadata, the config snapshotted at launch (`attackMeta`), resolved kernel/λ, summary, field descriptions, and per-frame metrics collected generically from each TraceFrame. Optional: poison coordinates, model parameters (SVs as indices + α), gradient/vector fields, dataset. Metrics CSV has one row per frame.
  - Dev builds expose the pipeline on `window.__mlsvExport` for scripted checks.
- **Original datasets on the server**: complete MNIST, CIFAR-10 and the full regression files live in `data/`; `/api/datasets/images` and `/api/datasets/regression` sample per request. Pang's hand-drawn "MNIST"/"CIFAR" stand-ins are gone: `mnist` / `cifar` are now real MNIST 7v1 and CIFAR-10 ship-vs-frog on 2-D PCA (`loaders/images.ts`), with a PCA state whose inverse maps canvas points back to pixels exactly.
- Compute still runs client-side; a GPU on the host does not speed up attacks unless they are moved server-side.


## Attacks on the server + CUDA backend (2026-09-28)

- **Server execution**: `POST /api/attack` (`server/attack.ts`) runs a module's own `runAttack` in Node and streams gzip NDJSON frames, flushed per frame; the browser uploads the dataset + clean model it already has (gzip, `lib/serverAttack.ts`; MNIST ≈ 9 MB JSON → 0.6 MB). "Run Attack On" (`computeTarget`: auto/server/browser, `ComputeTarget.tsx`) chooses; `auto` falls back to the browser only when the server is unusable. `runAttack` gained an optional `signal`; disconnecting stops the server-side attack (CPU 120% → 0% within 1.5 s, measured). Clean training and the Fig. 1 heatmap stay in the browser.
- **Compute backend** (`engine/compute`): `gram`, `inverse`, `rankOneInverse`, `denseMatrix`. CPU implementation keeps the old summation order — `RegressionFit` on it is bit-identical to before (checked on House/Warfarin, all four regularisers). CUDA implementation: FP64 cuBLAS/cuSOLVER addon `native/cuda/mlsv_cuda.cu` (N-API C, built by `scripts/build_cuda.mjs` with nvcc, no node-gyp), inverse and design matrix kept resident on the GPU, per-op size thresholds from `npm run bench:compute` (RTX 4090 at n = 275: XᵀX 116×, inverse 21×, row swap 12×, A·v + Aᵀr 8×).
- **Where the time was** (profiles): Jagielski — Gauss–Jordan inverse, Sherman–Morrison swaps, and then (once those were on the GPU) the outer objective W, ~67%: evaluated ~200×/iteration with its gradient every time, over nested arrays. Now value-only in the line search, residuals via a resident matrix. Biggio — Eq. (10)'s ∂α term was summed per validation point (m·n_s·d); summing the coefficients over k first makes it n_s·d (exact reordering, 1e-14 vs old).
- **Measured per frame** (original browser code → new CPU → server RTX 4090): House ridge 1667 → 1122 → 146 ms; Warfarin 1005 → 756 → 140; House OLS 547 → 203 → 20; Loan 282 → 187 → 149; Biggio MNIST linear 68 → 33 → 39, RBF 119 → 56 → 57 (Biggio's per-step work is below GPU break-even; the gain is the CPU reordering, available in the browser too).
- **Verification**: new Jagielski outer objective = old formula to ≤ 8e-16 on both backends; ∇_{(x_c,y_c)}W vs central differences ~1e-8 relative on CPU and CUDA (ridge, LASSO). Biggio RBF finite differences need h ≳ 1e-2: below ~1e-4 the SMO tolerance (1e-5) dominates (MNIST: 1.197 vs analytic 1.200 at h = 1e-2).
- **Pre-existing, not fixed**: OLS (`OLS_RIDGE = 1e-5`) drifts between its every-32-update refits — on Warfarin predictions differ from an exact refit by up to 21 on test points and training MSE by 3% (the CUDA path happens to drift less). GPU memory per server process is ~430 MiB (context + library workspace), stable across aborted/completed runs.

## 3-D canvas view (2026-09-28)

- 2D/3D toggle at the canvas's top-right (`canvasView` in the store; `components/Canvas3D.tsx`). Classification: floor = the 2-D feature plane (PCA plane for MNIST), height = decision value f(x) of the model on screen (poisoned when a frame is shown), surface coloured by predicted class, clean + poisoned boundaries drawn on the f = 0 plane. Regression: floor = first two principal directions (a 2-D `fitView` computed on the fly — the dataset's own view is 1-D), height = y; poisoned prediction surface filled, clean one as wireframe. Disabled for the 1-D regression toys.
- Plain SVG, orthographic camera, drag to rotate / wheel to zoom / double-click to reset; faces, lines and points depth-sorted together. Model evaluation is memoised apart from projection, so rotating only re-projects; frames on high-d data are throttled like the 2-D boundary (`hooks/useThrottled.ts`, moved out of Canvas.tsx). The 2-D svg stays mounted (d3-zoom is bound once) but hidden, and `data-export="attack-canvas"` moves to whichever svg is visible.
- Checked in headless Chrome: Biggio Moons/MNIST, Jagielski House, Pang MNIST, dark mode, 1-D fallback — no console errors. SVG export from the 3-D view not yet exercised.
