# Memory — ML Security Attack Visualizer

> **Last Updated**: 2026-09-28

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

## Pang 2021 — third pass: pixel space on real data (2026-09-28)

Rechecked against the paper (§3.2, Eq. 6–9, Algorithm 1, §4) and the authors' `online_accu_train.py`.

- **What was wrong**: perturbations lived in a 2-D PCA of the images with ε = 0.8 on a [−2,2] canvas (≈20% of the range) — not the paper's L∞ pixel budget (8/255, 16/255). Synthetic Gaussian/Moons/Circles were still offered. G_t and every batch gradient were central differences over 3 parameters. Burn-in was full-batch GD. The early-stop floor was an absolute accuracy (0.6).
- **Now**: logistic regression on the raw pixels (785 / 3073 parameters), input normalised per channel like the reference ResNet's forward (plus 1/√d so one β fits MNIST and CIFAR); PGD in pixel space with α = 2ε/C and clipping to [0,1]; mini-batch SGD burn-in; stream of fresh batches with a held-out trigger batch. Algorithm 1: P(S_T) crafted at θ₀ (reference `craft_tri`, ball around the clean trigger), per round G_t → A_t by ascending ĝ(A_t)ᵀ(ĝ(S_t) + λĜ_t) → update → secrecy check (Eq. 7 γ vs the clean trajectory, §4 accuracy floor as a drop from θ₀) → optionally re-craft P. Gradient normalisation is an option (Alg. 1 "optional"; reference normalises): off = Eq. 7's raw inner product.
- **Exact gradients**: ∇L, Hessian-vector products, G_t = H_val ∂a/∂u + H_P ∂a/∂v, and the pixel gradients of H_t and of the trigger alignment are closed-form; all four match central differences to ~1e-10.
- **Datasets**: `pangMnist35`, `pangMnist71` (2000-image stream, 500 S_val, full test set), `pangCifarShipFrog`, `pangCifarPlaneBird` (1000 / 500 / 1000). Old `mnist`/`cifar` 2-D PCA entries, `loadImagePair2D`, and the `pcaState`/`images` branches of the image panels removed. Image panels handle RGB; the poison panel shows clean / poisoned / δ via the new `poisonSource` frame field, and the canvas draws clean→poisoned segments for it.
- **Trigger frame** also reports `preTriggerAccuracy` and `vanillaTriggerAccuracy` (same kind of trigger fed to the clean counterfactual θ̃_T — Table 1's baseline), so the explainer compares single-step drops instead of asking the user to rerun with λ = 0.
- **Honest size of the effect**: with correct labels and ε = 16/255 a linear victim can't be steered far, so drops are a few to ~10 points (paper's online Table 1 on 10-class CIFAR-10: 3–11 points; the collapses to 10–30% in its Tables 3–6 are the federated setting — see the 2026-09-29 correction below) and vary run to run. Defaults β = 8, B = 25, 5 burn-in epochs, T = 40, λ = 2, raw inner product. No defences (anomaly detection, gradient clipping) are modelled — by request.
- Upload: image features go to `/api/attack` as base64 bytes (CIFAR sample ≈ 100 MB as JSON → ~6 MB). Trace batches rounded to 1e-5, PGD gradients to 4 significant digits.

## Pang 2021 — UI leftovers (2026-09-28)

- Control-panel model heading comes from the optional `AlgorithmModule.modelLabel` (Pang: "Online Logistic Regression"); default still "SVM Model" / "Regression Model".
- Point inspector: optional `AlgorithmModule.pointDetails(ctx)` returns the table (two value columns, coloured ↑/↓ by `worse`), a note and extra readouts; without it the SVM / regression rows render as before. Pang (`pang2021/inspector.ts`): a poison point compares its clean source image with the perturbed one under θ_t, the model that batch was fed to (margin y·z, P(correct), logistic loss, ‖δ‖∞, ‖δ‖₂, source index, ‖∇ₓH_t‖); a training point compares θ₀ with the frame's model and says whether it is in the frame's batch.
- Fixed: the inspector read `attackTrace[currentIteration − 1]` while the canvas draws `attackTrace[currentIteration]`, so a clicked poison point showed the previous frame's data (for Pang a different image).
- Math panel: classification wording comes from the optional `AlgorithmModule.mathPanel` (`MathPanelLabels` in registry.ts); the SVM strings moved verbatim into `SVM_LABELS` in MathPanel.tsx as the default. Pang (`pang2021/panel.ts`): w / b / ‖w‖ of the logistic regression, ℒ_log instead of ℒ_hinge, no support-vector rows, the Eq. 7 alignment as the objective (lower = stronger attack), the paper's Eq. 3/6/7/9 as reference equations.
- Still open: `GuidedTour.tsx` shows Jagielski's regression tour for every non-Biggio algorithm, Pang included.


## Pang 2021 — neural-network victim (2026-09-29)

- **Victim**: `victim` = `mlp` (default) | `logreg`, plus `hiddenUnits` (16–256, default 32), `hiddenLayers` (1–3), `activation` (ReLU | tanh), shown only for the MLP via the new optional `ConfigField.showIf`. `model.ts` is now one feed-forward net over the raw pixels, `sizes = [d, h…, 1]`; zero hidden layers *is* logistic regression, so there is a single code path (on it the new code matches the old closed forms to 1e-16).
- **Exact second order via Pearlmutter's R-operator**: `rop(stats, v)` returns H·v and ∂/∂x̃ [vᵀ∇_θL] from one forward/backward pass, so G_t (Eq. 9), trigger crafting and batch crafting need no finite differences. Checked against central differences (ReLU and tanh, 0–3 hidden layers): ∇_θL, Hv, pixel gradient, G_t (dot and cosine) all ~1e-9 relative.
- **Parameterisation**: hidden units carry 1/√fan-in (NTK-style), like the 1/√d on the pixels, so one β means the same for both victims; He init N(0, 2) for ReLU layers, zero init for logistic regression.
- **Wire format**: raw models are `PackedNet` — parameters as base64 IEEE-754 (float64 for the clean model, float32 on frames; an MLP on CIFAR has ~100k–200k parameters). `unpackNet` decodes once per object (WeakMap). `modelLabel` and `mathPanel` on `AlgorithmModule` may now be functions of the merged config (MLP rows: w_out, b_out, ‖W‖_F).
- **Speed**: S_val's forward/backward pass is computed once per θ_t and shared by G_t, the trigger re-crafting, the alignment readout and the next round; the candidate's test evaluation is reused. Per round (32 units, server CPU): ~0.5 s MNIST, ~1.5 s CIFAR-10; a default CIFAR run end-to-end ~15 s. Burn-in in the browser ~1–2 s.
- **Measured effect** (headless harness, real data, paired against the vanilla trigger on the clean trajectory θ̃_T, Table 1's comparison). Defaults β = 1, λ = 1, raw inner product, 32 units: CIFAR ship/frog accumulated drop 15.8 vs vanilla 6.0 points (+9.8 ± 1.8, 7/8 runs); plane/bird 17.0 vs 8.0; MNIST 3v5 10.2 vs 10.8 (no gain — the trigger alone already costs ~11); MNIST 7v1 ~1 point either way. Logistic regression at β = 1: ≤ 1 point.
- **What didn't work, and why the defaults are what they are**: at β ≥ 2 the poisoned trigger alone drops the MLP 10–25 points and the accumulative batches trip the §4 accuracy monitor within 1–5 rounds, so accumulation adds nothing measurable over the noise. Cosine normalisation (the reference code's choice) was weaker than Eq. 7's raw inner product on every setting tried. A clean trigger (Table 6) did nothing on either victim. Lower β with a looser monitor (β = 0.5, 10 burn-in epochs, Δacc 0.1, λ = 2) also works: +6.4 CIFAR, +3.1 MNIST, 4/4 runs each.
- Not changed: `GuidedTour.tsx` still shows the regression tour for Pang; the in-browser attack path runs on the main thread (~0.5–1.5 s per round with the MLP), so the default server path is the one to use.

## Pang 2021 — what the paper actually reports online (2026-09-29, correction)

- Earlier entries and the explainer compared our drops to "the paper's collapse". That was wrong. §4.1 / Table 1 (online, ResNet-18, 10-class CIFAR-10, SGD lr 0.1 momentum 0.9 wd 1e-4, batch 100, C = 100 PGD steps, α = 2ε/C, BN in train mode, early stop at ~80% accuracy) reports single-step drops (before − after trigger) of: vanilla poisoned trigger ε = 16/255 −2.53; + accumulative phase −3.54; + optimizing P −8.92; + weight momentum (momentum 1.1 during accumulation) −11.09. Clean trigger + accumulative phase −3.95, + weight momentum −8.23.
- The drops to 10–30% (Tables 3, 4, 5, 6 and Fig. 3b, including MNIST 98.27 → 22.49) are §4.2 **federated** learning (Algorithm 2): the attacker submits gradients directly, bounded only by a norm η / "loss scaling", accumulating for 200–1000 steps. This app implements only the online setting.
- So the MLP victim's CIFAR drops here (~16 points vs ~6 for the vanilla trigger, binary task) are in line with, or above, the paper's online numbers. Not modelled from Table 1: SGD momentum and the "weight momentum" trick.

## Pang 2021 — the paper's online optimiser: momentum + weight momentum (2026-09-29)

- **Victim optimiser = reference code's**: `optimStep` is torch.optim.SGD (d = g + λ_wd θ; v ← μv + d, v = d on a fresh optimiser; θ ← θ − βv), checked bit-exact against a hand-written loop. Burn-in: μ = 0.9, weight decay 1e-4 (`train_cifar.py`). Online phase: a fresh optimiser, μ = 0.9, no weight decay (`online_accu_train.py`); the attacked model and the clean counterfactual θ̃ each keep their own buffer, and **the trigger step carries the buffer** — θ_{T+1} = θ_T − β(μv_T + ∇L(P)). The Eq. 6 forecast on the trigger frame is for that actual step; frames carry `velocityNorm` = ‖v_t‖.
- **Weight momentum** (`weightMomentum`, §4.1 / Table 1): μ = 1.1 for accumulative updates only (not in the reference script; implemented from the paper's text). The trigger and the clean fallback on early stop use the victim's μ. It assumes the attacker can set the victim's momentum — said in the tooltip.
- New config: `momentum` (model, default 0.9), `weightMomentum` (attack, default on), PGD steps up to C = 100 (paper's value; default stays 10 — C = 30 was no better). β slider now 0.01–20; default β 0.3.
- **Tuning** (MLP 32, λ ∈ {0,1}, 4 reps, paired against the vanilla trigger on θ̃_T): with μ = 0.9 the vanilla poisoned trigger is ≤ 1 point (paper 2.5); without weight momentum accumulation adds 1–3 points on CIFAR; with it, CIFAR β 0.05 → 1.9, 0.1 → 2.8, 0.2 → 6.5–8.1, **0.3 → 8.0–11.4**, 0.5 → 8.7–9.8 (vanilla grows to 2.5–3.7). MNIST 2–4.5 points. λ = 0 vs 1 are close with weight momentum on (the buffer itself accumulates); λ = 1 early-stops sooner (~5–7 rounds vs ~16).
- **At the defaults** (β 0.3, μ 0.9, weight momentum, λ 1, T 40, C 10, B 25, 5 burn-in epochs): CIFAR ship/frog accumulated 6.2 vs vanilla 0.6 points (8/8 runs), plane/bird 5.8 vs 0.7 (4/4), MNIST 3v5 3.7 vs 0.8 (8/8), 7v1 1.4 vs −0.1 (4/4); accuracy before the trigger sags ~5 points during accumulation (paper: 83.4 → 80.2). Logistic regression at the same settings: 0.7 vs 0.1. ~1.7 s/round on CIFAR (server CPU, runs in parallel).
- Compared with the paper's Table 1 (ε 16/255, 10-class CIFAR-10, ResNet-18): 11.09 vs 2.53 with all tricks. Same shape, smaller absolute numbers on a two-class MLP. The earlier plain-SGD defaults (β 1, no momentum) gave bigger raw drops (~16 vs ~6) because a single un-damped step is larger — but that is not the paper's optimiser.

## Pang 2021 — federated setting, Algorithm 2 (2026-09-29)

- **What**: `setting` = `online` | `federated` (attack panel; online-only / federated-only fields via `showIf`). `pang2021/federated.ts`: the poisoners set the whole aggregate each round (recovered offset, Eq. 14): U = w·∇L(S_t) + λ∇_θ a(σ∇L(S_T), ∇L(S_val)), a = cosine (reference `feder_accu_train.py`) or Eq. 10's raw inner product; w = 0 is the reference code (λ-term only), w = 1 Algorithm 2's H_t; σ = −1 is the reversed trigger P = −s∇L(S_T) (Eq. 13, `adapt_tensor_reverse`). Server: SGD momentum 0.9, no weight decay, buffer carried into the trigger, `clip_grad_norm_` ℓ2/ℓ∞ clipping on every aggregate (Tables 3–5). Baselines on the honest trajectory θ̃_T (same stream, same clipping): the same trigger, and Table 3's direct poisoner −s_d∇L(S_val) (`craft_direct_NEW`) → `directAttackAccuracy`. Monitor off by default (the reference's threshold 1.202 never fires); when on, the white-box attacker checks each candidate round and sends the honest aggregate instead. ≤ ~50 frames per run whatever T is.
- **Exact gradients**: U = ∇Φ with Φ = w·L(S_t) + λ·a(σ∇L(S_T), ∇L(S_val)); central differences agree to ≤ 4e-9 (ReLU/tanh, 0–3 hidden layers, both σ, w ∈ {0,1}, cosine and inner product). `clipUpdate` matches clip_grad_norm_ (coefficient c/(‖g‖+1e-6), capped at 1).
- **New optional TraceFrame fields**: `updateNorm`, `clipFactor`, `directAttackAccuracy`. Registry: `explainerSteps` may be a function of config (like `mathPanel`); `ExplainerOverlay` resolves it. Federated explainer cards in `pang2021/federatedExplainer.ts`; math panel shows Eq. 10–13.
- **Tuning** (headless, real data, MLP 32, 300 rounds unless noted, monitor off, paired per seed against the same trigger on θ̃_T): the reference's cosine objective turns the trigger/val alignment to ≈ −0.8 within 100 rounds but the clean trigger then moves accuracy ≤ 1 point — one step of β‖∇L(S_T)‖ is too small, however well aimed. λ ≥ 1 (cosine or inner) instead drags the model to chance *before* the trigger (47–55%), the opposite of Fig. 3b. T = 1000 did not help. Reversed trigger: at best +3 points on MNIST (λ 0.2, s 1) with pre-trigger accuracy already 58%. What works best: Eq. 10's raw inner product (rewards trigger-gradient magnitude) with a larger server step (β 1–3).
- **At the defaults** (inner product, λ 0.3, T 300, clean trigger, server β 1, w = 0, no clipping), 8 seeds each: MNIST 3v5 accumulated drop 3.0 vs 0.0 without accumulation (+2.9 ± 1.4, 7/8), before-trigger accuracy 95.0 → 84.9; CIFAR ship/frog 0.6 vs −0.2 (+0.8 ± 0.2, 7/8), 89.2 → 85.6. ~0.2 s/round MNIST, ~0.5 s/round CIFAR (server CPU, many runs in parallel).
- **Clipping (Table 3 analogue, 3 seeds each)**: direct poisoner unclipped s_d = 50: MNIST 25.3, CIFAR 40.4 points; under ℓ2 ≤ 1: 3.8 / 3.7 — clipping neutralises it, as in the paper. Accumulated trigger under clipping: MNIST ℓ2 ≤ 10 5.6 (direct s_d 10: 9.2), ℓ∞ ≤ 1 4.9 (6.3), ℓ∞ ≤ 0.1 2.6 (1.2), ℓ2 ≤ 0.1 0.5 (0.6); CIFAR ℓ∞ ≤ 1 3.4 (7.9), ℓ∞ ≤ 0.1 0.4 (0.5), ℓ2 ≤ 0.1 0.0 (0.4). Accumulation beats the direct poisoner only at MNIST ℓ∞ ≤ 0.1.
- **Against the paper** (Table 3, 10-class CIFAR-10, ResNet-18 with BN, T = 1000, λ 0.01–0.08): clean trigger after accumulation → 11–34% from ~83%; direct poisoner 65/41/10% at scale 10/20/50. The collapse does **not** reproduce here. Differences: two-class task (chance 50%), a 32-unit MLP without batch norm (the reference also runs BN in train mode through every crafting forward pass), CPU-bounded T, S_val from the training split. The honest statement: the mechanism (alignment driven negative, trigger primed, clipping bypassed by small updates) is implemented exactly and measurable, but the effect on this victim is a few points.
