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
