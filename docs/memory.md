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
