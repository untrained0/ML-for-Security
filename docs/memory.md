# Memory — ML Security Attack Visualizer

> **Last Updated**: 2025-07-30 14:35 IST

---

## Current Status: 🟢 Phase 1 — Foundation Complete, Running

### ✅ Completed

| Item | File(s) | Date |
|---|---|---|
| Research references & Biggio 2012 paper | — | 2025-07-30 |
| Analyzed PoisonRegression codebase | `poisoning/poison.py`, `gd_poisoners.py` | 2025-07-30 |
| Created all 6 docs (PRD, Architecture, Rules, Phases, Design, Memory) | `docs/*.md` | 2025-07-30 |
| Updated Architecture & Rules for Next.js | `docs/architecture.md`, `docs/rules.md` | 2025-07-30 |
| Initialized Next.js 16 project | `ml-security-viz/` | 2025-07-30 |
| Installed deps: zustand, d3, recharts, katex | `package.json` | 2025-07-30 |
| Created engine: linalg.js | `src/engine/linalg.js` | 2025-07-30 |
| Created engine: kernels.js (+ fixed symmetric matrix bug) | `src/engine/kernels.js` | 2025-07-30 |
| Created engine: svm.js (SMO solver) | `src/engine/svm.js` | 2025-07-30 |
| Created engine: datasets.js | `src/engine/datasets.js` | 2025-07-30 |
| Created engine: poisoning.js | `src/engine/poisoning.js` | 2025-07-30 |
| Created Zustand store | `src/store/useStore.js` | 2025-07-30 |
| Created root layout + globals.css | `src/app/layout.js`, `src/app/globals.css` | 2025-07-30 |
| Created main page (orchestrator) | `src/app/page.js`, `src/app/page.module.css` | 2025-07-30 |
| Created AppHeader component | `src/components/AppHeader.js` | 2025-07-30 |
| Created ControlPanel component | `src/components/ControlPanel.js` | 2025-07-30 |
| Created Canvas component (SVG scatter + marching squares) | `src/components/Canvas.js` | 2025-07-30 |
| Created PlaybackBar component | `src/components/PlaybackBar.js` | 2025-07-30 |
| Created MathPanel component | `src/components/MathPanel.js` | 2025-07-30 |
| Created Timeline component (Recharts) | `src/components/Timeline.js` | 2025-07-30 |
| Dev server running, compiling clean | `npm run dev` → localhost:3000 | 2025-07-30 |
| Added PCA and MNIST generator (Digits 1 v 7) | `src/engine/pca.js`, `src/engine/mnist.js` | 2025-07-30 |
| Built TestImagePanel for clean vs poisoned diffs | `src/components/TestImagePanel.tsx` | 2025-07-30 |
| Added sidebar toggle controls for full-screen canvas | `src/components/PlaybackBar.tsx`, `page.tsx` | 2025-07-30 |
| Converted UI to Tailwind CSS v4 | All `.tsx` components, `globals.css` | 2025-07-30 |
| Offloaded Poisoning Engine to Web Worker (Phase 2.1) | `src/engine/worker.ts`, `page.tsx`, `useStore.ts` | 2025-07-30 |
| Integrated KaTeX rendering for math equations | `MathPanel.tsx`, `MathEq.tsx` | 2025-07-30 |
| Built Δw convergence tracker | `worker.ts`, `Timeline.tsx` | 2025-07-30 |
| Tracked SV changes and added SV flashing animation | `worker.ts`, `Canvas.tsx` | 2025-07-30 |
| Built Comparison Metrics Table (Precision/Recall/F1) | `ComparisonTable.tsx`, `metrics.ts` | 2025-07-30 |
| Rendered objective function heatmap and gradient vector field | `worker.ts`, `Canvas.tsx`, `ControlPanel.tsx` | 2025-07-30 |
| Rendered poison point descent trajectories | `Canvas.tsx` | 2025-07-30 |
| Added JSON and SVG Export buttons | `PlaybackBar.tsx` | 2025-07-30 |
| Implemented CSV Dataset Upload parser | `ControlPanel.tsx` | 2025-07-30 |
| Created Onboarding Tutorial overlay | `TutorialModal.tsx`, `AppHeader.tsx`, `useStore.ts` | 2025-07-30 |
| Updated comprehensive documentation | `README.md` | 2025-07-30 |
| Fixed infinite pan/zoom bug and canvas rendering | `Canvas.tsx` | 2026-07-30 |
| Built scalable multi-algorithm registry framework | `src/engine/algorithms/registry.ts`, `index.ts` | 2026-07-30 |
| Wrapped existing Biggio 2012 SVM logic in new framework | `src/engine/algorithms/biggio2012.ts` | 2026-07-30 |
| Implemented Jagielski 2018 Regression Poisoning (KKT) | `src/engine/algorithms/jagielski2018.ts` | 2026-07-30 |
| Added regression datasets (Linear, Quadratic, Sine) | `datasets.ts` | 2026-07-30 |
| Added 1D approximations of 3 real-world datasets (Warfarin, Lending Club, House Pricing) | `datasets.ts` | 2026-07-30 |
| Implemented LASSO (Coordinate Descent) and OLS | `jagielski2018.ts` | 2026-07-30 |
| Implemented Active-Set KKT gradients for LASSO | `jagielski2018.ts` | 2026-07-30 |
| Fixed regression Timeline / PlaybackBar undefined crashes | `Timeline.tsx`, `PlaybackBar.tsx` | 2026-07-30 |
| UI adaptive rendering for Regression (lines, residuals) vs Classification (contours) | `Canvas.tsx`, `MathPanel.tsx`, `ControlPanel.tsx`, `PointInspector.tsx` | 2026-07-30 |
| PointInspector math metrics for individual points | `PointInspector.tsx` | 2026-07-30 |
| GuidedTour Helper Box for algorithmic walk-through | `GuidedTour.tsx` | 2026-07-30 |
| Concept Explainer System (clickable math popovers) | `ExplainerCard.tsx`, `explainers.tsx`, `ExplainerIcon.tsx` | 2026-07-30 |

### 🐛 Bugs Fixed

| Bug | Root Cause | Fix |
|---|---|---|
| `TypeError: Cannot set properties of undefined` in `computeKernelMatrix` | Symmetric fill `K[j][i]=val` accessed uninitialized row | Initialize ALL rows in a first pass |
| `TypeError: Cannot destructure property 'mean'` | `mnistData` raw data wasn't saved in `setDataset` | Pass `raw` MNIST data into `setDataset` in `page.tsx` |
| Test Image Predictor showing "Not attacked" | `poisonedRawModel` was not pushed to trace history | Return `rawModel` from `poisonIteration`, push to trace, and strip `kernelFn` during worker IPC |
| Worker failing to build `computeGradient` | Missing import | Add `computeGradient` import in `worker.ts` |
| `TypeError: model.supportIndices is not iterable` in PointInspector | Used display model instead of raw model | PointInspector uses `cleanRawModel` and `poisonedRawModel` which retain functional math fields |
| `The final argument passed to useEffect changed size between renders` | React dependency array mutability on zoom | Extracted variables or stabilized array references in dependencies |

### 📋 Up Next (Phase 6: Multi-Attack Framework - Future)

- Label-flip attack (6.1)
- Evasion attack (FGSM-style) (6.2)
- Backdoor attack (6.3)
- Defense visualizations (TRIM, RONI) (6.4)
- ~~Regression poisoning (6.5)~~ (Done: Jagielski 2018)
- Paper replication dashboard (6.6) (Done: registry framework established)

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | Next.js 16 (App Router) | User requested modern framework |
| State | Zustand | Lightweight, no boilerplate |
| Charts | Recharts | Native React, built on D3 |
| Canvas rendering | Raw SVG + marching squares | No D3 imperative DOM needed; React-friendly |
| Compute offload | setTimeout (Phase 1) → Web Worker (Phase 2) | Quick to ship; Web Worker comes next |

---

## File Manifest

```
docs/
├── prd.md           ✅ 
├── architecture.md  ✅ (updated for Next.js)
├── rules.md         ✅ (updated for Next.js)
├── phases.md        ✅
├── design.md        ✅
└── memory.md        ✅ (this file)

ml-security-viz/src/
├── app/
│   ├── globals.css      ✅
│   ├── layout.tsx       ✅
│   └── page.tsx         ✅
├── components/
│   ├── AppHeader.tsx     ✅
│   ├── Canvas.tsx        ✅
│   ├── ControlPanel.tsx  ✅
│   ├── MathPanel.tsx     ✅
│   ├── PlaybackBar.tsx   ✅
│   ├── Timeline.tsx      ✅
│   ├── PointInspector.tsx✅
│   ├── GuidedTour.tsx    ✅
│   ├── ExplainerCard.tsx ✅
│   └── ExplainerIcon.tsx ✅
├── engine/
│   ├── architectures/    ✅ (Decoupled plugin framework)
│   │   ├── registry.ts   ✅
│   │   ├── index.ts      ✅
│   │   ├── biggio2012/   ✅ (SVM classification)
│   │   └── jagielski2018/✅ (Ridge/LASSO regression)
│   ├── data/
│   │   ├── datasets.ts   ✅
│   │   ├── raw/          ✅ (JSON datasets)
│   │   └── loaders/      ✅
│   ├── ui/
│   │   └── explainers.tsx✅
│   ├── linalg.ts         ✅
│   └── worker.ts         ✅
└── store/
    └── useStore.ts       ✅
```
