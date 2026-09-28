# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

- `ml-security-viz/` — the Next.js app. **All npm commands run from here**, not the repo root.
- `docs/` — PRD, architecture, design system, phases, and a running project memory log.
- `CONTRIBUTING.md` — step-by-step recipe for adding a new attack algorithm or dataset.

## Commands

```bash
cd ml-security-viz
npm install
npm run dev     # http://localhost:3000
npm run build   # type-check + production build — the only automated gate
npm run lint
```

There is **no test suite**. `npm run build` is the verification step; it must pass with zero errors. Manual verification: pick each algorithm from the dropdown → generate dataset → train → launch attack → step through iterations and confirm the canvas, MathPanel, and Timeline all update.

`scripts/fetch_datasets.js` regenerates `src/engine/data/raw/*.json` from remote CSVs (run manually, rarely). The `replace_*.js` files at the app root are one-off Tailwind-migration codemods, not part of the build.

## Architecture

### Algorithm registry — the central pattern

Every attack paper is a self-contained plugin under `src/engine/architectures/<paper_key>/` implementing the `AlgorithmModule` interface in [registry.ts](ml-security-viz/src/engine/architectures/registry.ts). The UI is fully generic: `ControlPanel` renders sliders/selects straight from `configSchema`, `Canvas` branches only on `modelType` (`'classification'` vs `'regression'`), and everything downstream reads `TraceFrame` objects.

A module supplies `trainClean()`, `runAttack()`, `predict()`, a `datasets: string[]` whitelist, `configSchema` + `defaultConfig`, and optional `explainerSteps`. It calls `registerAlgorithm(mod)` at module scope; registration only happens because [architectures/index.ts](ml-security-viz/src/engine/architectures/index.ts) side-effect-imports every folder. **Adding an algorithm means a new folder plus one import line — never edit another algorithm's folder.**

Current modules: `biggio2012` (SVM poisoning, SMO solver + kernels), `jagielski2018` (ridge/LASSO/OLS regression poisoning via KKT implicit differentiation), `pang2021` (accumulative poisoning on online logistic regression).

### Two execution paths for attacks

[page.tsx](ml-security-viz/src/app/page.tsx) is the orchestrator, and it special-cases `biggio2012`:

- **`biggio2012`** goes through the Web Worker at [engine/worker.ts](ml-security-viz/src/engine/worker.ts), which imports Biggio's code directly and is *not* generic. Because `postMessage` cannot serialize functions, the worker strips `kernelFn` from the model and `page.tsx` re-attaches a freshly built kernel on every `PROGRESS` message. The worker also owns heatmap/quiver computation (`COMPUTE_HEATMAP`).
- **Every other algorithm** runs `alg.runAttack(...)` on the main thread, yielding between iterations with `setTimeout(step, 0)` so React can paint. Follow that pattern in new modules.

Both paths funnel frames into `useStore.getState().appendAttackTraceFrame(frame)`.

### Guided explainer

`AlgorithmModule.explainerSteps` drives the walkthrough panel. A step carries the symbolic `equation` plus optional `liveEquation(frame, config)`, `readouts(frame, config)`, and `visual` — all evaluated against the frame the user is scrubbed to, so the maths is bound to the running model rather than illustrating it. `visual` names a diagram in [ExplainerVisual.tsx](ml-security-viz/src/components/ExplainerVisual.tsx), which draws real vectors out of the trace. Anything a step needs to draw must therefore be emitted on the `TraceFrame` (see `gradVal`, `gradTrigger`, `gradAccum`).

### TraceFrame contract

`TraceFrame` is the single wire format between engine and UI. Required on every frame: `iteration`, `poisonX`, `poisonY`, `poisonedModel` (display-friendly), `poisonedRawModel` (feeds `predict`), `objectiveValue`, `gradients`, `gradientNorms`, `deltaW`. Everything else — classification accuracy, regression MSE, SV churn, the Pang online-learning fields (`phase`, `batchIndex`, `secretAccuracy`, …) — is optional. **New fields must be optional (`?`)** so other algorithms keep compiling.

### State

Zustand store at [useStore.ts](ml-security-viz/src/store/useStore.ts) is the single source of truth; components read it via `useStore()`, never prop-drill, and never store derived values. `setActiveAlgorithm` deliberately wipes dataset/model/trace — switching algorithms is a full reset.

Config resolution has three layers, merged in `getMergedConfig()` in `page.tsx`: `alg.defaultConfig` → top-level legacy store fields (`kernelType`, `svmC`, `attackEta`, … kept for `biggio2012` back-compat) → `algorithmConfig` overrides written by `ControlPanel`. When adding a config key, put it in `configSchema` + `defaultConfig` only; don't add new top-level store fields.

### Datasets

`DATASETS` in [datasets.ts](ml-security-viz/src/engine/data/datasets.ts) maps key → `{ name, fn, icon, desc }` plus flags. Synthetic 2D generators return `{X, y}` and go through `splitDataset` (60/20/20 train/valid/test). Image datasets are flagged `isMNIST` / `isCIFAR`, dynamically imported from `data/loaders/`, and use `splitDatasetWithImages` — those splits carry **both** `X` (higher-dim PCA features the model trains on) and `X2D` (the 2D projection the canvas draws), so code touching image datasets must not assume `X` is 2D. `isRegression` flags regression datasets. An algorithm only sees datasets listed in its `datasets` array.

### Styling

Tailwind CSS v4, **light + dark**, configured entirely in CSS. [globals.css](ml-security-viz/src/app/globals.css) declares raw tokens twice — `:root` (light) and `.dark` (dark) — and `@theme inline` maps them onto utilities that re-resolve at runtime. Switching themes is just a class on `<html>`; there are no `dark:` variants on components and there shouldn't be. Use semantic tokens (`bg-card`, `text-muted-foreground`, `border-border-subtle`, `bg-attack`, `text-clean`, `fill-data-poison`) and the component classes `glass-panel`, `eyebrow`, `data-value`. Never hardcode a color — `bg-white`, `text-gray-400`, `bg-[#059669]`, `stroke="rgba(255,255,255,0.2)"`, or a hex literal in a Recharts prop are all theme bugs. Color is semantic: red = attack, emerald = clean, amber = caution, blue = info. Full system in `docs/design.md`.

Three things that bite:

- **Base CSS must stay inside `@layer base`.** Unlayered CSS outranks every layered rule regardless of specificity, so an unlayered `button { background: none }` silently beats `bg-primary` and `svg { display: block }` beats `hidden`. This was a real bug — the reset is now wrapped in `@layer base`, and adding rules outside it will re-break utilities.
- **Font tokens must point at the `next/font` CSS variables** (`var(--font-inter)`), not the literal family name. Naming `"Inter"` directly falls back to the system font, because the loader registers a hashed family name. Same in SVG: `className="font-sans"`, never `fontFamily="Inter"`.
- **Theme state.** Resolved before first paint by an inline script in [layout.tsx](ml-security-viz/src/app/layout.tsx) (saved choice → system preference → dark), mirrored into the store, owned by [ThemeToggle.tsx](ml-security-viz/src/components/ThemeToggle.tsx). Code that can't use CSS — Recharts style objects, JS-computed SVG attributes — reads [`useThemeTokens()`](ml-security-viz/src/hooks/useThemeTokens.ts); it's a last resort, not a convenience.

## Conventions and gotchas

- `strict: false` in tsconfig, `@/*` → `src/*`. Engine code is deliberately dependency-free — matrix ops live in `engine/linalg.ts`; do not pull in math.js or TensorFlow.js.
- All interactive components need `'use client'` (App Router).
- `border-border`, not `border-border-default`.
- Formulas must match paper notation, with equation references in comments/tooltips (e.g. "Eq. 7 from Biggio et al. 2012"). Guard matrix ops against NaN/Infinity.
- **`docs/rules.md` is stale** — it predates the Tailwind migration and still says "Next.js 15", "CSS Modules", "do not use TailwindCSS", and `.js` engine files. The live conventions are Tailwind v4 + TypeScript, as described above and in `CONTRIBUTING.md` / `docs/design.md`. Its guidance on math accuracy and error handling still holds.
- `docs/memory.md` is a dated changelog of significant work; `docs/phases.md` tracks the roadmap. Update them when landing something substantial.
