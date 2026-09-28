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
npm run dev     # http://localhost:3000  (predev fetches datasets if missing)
npm run build   # type-check + production build — the only automated gate
npm run lint
npm run fetch-data   # download the original datasets into data/ (`npm run fetch-data -- --force` refreshes)
npm run fetch-data -- --probe                # test every source host (and the proxy) without downloading
npm run fetch-data -- --only mnist,regression   # subset: mnist | regression | cifar10
npm run build:cuda   # build the optional CUDA addon native/build/mlsv_cuda.node (`-- --force` rebuilds)
npm run bench:compute  # CPU vs CUDA timings behind the GPU size thresholds
```

Behind a proxy, export `HTTPS_PROXY` / `HTTP_PROXY` (optionally `http://user:pass@host:port`, URL-encoded; `NO_PROXY` honoured) — the fetch script tunnels through it itself, because Node's built-in `fetch` ignores those variables. A TLS-inspecting proxy additionally needs `NODE_EXTRA_CA_CERTS`.

There is **no test suite**. `npm run build` is the verification step; it must pass with zero errors. Manual verification: pick each algorithm from the dropdown → generate dataset → train → launch attack → step through iterations and confirm the canvas, MathPanel, and Timeline all update.

`scripts/fetch_datasets.mjs` downloads the complete original datasets into `data/` (git-ignored, server-side only; `DATA_DIR` overrides): MNIST (IDX, 60k/10k), CIFAR-10 (binary batches, 50k/10k) and the authors' preprocessed Warfarin/Loan (all 168k records)/House CSVs for Jagielski 2018. It runs automatically before `dev`/`build`, skips files already present, and never fails the build; `scripts/build_cuda.mjs` runs right after it the same way (skips with a message when there is no `nvcc`). The browser never downloads a whole dataset: route handlers in `src/app/api/datasets/{images,regression}` sample per request from `src/server/datasets.ts`, which caches each dataset in memory. The app therefore needs a Node server (`next start`), not a static export. Clean models are trained in the browser; attacks run on the server by default (see "Where attacks run" below). `src/engine/data/raw/*.json` is obsolete (mislabeled stand-ins) and unused. The `replace_*.js` files at the app root are one-off Tailwind-migration codemods, not part of the build.

## Architecture

### Algorithm registry — the central pattern

Every attack paper is a self-contained plugin under `src/engine/architectures/<paper_key>/` implementing the `AlgorithmModule` interface in [registry.ts](ml-security-viz/src/engine/architectures/registry.ts). The UI is fully generic: `ControlPanel` renders sliders/selects straight from `configSchema`, `Canvas` (and its 3-D view `Canvas3D`, toggled by `canvasView`) branches only on `modelType` (`'classification'` vs `'regression'`), and everything downstream reads `TraceFrame` objects.

A module supplies `trainClean()`, `runAttack()`, `predict()`, a `datasets: string[]` whitelist, `configSchema` + `defaultConfig`, and optional `explainerSteps`. It calls `registerAlgorithm(mod)` at module scope; registration only happens because [architectures/index.ts](ml-security-viz/src/engine/architectures/index.ts) side-effect-imports every folder. **Adding an algorithm means a new folder plus one import line — never edit another algorithm's folder.**

Current modules: `biggio2012` (SVM poisoning: LIBSVM-style SMO, closed-form Eq. 10 gradient, sequential single-point attacks), `jagielski2018` (OLS/ridge/LASSO/elastic-net regression poisoning, Algorithm 1 with the joint (x_c, y_c) KKT gradient), `pang2021` (accumulative poisoning on online logistic regression). Analytic gradients in the first two were verified against finite differences; keep it that way when changing them.

### Where attacks run

The "Run Attack On" control (`computeTarget` in the store: `auto` | `server` | `browser`, component `ComputeTarget.tsx`) picks one of three paths in `runAttack` in [page.tsx](ml-security-viz/src/app/page.tsx):

- **Server** (`auto` default): [lib/serverAttack.ts](ml-security-viz/src/lib/serverAttack.ts) gzips the dataset (minus images/PCA view) and the clean model (minus `rawModel.K/X/y/kernelFn`) to `POST /api/attack`; [server/attack.ts](ml-security-viz/src/server/attack.ts) calls the module's own `runAttack` and streams gzip NDJSON (`meta`, one `frame` per TraceFrame, `done`/`error`), flushed per frame. A disconnect aborts the attack via the `signal` argument. `auto` falls back to the browser only on `ServerUnavailableError` (unreachable/404/5xx before the first frame); a failing attack is reported, not retried.
- **Browser, `biggio2012`**: the Web Worker at [engine/worker.ts](ml-security-viz/src/engine/worker.ts) (same `biggioAttack` generator). The worker also owns the Fig. 1 objective heatmap (`COMPUTE_HEATMAP`, 2-D datasets only), which always runs in the browser.
- **Browser, everything else**: `alg.runAttack(...)` on the main thread, yielding with `setTimeout(step, 0)`.

Every path funnels frames into `useStore.getState().appendAttackTraceFrame(frame)`. Biggio frames carry a compact SV-only model without `kernelFn` (functions cross neither `postMessage` nor JSON); `page.tsx` re-attaches one built from `kernelSpec(config, …)`. New modules must accept `signal` and stop when it is aborted, and keep frames JSON-safe (CONTRIBUTING.md).

### Compute backend (CPU / CUDA)

Heavy dense linear algebra goes through `compute()` from [engine/compute](ml-security-viz/src/engine/compute/backend.ts): `gram` (XᵀX), `inverse`, `rankOneInverse` (Sherman–Morrison-updated inverse) and `denseMatrix` (resident A·v / Aᵀ·v). The browser always uses the TypeScript CPU backend. The server installs a backend once per process in [server/compute](ml-security-viz/src/server/compute/index.ts) (`MLSV_COMPUTE=auto|cpu|cuda`, default `auto`; `GET /api/compute` reports it): the CUDA backend wraps the FP64 cuBLAS/cuSOLVER addon in `native/cuda/mlsv_cuda.cu`, loaded with `process.dlopen` from `native/build/` (git-ignored), and sends an operation to the GPU only above measured size thresholds (`CUDA_THRESHOLDS` in `server/compute/cuda.ts`). Objects from `rankOneInverse`/`denseMatrix` hold GPU memory — `dispose()` them (Jagielski does so in a `finally`). Today Jagielski's `RegressionFit` and outer objective use it; Biggio's per-step work is too small for the GPU (its gradient was instead reordered to O(n_s·d)), and Pang's 2-D model gains nothing.

### Guided explainer

`AlgorithmModule.explainerSteps` drives the walkthrough panel. A step carries the symbolic `equation` plus optional `liveEquation(frame, config)`, `readouts(frame, config)`, and `visual` — all evaluated against the frame the user is scrubbed to, so the maths is bound to the running model rather than illustrating it. `visual` names a diagram in [ExplainerVisual.tsx](ml-security-viz/src/components/ExplainerVisual.tsx), which draws real vectors out of the trace. Anything a step needs to draw must therefore be emitted on the `TraceFrame` (see `gradVal`, `gradTrigger`, `gradAccum`).

### TraceFrame contract

`TraceFrame` is the single wire format between engine and UI. Required on every frame: `iteration`, `poisonX`, `poisonY`, `poisonedModel` (display-friendly), `poisonedRawModel` (feeds `predict`), `objectiveValue`, `gradients`, `gradientNorms`, `deltaW`. Everything else — classification accuracy, regression MSE, SV churn, the Pang online-learning fields (`phase`, `batchIndex`, `secretAccuracy`, …) — is optional. **New fields must be optional (`?`)** so other algorithms keep compiling.

### State

Zustand store at [useStore.ts](ml-security-viz/src/store/useStore.ts) is the single source of truth; components read it via `useStore()`, never prop-drill, and never store derived values. `setActiveAlgorithm` deliberately wipes dataset/model/trace — switching algorithms is a full reset.

Config resolution has three layers, merged in `getMergedConfig()` in `page.tsx`: `alg.defaultConfig` → top-level legacy store fields (`kernelType`, `svmC`, `attackEta`, … kept for `biggio2012` back-compat) → `algorithmConfig` overrides written by `ControlPanel`. When adding a config key, put it in `configSchema` + `defaultConfig` only; don't add new top-level store fields.

### Datasets

Entries either have `fn(n)` (unsplit `{X, y}`, split by `splitDataset` with optional `split` ratios) or `load(n)` (a pre-split dataset, possibly async — the real image and regression benchmarks are sampled from `/api/datasets/*`). Dataset-level optional fields: `view` (PCA window for drawing high-dimensional data; canvas uses `project`/`lift` from `engine/data/view.ts` and evaluates models via `alg.predict` on the lifted points), `bounds` (feasible box for poisoning points), `displayRange`, `imageShape`, `classNames`, `onehotGroups`. Models always train on the full feature space; never on the view.


`DATASETS` in [datasets.ts](ml-security-viz/src/engine/data/datasets.ts) maps key → `{ name, fn | load, icon, desc }` plus `isRegression`. An algorithm only sees datasets listed in its `datasets` array. Image data comes in two shapes, so never assume `X` is 2-D:

- Biggio's `mnist71` / `mnist98` / `mnist40`: `X` is the raw 784-pixel vector in [0,1], with `imageShape`, `classNames` and a `view` for drawing.
- Pang's `mnist` / `cifar`: `X` (= `X2D`) is a 2-D PCA of real MNIST / CIFAR-10 images; each split also carries 0–255 `images` and `labels`, and the dataset a `pcaState` whose inverse maps a canvas point back to pixels.

### Styling

Tailwind CSS v4, **light + dark**, configured entirely in CSS. [globals.css](ml-security-viz/src/app/globals.css) declares raw tokens twice — `:root` (light) and `.dark` (dark) — and `@theme inline` maps them onto utilities that re-resolve at runtime. Switching themes is just a class on `<html>`; there are no `dark:` variants on components and there shouldn't be. Use semantic tokens (`bg-card`, `text-muted-foreground`, `border-border-subtle`, `bg-attack`, `text-clean`, `fill-data-poison`) and the component classes `glass-panel`, `eyebrow`, `data-value`. Never hardcode a color — `bg-white`, `text-gray-400`, `bg-[#059669]`, `stroke="rgba(255,255,255,0.2)"`, or a hex literal in a Recharts prop are all theme bugs. Color is semantic: red = attack, emerald = clean, amber = caution, blue = info. Full system in `docs/design.md`.

Three things that bite:

- **Base CSS must stay inside `@layer base`.** Unlayered CSS outranks every layered rule regardless of specificity, so an unlayered `button { background: none }` silently beats `bg-primary` and `svg { display: block }` beats `hidden`. This was a real bug — the reset is now wrapped in `@layer base`, and adding rules outside it will re-break utilities.
- **Font tokens must point at the `next/font` CSS variables** (`var(--font-inter)`), not the literal family name. Naming `"Inter"` directly falls back to the system font, because the loader registers a hashed family name. Same in SVG: `className="font-sans"`, never `fontFamily="Inter"`.
- **Theme state.** Resolved before first paint by an inline script in [layout.tsx](ml-security-viz/src/app/layout.tsx) (saved choice → system preference → dark), mirrored into the store, owned by [ThemeToggle.tsx](ml-security-viz/src/components/ThemeToggle.tsx). Code that can't use CSS — Recharts style objects, JS-computed SVG attributes — reads [`useThemeTokens()`](ml-security-viz/src/hooks/useThemeTokens.ts); it's a last resort, not a convenience.

## Conventions and gotchas

- `strict: false` in tsconfig, `@/*` → `src/*`. Engine code is deliberately dependency-free — matrix ops live in `engine/linalg.ts` and `engine/compute/`; do not pull in math.js or TensorFlow.js. GPU code lives only in `native/` and `src/server/compute/`, never imported by the engine directly.
- All interactive components need `'use client'` (App Router).
- `border-border`, not `border-border-default`.
- Formulas must match paper notation, with equation references in comments/tooltips (e.g. "Eq. 7 from Biggio et al. 2012"). Guard matrix ops against NaN/Infinity.
- **`docs/rules.md` is stale** — it predates the Tailwind migration and still says "Next.js 15", "CSS Modules", "do not use TailwindCSS", and `.js` engine files. The live conventions are Tailwind v4 + TypeScript, as described above and in `CONTRIBUTING.md` / `docs/design.md`. Its guidance on math accuracy and error handling still holds.
- `docs/memory.md` is a dated changelog of significant work; `docs/phases.md` tracks the roadmap. Update them when landing something substantial.
