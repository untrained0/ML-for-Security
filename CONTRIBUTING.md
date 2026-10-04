# Contributing to ML Security Attack Visualizer

Welcome! This guide will help you add new attack algorithms, datasets, or UI features with minimal merge conflicts.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd ml-security-viz
npm install

# 2. Start dev server
npm run dev    # → http://localhost:3000

# 3. Verify your changes build
npm run build  # Must pass with zero errors
```

---

## Project Overview

This is an interactive visualization tool for adversarial machine learning attacks. Users can:
- Select a dataset and an attack algorithm
- Configure model and attack parameters
- Run the attack step-by-step and watch the model degrade
- Inspect mathematical details at each iteration

**Tech Stack**: Next.js 16, React 19, Tailwind CSS v4, Zustand, Recharts, KaTeX

---

## Architecture At a Glance

```
src/
├── app/           # Next.js pages + global CSS
├── components/    # React UI components
├── engine/        # All computation (algorithms, data, math)
│   ├── architectures/   # ★ ATTACK MODULES — one folder per paper
│   │   ├── registry.ts  # AlgorithmModule / ViewModule interfaces + registries
│   │   ├── categories.ts # Attack categories = the header tabs (poisoning, evasion)
│   │   ├── index.ts     # Barrel file — import all modules here
│   │   ├── biggio2012/  # SVM Poisoning
│   │   ├── jagielski2018/ # Regression Poisoning
│   │   └── pang2021/    # Accumulative Poisoning (Online Learning)
│   ├── data/            # Datasets + loaders
│   └── linalg.ts        # Matrix operations
├── views/         # React views of non-geometric attacks (registry.ts + one folder per view)
└── store/         # Zustand state management
```

### Key Design Pattern: Algorithm Registry

Every attack is a self-contained `AlgorithmModule` registered via `registerAlgorithm()`. The UI is fully generic — it reads `configSchema` to render controls and `TraceFrame` to render visualizations.

Attacks come in two kinds, and every one belongs to a **category** (a header tab):

- **Geometric** — an `AlgorithmModule`: its results are points and boundaries on the 2-D canvas, with the control panel, timeline and math panel around it (everything below).
- **Non-geometric** — a `ViewModule`: its results are something else (LLM attacks first), so it brings its own React view, which replaces the whole workspace below the header. See [How to Add a Non-Geometric Attack View](#how-to-add-a-non-geometric-attack-view).

---

## How to Add a New Attack Algorithm

This is the most common contribution. Follow these steps:

### 1. Create Your Folder

```bash
mkdir src/engine/architectures/yourpaper2024
```

### 2. Create `model.ts` — Your ML Model

```typescript
// src/engine/architectures/yourpaper2024/model.ts

export function trainModel(X: number[][], y: number[], config: any) {
  // Train your model (SVM, logistic regression, neural net, etc.)
  return { /* model parameters */ };
}

export function predict(model: any, x: number[]): number {
  // Return prediction (class label or regression value)
  return 0;
}
```

### 3. Create `attack.ts` — Your Attack Logic

```typescript
// src/engine/architectures/yourpaper2024/attack.ts
import type { TraceFrame } from '../registry';

export function runMyAttack(
  dataset: any,
  cleanModelState: any,
  config: Record<string, any>,
  onProgress: (frame: TraceFrame) => void,
  onComplete: () => void,
  onError: (msg: string) => void,
  signal?: AbortSignal
) {
  // Main attack loop
  // Call onProgress(frame) at each iteration
  // Call onComplete() when done
  // Call onError(msg) on failure
  // Yield between steps with setTimeout(step, 0), and return without calling onComplete once
  // signal?.aborted is true
}
```

The same `runAttack` runs in the browser **and** on the attack server (`POST /api/attack`, which
uploads the dataset and clean model and streams frames back as JSON), so:

- Frames must survive `JSON.stringify`: plain arrays and numbers. A function on a frame (like
  Biggio's `kernelFn`) is dropped and has to be re-attached by `page.tsx`.
- Anything you need from the clean model must be in `modelState` or its `rawModel`; the upload
  drops `rawModel.K`, `rawModel.X`, `rawModel.y` and `rawModel.kernelFn`.
- Dense linear algebra that dominates run time should go through `compute()` from
  `src/engine/compute` (XᵀX, inverse, rank-one updated inverse, resident matrix-vector products):
  it is plain TypeScript in the browser and CUDA on a server with a GPU. Release anything it
  returns with `dispose()` (GPU memory), e.g. in a `finally` around your generator.

### 4. Create `index.ts` — Register Your Module

```typescript
// src/engine/architectures/yourpaper2024/index.ts
import { AlgorithmModule, registerAlgorithm } from '../registry';
import { trainModel, predict } from './model';
import { runMyAttack } from './attack';

const yourpaper2024: AlgorithmModule = {
  key: 'yourpaper2024',
  name: 'Your Attack Name',
  paper: 'https://arxiv.org/abs/XXXX.XXXXX',
  paperShort: 'Author et al. 2024',
  category: 'poisoning',       // header tab: a key of ATTACK_CATEGORIES (see "Choosing a category")
  modelType: 'classification', // or 'regression'

  datasets: ['gaussian', 'moons'], // Which datasets this algorithm supports

  configSchema: [
    {
      key: 'paramName', label: 'Parameter Label',
      type: 'range', section: 'attack',
      min: 0.01, max: 1.0, step: 0.01,
      tooltip: 'Description of what this parameter does'
    },
    // A field shown only while another has one of the given values:
    // { key: 'hiddenUnits', …, showIf: { key: 'victim', equals: 'mlp' } },
    // Add more config fields...
  ],

  defaultConfig: {
    paramName: 0.5,
  },

  trainClean(dataset, config) {
    const model = trainModel(dataset.train.X, dataset.train.y, config);
    return { modelState: { /* display data */ }, rawModel: model };
  },

  runAttack: runMyAttack,

  predict(rawModel, x) {
    return predict(rawModel, x);
  },
};

registerAlgorithm(yourpaper2024);
export default yourpaper2024;
```

### 5. Register in the Barrel File

Add one line to `src/engine/architectures/index.ts`:

```typescript
import './biggio2012';
import './jagielski2018';
import './pang2021';
import './yourpaper2024';  // ← Add this
```

### 6. (Optional) Add Explainer Steps

Create `explainer.ts` to provide Transformer Explainer-style guided learning:

```typescript
export const yourExplainerSteps = [
  {
    id: 'step-1',
    title: 'Step 1: What Happens',
    equation: '\\theta_{t+1} = \\theta_t - \\beta \\nabla L',
    description: 'Plain-text explanation of this step...',
    phase: 'attack',
  },
  // More steps...
];
```

Then reference it in your `index.ts`:
```typescript
import { yourExplainerSteps } from './explainer';
// ...
explainerSteps: yourExplainerSteps,
```

### 7. Verify

```bash
npm run build  # Must pass with zero errors
```

---

## Choosing a Category

Every attack declares `category`, the header tab it is listed under. The categories live in `src/engine/architectures/categories.ts`:

| key | label | for |
|---|---|---|
| `poisoning` | Poisoning | training-time attacks: corrupting the data (or updates) the model learns from |
| `evasion` | Evasion | test-time attacks: perturbing inputs to a trained model |

A tab appears only once at least one registered attack uses its category, and the header's attack dropdown lists only the attacks of the active tab. The active tab is derived from the active attack; clicking a tab reopens the attack last used in it (`lastAttackByCategory` in the store), else the first. An unknown key does not break the app — it logs a warning and shows a title-cased tab — but declare it properly instead.

### Adding a category

Add one entry to `ATTACK_CATEGORIES` in `categories.ts` — `{ key, label, order, description }`; `order` sets the tab position and `description` is the tab's tooltip. Only add a category once an attack needs it: empty categories are hidden anyway.

---

## How to Add a Non-Geometric Attack View

For an attack whose output is not points on the canvas (e.g. an LLM poisoning attack: prompts, generations, metrics). It has two halves, each one folder plus one import line.

### 1. Register a `ViewModule` (engine side, no React)

```typescript
// src/engine/architectures/yourllmattack/index.ts
import { registerViewModule } from '../registry';

registerViewModule({
  kind: 'view',
  key: 'yourllmattack',              // unique across ALL attacks (geometric and view)
  name: 'Your LLM Attack',
  paper: 'https://arxiv.org/abs/XXXX.XXXXX',
  paperShort: 'Author et al. 2025',
  category: 'poisoning',
  view: 'llm-trace',                 // key of the React view below; several modules may share one
  description: 'One line for tooltips',
  dataUrl: '/llm/yourllmattack',     // optional: where the view finds this module's static data (under public/)
});
```

and one line in `src/engine/architectures/index.ts`: `import './yourllmattack';`

### 2. Register the view component (UI side)

```tsx
// src/views/llm-trace/index.tsx
'use client';
import { registerView, type ViewProps } from '../registry';

function LlmTraceView({ module }: ViewProps) {
  return <div className="p-6 text-foreground">{module.name}</div>;  // semantic tokens only
}

registerView('llm-trace', LlmTraceView);
```

and one line in `src/views/index.ts`: `import './llm-trace';`

The page renders the view in place of the geometric workspace (canvas, control panel, timeline, explainer, exports), so none of their effects — dataset generation, training, compute polling — run while it is active. If the module names a view that is not registered, the page says so. `getAlgorithm()` stays geometric-only: code that may see any attack key uses `getAttack(key)` (kind, name, category) or `findAlgorithm(key)` (undefined for a view module).

### A worked example: `wan2023` + the `llm-poisoning` view

Wan et al. 2023 (LLM instruction-tuning poisoning) is the first view module, and the pattern for LLM attacks whose runs happen outside the app:

- **Data**: no model runs in the browser. The runs are exported elsewhere (here the thesis repository's `scripts/export_viz_trace.py`, schema `wan2023-llm-trace` v1) and committed as static files under `public/llm/wan2023/` with a `manifest.json` (byte size + sha256 per file), the export's `SCHEMA.md` and a provenance `README.md`. Copy or refresh them only with `npm run sync:llm-export -- <export-dir>`: it verifies every source file against the manifest before writing and every copy after, and fails on any mismatch (`npm run sync:llm-export -- --verify public/llm/wan2023` re-checks in place).
- **Module**: `src/engine/architectures/wan2023/index.ts` — a `registerViewModule` call with `view: 'llm-poisoning'` and `dataUrl: '/llm/wan2023'`.
- **View**: `src/views/llm-poisoning/` — `types.ts` mirrors the schema; `data.ts` fetches the small files on mount and checks `schema`/`schema_version`, while the large `predictions.json` is fetched only when a panel asks for it (`usePredictions(url, enabled)`); one file per panel. The view is generic over any module whose `dataUrl` holds a `wan2023-llm-trace` export, so another LLM poisoning attack exported in that schema needs only a new module folder and its data.
- **Rules it follows**: numbers come from the export as-is — derive only what the schema defines (here: label probabilities normalised from log-probs), never recompute an aggregate differently; claims in the text cite the paper (§, Fig., Table) and say n; text from sensitive (toxicity) tasks is blurred until the reader shows it; semantic tokens only, and Recharts colours from `useThemeTokens()`.

---

## How to Add a New Dataset

Edit `src/engine/data/datasets.ts`:

```typescript
export const DATASETS: Record<string, DatasetGenerator> = {
  // ... existing datasets ...
  myDataset: {
    name: 'My Dataset',
    generate: (n: number) => {
      const X: number[][] = [];
      const y: number[] = [];
      // Generate n data points
      return { X, y };
    },
    modelTypes: ['classification'], // or ['regression'] or both
  },
};
```

Then add the dataset key to your algorithm's `datasets` array.

---

## Coding Conventions

### File Naming
- **Algorithm modules**: `src/engine/architectures/<paper_key>/` — all lowercase, paper year
- **Components**: `src/components/PascalCase.tsx`
- **Engine files**: `src/engine/camelCase.ts`

### Styling
- Use **Tailwind CSS v4** utility classes
- Use **design tokens** from `globals.css` (e.g., `bg-secondary`, `text-muted-foreground`, `border-border-subtle`)
- **Never** use hardcoded colors like `bg-white`, `text-gray-400`, `bg-[#059669]`, or a hex literal in a chart prop — the app has a **light and a dark theme**, and a hardcoded color is broken in one of them
- **Don't write `dark:` variants.** The token layer handles theming; a `dark:` override is a second source of truth
- Any new base/element CSS goes **inside `@layer base`** — unlayered CSS silently outranks every Tailwind utility
- Panel headers use the `eyebrow` CSS class
- Glass panels use the `glass-panel` CSS class
- Check your change in **both themes** (toggle is in the header) before opening a PR
- Reference `docs/design.md` for the full token system

### TraceFrame Contract
- Every algorithm must emit `TraceFrame` objects via `onProgress()`
- Required fields: `iteration`, `poisonX`, `poisonY`, `poisonedModel`, `poisonedRawModel`, `objectiveValue`, `gradients`, `gradientNorms`, `deltaW`
- Optional fields are typed with `?` — don't break existing fields

### State Management
- All shared state lives in `src/store/useStore.ts` (Zustand)
- Access via `useStore()` hook in components
- Never store derived state — compute it in components

---

## Common Pitfalls

| Pitfall | Solution |
|---|---|
| Using `border-border-default` | Use `border-border` (no `-default` suffix) |
| Using `bg-[var(--my-token)]` | Use `bg-secondary` or proper Tailwind token |
| Utility class has no effect | Some unlayered CSS is beating it — move that rule into `@layer base` |
| Font looks like the system font | Font tokens must reference `var(--font-inter)`, not `"Inter"` |
| Color in a Recharts prop / JS-built SVG | Read it from `useThemeTokens()` so it follows the theme |
| Import cycle in architectures | Only import from `../registry` and `../../linalg` |
| TraceFrame field missing | Check the `TraceFrame` interface in `registry.ts` |
| Model serialization in Web Worker | `postMessage` can't send functions — strip `kernelFn` |
| Canvas doesn't update | Make sure `attackTrace` in Zustand is updated with new array ref |

---

## Testing

```bash
npm run build    # Type checking + production build
npm run dev      # Manual testing in browser
```

There are no automated unit tests (yet). Manual verification workflow:
1. Select each algorithm from the dropdown
2. Generate dataset → Train model → Launch attack
3. Step through iterations and verify canvas updates
4. Check MathPanel values match expected behavior
5. Verify the ExplainerOverlay shows correct steps (if present)

---

## Merge Conflict Prevention

The project is structured to minimize conflicts:

| Area | Who touches it | Files |
|---|---|---|
| **New algorithm** | Algorithm author | `src/engine/architectures/<key>/` (isolated folder) + 1 line in `index.ts` |
| **New dataset** | Data contributor | `src/engine/data/datasets.ts` (append only) |
| **UI component** | Frontend dev | `src/components/` (one file per component) |
| **Design tokens** | Designer | `src/app/globals.css` (append to `@theme`) |
| **State** | Core team | `src/store/useStore.ts` (centralized) |

**Golden rule**: Each algorithm lives in its own folder. You never need to modify another algorithm's code.

---

## Documentation

Keep these docs up to date when making significant changes:
- `docs/architecture.md` — System architecture + folder structure
- `docs/phases.md` — Development roadmap + task tracking
- `docs/memory.md` — Project history + key decisions
- `docs/design.md` — UI design system tokens + component recipes
- `CONTRIBUTING.md` — This file
