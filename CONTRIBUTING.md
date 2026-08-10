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
│   │   ├── registry.ts  # Core interface + global registry
│   │   ├── index.ts     # Barrel file — import all modules here
│   │   ├── biggio2012/  # SVM Poisoning
│   │   ├── jagielski2018/ # Regression Poisoning
│   │   └── pang2021/    # Accumulative Poisoning (Online Learning)
│   ├── data/            # Datasets + loaders
│   └── linalg.ts        # Matrix operations
└── store/         # Zustand state management
```

### Key Design Pattern: Algorithm Registry

Every attack is a self-contained `AlgorithmModule` registered via `registerAlgorithm()`. The UI is fully generic — it reads `configSchema` to render controls and `TraceFrame` to render visualizations.

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
  onError: (msg: string) => void
) {
  // Main attack loop
  // Call onProgress(frame) at each iteration
  // Call onComplete() when done
  // Call onError(msg) on failure
}
```

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
  modelType: 'classification', // or 'regression'

  datasets: ['gaussian', 'moons'], // Which datasets this algorithm supports

  configSchema: [
    {
      key: 'paramName', label: 'Parameter Label',
      type: 'range', section: 'attack',
      min: 0.01, max: 1.0, step: 0.01,
      tooltip: 'Description of what this parameter does'
    },
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
