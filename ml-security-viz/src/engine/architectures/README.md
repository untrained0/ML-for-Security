# Algorithm Modules — Developer Guide

## Structure Convention

Every algorithm follows this folder layout:

```
<paper_key>/
├── index.ts         # AlgorithmModule registration + configSchema
├── model.ts         # Model training / prediction logic
├── attack.ts        # Attack loop + TraceFrame emission
├── gradient.ts      # (Optional) Gradient computation helpers
└── explainer.ts     # (Optional) Explainer step definitions
```

## How to Add a New Algorithm

1. Create folder: `src/engine/architectures/<paper_key>/`
2. Implement `model.ts` → training + prediction
3. Implement `attack.ts` → attack loop emitting `TraceFrame`
4. Implement `index.ts` → `registerAlgorithm(module)`
5. Add `import './<paper_key>'` to `index.ts` (barrel file)
6. Run `npm run build` to verify

## Existing Algorithms

| Key | Paper | Type | Files |
|---|---|---|---|
| `biggio2012` | Biggio et al. 2012 — SVM Poisoning | Classification | model.ts, attack.ts, kernels.ts |
| `jagielski2018` | Jagielski et al. 2018 — Regression Poisoning | Regression | model.ts, attack.ts |
| `pang2021` | Pang et al. 2021 — Accumulative Poisoning | Classification | model.ts, attack.ts, gradient.ts, explainer.ts |

## Key Interfaces

See `registry.ts` for:
- `AlgorithmModule` — the contract every module implements
- `TraceFrame` — the data emitted at each attack iteration
- `ConfigField` — schema for auto-generated UI controls

## Rules

- **Never** modify another algorithm's folder
- Import only from `../registry` and `../../linalg`
- All `TraceFrame` extensions must be optional (`?`) to avoid breaking others
