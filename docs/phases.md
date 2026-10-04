# Phases — ML Security Attack Visualizer

## Phase 1 — Foundation & SVM Core (Current)

> **Goal**: Get a working interactive SVM with decision boundary on screen.

| # | Task | Deliverable |
|---|---|---|
| 1.1 | Set up project skeleton | `index.html`, CSS design system, JS module structure |
| 1.2 | Implement linear algebra utilities | `linalg.js` — matrix multiply, transpose, inverse, SVD for 2D |
| 1.3 | Implement kernel functions | `kernels.js` — linear, RBF, polynomial |
| 1.4 | Implement SVM solver (SMO) | `svm.js` — Sequential Minimal Optimization for binary SVM |
| 1.5 | Build dataset generators | `datasets.js` — moons, circles, blobs, XOR, Gaussian clusters |
| 1.6 | Build D3 scatter canvas | `canvas.js` — interactive 2D canvas with zoom, pan, click-to-add |
| 1.7 | Render decision boundary | Contour plot overlay showing the SVM decision surface |
| 1.8 | Build control panel UI | Kernel type, C parameter, dataset selector, train button |
| 1.9 | Wire everything together | `app.js` + `state.js` — training triggers boundary redraw |
| 1.10 | Polish Phase 1 UI | Animations, responsive layout, dark theme |

**Exit Criteria**: User can select a dataset, choose a kernel, train an SVM, and see the decision boundary update in real time.

---

## Phase 2 — Poisoning Attack Engine

> **Goal**: Implement the Biggio et al. 2012 gradient-ascent poisoning attack and run it step-by-step.

| # | Task | Deliverable |
|---|---|---|
| 2.1 | Set up Web Worker infrastructure | `worker.js` — message protocol, error handling |
| 2.2 | Implement gradient computation | Gradient of attacker objective w.r.t. attack point (Eq. 7–10 from paper) |
| 2.3 | Implement line search | Backtracking line search with Armijo condition |
| 2.4 | Implement poisoning loop | Full gradient-ascent attack with configurable iterations |
| 2.5 | Implement initialization strategies | Random flip, leverage-based, Cook's distance, influence-based |
| 2.6 | Build playback controls | Play / Pause / Step / Reset buttons with speed slider |
| 2.7 | Animate poison point injection | Visual feedback when new poison points are added to the dataset |
| 2.8 | Show boundary shift per iteration | Smooth transition of decision boundary as attack progresses |
| 2.9 | Store attack trace | Save full history of poison points, boundary, and metrics per iteration |

**Exit Criteria**: User can configure and run a poisoning attack, step through iterations, and watch the decision boundary degrade.

---

## Phase 3 — Mathematical Inspector & Timeline

> **Goal**: Show the user exactly what's changing inside the model at each step.

| # | Task | Deliverable |
|---|---|---|
| 3.1 | Build math inspector panel | Side panel showing w, b, α, support vectors, kernel matrix |
| 3.2 | Integrate KaTeX rendering | Render all formulas with proper LaTeX notation |
| 3.3 | Highlight changed values | Animate value changes (flash green for increase, red for decrease) |
| 3.4 | Build loss timeline chart | Chart.js line chart: validation error, test error, objective value |
| 3.5 | Build Δw timeline | Track norm of weight change per iteration |
| 3.6 | Build support vector tracker | Show which points become/lose support vector status |
| 3.7 | Add equation references | Link each formula to the paper equation number + tooltip explanation |
| 3.8 | Add gradient direction indicator | Arrow on canvas showing current gradient direction for each poison point |

**Exit Criteria**: Full mathematical transparency — user can trace every numerical change from the paper through the UI.

---

## Phase 4 — Gradient Field & Comparison Mode

> **Goal**: Replicate the paper's gradient field visualization and enable before/after comparison.

| # | Task | Deliverable |
|---|---|---|
| 4.1 | Implement objective function heatmap | Contour plot of attacker's objective over (x, y) space |
| 4.2 | Implement gradient vector field | Quiver plot of gradient directions overlaid on the heatmap |
| 4.3 | Build before/after toggle | Side-by-side or overlay view comparing clean vs. poisoned model |
| 4.4 | Build comparison metrics table | Table showing accuracy, precision, recall, F1 for both models |
| 4.5 | Add poison point trajectory paths | Draw the gradient descent path of each poison point (as in Fig. 2) |

**Exit Criteria**: User sees the full attack landscape — where poison points are most effective and how they got there.

---

## Phase 5 — Polish, Export & Documentation

> **Goal**: Make it thesis-ready and shareable.

| # | Task | Deliverable |
|---|---|---|
| 5.1 | ~~Export as PNG / SVG~~ ✅ | Screenshot of canvas + charts |
| 5.2 | ~~Export attack trace as JSON~~ ✅ | Full reproducibility data |
| 5.3 | ~~CSV dataset upload~~ ✅ | Parse and validate user CSV files |
| 5.4 | ~~Add onboarding tutorial~~ ✅ | Guided walkthrough for first-time users |
| 5.5 | Performance optimization | Profile and optimize for datasets up to 5,000 points |
| 5.6 | Cross-browser testing | Chrome, Firefox, Edge (latest 2 versions) |
| 5.7 | Write user documentation | README, usage guide, mathematical references |
| 5.8 | Deploy to GitHub Pages | Static hosting with CI |

---

## Phase 6 — Multi-Attack Framework

> **Goal**: Extend beyond SVM poisoning to a general adversarial ML visualization platform.

| # | Task | Deliverable |
|---|---|---|
| 6.1 | Label-flip attack | Visualize random and strategic label flipping |
| 6.2 | Evasion attack (FGSM-style) | Show adversarial perturbation at test time |
| 6.3 | Backdoor attack | Visualize trigger injection and model behavior |
| 6.4 | Defense visualizations | TRIM, RONI, data sanitization overlays |
| 6.5 | ~~Regression poisoning~~ ✅ | Jagielski et al. 2018 — Ridge/LASSO/OLS |
| 6.6 | ~~Paper replication dashboard~~ ✅ | Registry framework established |

---

## Phase 7 — Accumulative Poisoning (Pang et al. 2021) ✅

> **Goal**: Implement "Accumulative Poisoning Attacks on Real-time Data" (online learning setting).

| # | Task | Deliverable |
|---|---|---|
| 7.1 | ~~Logistic Regression model~~ ✅ | `pang2021/model.ts` — SGD-based training |
| 7.2 | ~~PGD gradient computation~~ ✅ | `pang2021/gradient.ts` — L∞ perturbation craft |
| 7.3 | ~~Two-phase attack loop~~ ✅ | `pang2021/attack.ts` — accumulate + trigger |
| 7.4 | ~~Explainer steps~~ ✅ | `pang2021/explainer.ts` — 8 step-by-step cards |
| 7.5 | ~~Algorithm registration~~ ✅ | `pang2021/index.ts` — configSchema + registry |
| 7.6 | ~~Neural-network victim~~ ✅ | `pang2021/model.ts` — MLP (1–3 hidden layers, ReLU/tanh) beside logistic regression; exact HVPs and pixel gradients via the R-operator |
| 7.7 | ~~Paper's online optimiser~~ ✅ | SGD momentum 0.9 carried into the trigger step, weight-momentum trick (Table 1), C up to 100 |
| 7.8 | ~~Federated setting (Algorithm 2)~~ ✅ | `pang2021/federated.ts` — gradient-level poisoners, server clipping, direct-poisoner baseline; measured vs Tables 3–6 |
| 7.9 | ~~Federated non-reproduction: ablation + reference ResNet-18~~ ✅ | BN / K-class / S_val / T ablation on the MLP (no collapse; `scripts/pang_ablation.ts`); the authors' code with ResNet-18 reproduces Table 3's collapse (`docs/data/pang_reference_resnet_2026-09-30/`) |
| 7.6 | ~~Extend TraceFrame~~ ✅ | Phase, batchIndex, perturbationNorm, secrecy, drift |

---

## Phase 8 — Transformer Explainer-Style Guided Learning ✅

> **Goal**: Provide interactive step-by-step math explanations inspired by the Transformer Explainer (CHI 2026).

| # | Task | Deliverable |
|---|---|---|
| 8.1 | ~~ExplainerOverlay component~~ ✅ | Floating panel with KaTeX equations + navigation |
| 8.2 | ~~DataFlowDiagram component~~ ✅ | Animated SVG clean vs. poisoned pipeline |
| 8.3 | ~~Algorithm-specific explainer integration~~ ✅ | `AlgorithmModule.explainerSteps` interface |
| 8.4 | ~~Phase-aware UI indicators~~ ✅ | Badges showing accumulative/trigger/baseline |

---

## Phase 9 — Future Extensions

> **Goal**: Continue expanding the attack/defense repertoire.

| # | Task | Deliverable |
|---|---|---|
| 9.1 | Add explainerSteps to Biggio 2012 | Step-by-step SVM poisoning explanation |
| 9.2 | Add explainerSteps to Jagielski 2018 | Step-by-step regression poisoning explanation |
| 9.3 | Label-flip + evasion attacks | Lightweight attack modules |
| 9.4 | Defense visualizations | TRIM, RONI overlays |
| 9.5 | Web Worker migration for Pang 2021 | Move computation off main thread |
| 9.6 | ~~Attack-category tabs + non-geometric views~~ ✅ | `categories.ts` (poisoning, evasion), `ViewModule` + `src/views/` registry, `GeometricWorkspace` / `ViewWorkspace`, header tab bar — step 1 of LLM attacks in the app |
| 9.7 | ~~First LLM view: Wan et al. 2023 replay~~ ✅ | `wan2023` view module + `src/views/llm-poisoning/` (poison construction, attack over epochs, example browser, comparisons) over the static export in `public/llm/wan2023/`; `npm run sync:llm-export` verifies it — step 3 of LLM attacks in the app |

