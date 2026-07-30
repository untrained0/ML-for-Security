# PRD — ML Security Attack Visualizer

## 1. What to Build

An **interactive, browser-based visualization tool** that demonstrates how adversarial attacks (starting with **Poisoning Attacks on SVMs** — Biggio et al., 2012) manipulate machine learning models. The tool replicates the core mathematics of each attack paper and renders every intermediate state — per data-point, per iteration — so the user can watch the model's decision boundary, weights, support vectors, and loss surface change in real time.

### Core Capabilities

| Capability | Description |
|---|---|
| **Live Decision Boundary Canvas** | 2D scatter plot with a dynamic decision boundary (linear / RBF kernel) that redraws as poisoned points are injected. Inspired by TensorFlow Playground's interactive canvas. |
| **Poisoning Attack Engine** | Client-side JS implementation of the gradient-ascent poisoning attack from Biggio et al. (arXiv:1206.6389). Supports linear and RBF kernels. |
| **Step-by-Step Playback** | Play / Pause / Step controls to iterate through the attack one poisoned point or one gradient-ascent iteration at a time. |
| **Mathematical State Inspector** | Side panel showing real-time values of: weight vector **w**, bias **b**, support vectors, dual variables **α**, hinge loss, objective function value, gradient direction & magnitude. |
| **Loss / Metric Timeline** | Line charts tracking validation error, test error, objective value, and Δw across iterations. |
| **Gradient Field Overlay** | Optional vector-field overlay on the decision-boundary canvas showing the gradient of the attacker's objective at each grid point (as in the original paper's Figure 2). |
| **Attack Configuration Panel** | Controls for kernel type, regularization C, learning rate η, number of poison points, initialization strategy, step size decay β. |
| **Dataset Selector** | Built-in 2D datasets (moons, circles, linearly separable blobs, XOR) plus CSV upload. |
| **Before / After Comparison** | Side-by-side or toggle view comparing the clean model vs. poisoned model. |
| **Export & Share** | Export current state as PNG / SVG, and export attack trace as JSON for reproducibility. |

---

## 2. Targeted Users

| User Segment | Why They Care |
|---|---|
| **Graduate students & thesis writers** | Need to include clear, accurate attack visualizations in papers and presentations. |
| **ML Security researchers** | Want a rapid prototyping sandbox to test new attack / defense ideas visually. |
| **Course instructors** | Need an interactive demo for adversarial ML lectures (like TF Playground for neural nets). |
| **Security engineers** | Want to understand how poisoning attacks degrade model performance before deploying defenses. |
| **Conference / poster presenters** | Need eye-catching, interactive visualizations for live demos. |

---

## 3. Features (Prioritized)

### Phase 1 — MVP (SVM Poisoning Visualizer)

- [x] Interactive 2D scatter canvas with drag-to-add data points
- [x] SVM classifier with linear & RBF kernels (client-side)
- [x] Gradient-ascent poisoning attack (Biggio et al. 2012)
- [x] Play / Pause / Step iteration controls
- [x] Real-time decision boundary rendering
- [x] Mathematical state panel (w, b, α, support vectors, loss)
- [x] Loss & error timeline charts
- [x] Attack parameter controls (C, η, β, kernel, # poison points)
- [x] Built-in toy datasets (moons, blobs, circles)

### Phase 2 — Deep Dive

- [ ] Gradient vector field overlay
- [ ] Before / After comparison mode
- [ ] CSV dataset upload
- [ ] Export state as PNG / SVG / JSON
- [ ] Tooltip explanations on every mathematical term (like Transformer Explainer)
- [ ] Dark / Light theme toggle

### Phase 3 — Multi-Attack Framework

- [ ] Label-flip attack
- [ ] Backdoor / trojan attack visualization
- [ ] Evasion attack (adversarial examples at test time)
- [ ] Defense visualizations (TRIM, RONI, data sanitization)
- [ ] Regression poisoning integration (from existing PoisonRegression codebase)

### Phase 4 — Paper Replication Suite

- [ ] Full replication dashboard per paper (Biggio 2012, Jagielski 2018, etc.)
- [ ] Side-by-side paper figure ↔ interactive reproduction
- [ ] LaTeX-ready figure export
- [ ] Embeddable iframe mode for thesis HTML supplements
