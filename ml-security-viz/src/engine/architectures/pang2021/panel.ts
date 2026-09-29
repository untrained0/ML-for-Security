/**
 * Pang et al. 2021 — wording of the math panel. The model rows describe the online victim
 * (model.ts): for the MLP, w and b are its output layer and the norm covers every weight matrix;
 * for logistic regression they are the model itself. The objective is Eq. 7's alignment, and the
 * reference equations are the paper's online-learning ones (§2.2, §3.2).
 */

import type { MathPanelLabels } from '../registry';

const EQUATIONS: MathPanelLabels['equations'] = [
  { title: 'Online Update (Eq. 3, SGD momentum μ)', body: 'vₜ₊₁ = μ vₜ + ∇L(Sₜ; θₜ),  θₜ₊₁ = θₜ − β vₜ₊₁' },
  { title: 'Poisoned Trigger (Eq. 6)', body: 'min_P ∇L(S_val; θ_T)ᵀ ∇L(P(S_T); θ_T)' },
  { title: 'Accumulative Objective (Eq. 7)', body: 'min_{P,A} ∇L(S_val; A(θ_T))ᵀ ∇L(P(S_T); A(θ_T))  s.t.  L(S_val; A(θ_T)) ≤ L(S_val; θ_T) + γ' },
  { title: 'Accumulative Batch (Eq. 9)', body: 'max_{P,Aₜ} ∇L(Aₜ(Sₜ); θₜ)ᵀ [∇L(Sₜ; θₜ) + λ · ∇_θ(∇L(S_val; θₜ)ᵀ ∇L(P(S_T); θₜ))]' },
];

const SHARED: MathPanelLabels = {
  showSupportVectors: false,
  lowAccuracyHint: 'Try more burn-in epochs or a smaller learning rate β.',
  objective: {
    symbol: '\\langle \\nabla\\mathcal{L}_{val}, \\nabla\\mathcal{L}_{\\mathcal{P}} \\rangle',
    tooltip: 'Eq. 7 alignment of ∇L(S_val; θ) with the trigger gradient ∇L(P(S_T); θ) at this frame (a cosine when gradient normalisation is on). The attacker drives it down: negative means one trigger step raises the validation loss.',
    worse: 'lower',
  },
  gradientLabel: 'Avg ‖∇ₓH_t‖',
  equations: EQUATIONS,
};

const LOGREG: MathPanelLabels = {
  ...SHARED,
  weights: {
    symbol: '\\mathbf{w}', tooltip: 'Logistic-regression weights, one per pixel of the normalised image x̃ = (x − μ)/(σ√d)',
    poisonedSymbol: '\\mathbf{w}_t', poisonedTooltip: 'Weights of the online model at this frame, after its update',
  },
  bias: {
    symbol: 'b', tooltip: 'Bias of the logit z = wᵀx̃ + b',
    poisonedSymbol: 'b_t', poisonedTooltip: 'Bias of the online model at this frame',
  },
  weightNorm: {
    symbol: '\\|\\mathbf{w}\\|',
    tooltip: 'L2 norm of w. Accuracy depends only on the direction of w; a larger norm means more confident predictions and smaller gradients',
    poisonedSymbol: '\\|\\mathbf{w}_t\\|', poisonedTooltip: 'L2 norm of the online model\'s weights at this frame',
  },
  loss: {
    symbol: '\\mathcal{L}_{\\text{log}}', tooltip: 'Mean logistic loss log(1 + e^{−y·z}) of the burn-in model θ₀ on the test set',
    poisonedTooltip: 'Mean logistic loss of the online model at this frame, on the test set',
  },
};

const MLP: MathPanelLabels = {
  ...SHARED,
  weights: {
    symbol: '\\mathbf{w}_{\\text{out}}', tooltip: 'Output-layer weights of the MLP: the logit z = w_outᵀ h(x̃) + b_out reads the last hidden layer h(x̃)',
    poisonedSymbol: '\\mathbf{w}_{\\text{out},t}', poisonedTooltip: 'Output-layer weights of the online network at this frame, after its update',
  },
  bias: {
    symbol: 'b_{\\text{out}}', tooltip: 'Bias of the output logit',
    poisonedSymbol: 'b_{\\text{out},t}', poisonedTooltip: 'Output bias of the online network at this frame',
  },
  weightNorm: {
    symbol: '\\|W\\|_F',
    tooltip: 'Frobenius norm of all weight matrices W_1 … W_L together (biases excluded) — how far the network has grown from its initialisation scale',
    poisonedSymbol: '\\|W_t\\|_F', poisonedTooltip: 'Frobenius norm of the online network\'s weight matrices at this frame',
  },
  loss: {
    symbol: '\\mathcal{L}_{\\text{log}}', tooltip: 'Mean logistic loss log(1 + e^{−y·z}) of the burn-in network θ₀ on the test set',
    poisonedTooltip: 'Mean logistic loss of the online network at this frame, on the test set',
  },
};

export const pangMathPanel = (config: Record<string, any>): MathPanelLabels =>
  (config.victim ?? 'mlp') === 'mlp' ? MLP : LOGREG;
