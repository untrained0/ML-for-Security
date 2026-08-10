/**
 * Pang et al. 2021 — Explainer steps for Transformer Explainer-style guided learning
 * 
 * Each step provides a title, KaTeX equation, plain-text description,
 * and optionally highlights a UI element.
 */

export const pang2021ExplainerSteps = [
  {
    id: 'overview',
    title: 'Accumulative Poisoning — Overview',
    equation: '\\theta_{t+1} = \\theta_t - \\beta \\nabla_\\theta L(S_t; \\theta_t)',
    description:
      'In online learning, a model updates its parameters θ one batch at a time using SGD. ' +
      'The attacker exploits this sequential nature: instead of injecting all poison upfront, ' +
      'they subtly perturb each batch to secretly steer θ toward a vulnerable state.',
    highlightElement: 'canvas',
  },
  {
    id: 'accum-phase',
    title: 'Step 1: Accumulative Phase',
    equation: 'A_t(S_t) = S_t + \\delta_t, \\quad \\|\\delta_t\\|_\\infty \\leq \\varepsilon',
    description:
      'During the accumulative phase, each clean batch S_t is perturbed by a small δ_t ' +
      'bounded within an L∞ ball of radius ε. The perturbation is carefully crafted so that ' +
      'model accuracy stays above a secrecy threshold — the attack remains invisible.',
    phase: 'accumulative',
    highlightElement: 'poison-points',
  },
  {
    id: 'accum-objective',
    title: 'Step 2: Accumulative Objective',
    equation: '\\max_{\\delta_t} L(S_{\\text{trigger}}; \\theta_{t+1}(\\delta_t))',
    description:
      'Each perturbation δ_t is optimized to maximize the loss that a future trigger batch ' +
      'would cause. The attacker looks one step ahead: "if I perturb this batch, how much more ' +
      'damage can the trigger do?" This is solved using PGD (Projected Gradient Descent).',
    phase: 'accumulative',
  },
  {
    id: 'secrecy',
    title: 'Step 3: Secrecy Constraint',
    equation: '\\text{Acc}(\\theta_{t+1}) \\geq \\tau \\quad \\text{(e.g., } \\tau = 0.7\\text{)}',
    description:
      'The key insight: perturbations must be secret. After each SGD update on the perturbed batch, ' +
      'the model accuracy on the test set must stay above threshold τ. If it drops, the perturbation ' +
      'is discarded and the clean batch is used instead. This is what makes the attack stealthy.',
    phase: 'accumulative',
    highlightElement: 'accuracy-metric',
  },
  {
    id: 'pgd',
    title: 'Step 4: PGD Perturbation',
    equation: '\\delta_t^{(k+1)} = \\Pi_{\\|\\cdot\\|_\\infty \\leq \\varepsilon} \\left( \\delta_t^{(k)} + \\alpha \\cdot \\text{sign}(\\nabla_\\delta L) \\right)',
    description:
      'The perturbation is generated using Projected Gradient Descent. At each PGD step, ' +
      'we compute the gradient of the trigger loss w.r.t. δ, take a signed step (FGSM-style), ' +
      'then project back onto the L∞ ball. Multiple PGD steps refine the perturbation.',
    phase: 'accumulative',
  },
  {
    id: 'drift',
    title: 'Step 5: Parameter Drift',
    equation: '\\|\\theta_t - \\theta_0\\| \\nearrow \\quad \\text{while} \\quad \\text{Acc}(\\theta_t) \\approx \\text{Acc}(\\theta_0)',
    description:
      'Over multiple accumulative batches, θ silently drifts away from the clean initialization θ₀. ' +
      'Watch the "Accumulated Drift" metric rise while accuracy stays stable. ' +
      'This hidden drift is the accumulation — the model looks fine, but it\'s primed for the trigger.',
    phase: 'accumulative',
    highlightElement: 'drift-metric',
  },
  {
    id: 'trigger',
    title: 'Step 6: Trigger Batch',
    equation: '\\theta_{T+1} = \\theta_T - \\beta \\nabla_\\theta L(S_{\\text{trigger}}^*; \\theta_T)',
    description:
      'After the accumulative phase, the attacker releases a single crafted trigger batch S*_trigger. ' +
      'This batch is also generated via PGD, but now optimized to maximize validation loss ' +
      'given the accumulated model state θ_T. One SGD step on this batch causes a catastrophic accuracy drop.',
    phase: 'trigger',
    highlightElement: 'canvas',
  },
  {
    id: 'impact',
    title: 'Step 7: Catastrophic Drop',
    equation: '\\text{Acc}(\\theta_{T+1}) \\ll \\text{Acc}(\\theta_T) \\approx \\text{Acc}(\\theta_0)',
    description:
      'The result: accuracy plummets from near-clean levels to near-random in a single update step. ' +
      'This is the power of accumulation — without it, the trigger alone would have minimal effect. ' +
      'The accumulative phase magnified the trigger\'s destructive potential by secretly steering θ.',
    phase: 'trigger',
    highlightElement: 'accuracy-metric',
  },
];
