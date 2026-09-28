/**
 * Pang et al. 2021 — guided explainer steps.
 *
 * Each card pairs the symbolic equation with the same equation evaluated on
 * whichever frame the user is scrubbed to, so the numbers on screen are the
 * model's real numbers. Scrub the timeline and the maths moves with it.
 */

import type { ExplainerStep, ExplainerReadout, TraceFrame } from '../registry';

const fmt = (v: number | undefined, digits = 4) =>
  v === undefined || v === null || Number.isNaN(v) ? '—' : v.toFixed(digits);

const vec = (v: number[] | undefined, digits = 3) =>
  !v ? '?' : `\\begin{bmatrix}${v.map(x => x.toFixed(digits)).join(' \\\\ ')}\\end{bmatrix}`;

/** Angle between two weight-space vectors, in degrees. */
function angleBetween(a?: number[], b?: number[]): number | undefined {
  if (!a || !b || a.length !== b.length) return undefined;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  if (na === 0 || nb === 0) return undefined;
  return (Math.acos(Math.max(-1, Math.min(1, dot / (Math.sqrt(na) * Math.sqrt(nb))))) * 180) / Math.PI;
}

const alignmentTone = (a: number | undefined): ExplainerReadout['tone'] =>
  a === undefined ? 'neutral' : a < 0 ? 'attack' : 'clean';

export const pang2021ExplainerSteps: ExplainerStep[] = [
  {
    id: 'online-step',
    title: 'The setting: learning that never stops',
    equation: '\\theta_{t+1} = \\theta_t - \\beta \\nabla_\\theta \\mathcal{L}(S_t; \\theta_t)',
    description:
      'A real-time model retrains on every batch as it arrives (Eq. 3). Nobody reviews S_t before it is ' +
      'used — it is consumed, applied, and discarded. That single property is the entire attack surface: ' +
      'the attacker does not need to poison a stored dataset, only to be upstream of one update.',
    visual: 'online-step',
    highlightElement: 'canvas',
    liveEquation: (f, cfg) =>
      f ? `\\theta_{${f.iteration}} \\;\\rightarrow\\; \\theta_{${f.iteration + 1}}, \\quad \\beta = ${cfg.learningRate ?? 1.5}` : null,
    readouts: (f) => !f ? [] : [
      { label: 'batch', value: `${f.batchIndex ?? 0}` },
      { label: 'test acc', value: `${((f.poisonedAccuracy ?? 0) * 100).toFixed(1)}%`, tone: 'clean', meter: f.poisonedAccuracy },
    ],
  },

  {
    id: 'objective',
    title: 'The attacker\'s real objective',
    equation:
      '\\mathcal{L}(S_{val}; \\theta_{T+1}) \\approx \\mathcal{L}(S_{val}; \\theta_T) - \\beta \\underbrace{\\langle \\nabla\\mathcal{L}(S_{val}), \\nabla\\mathcal{L}(\\mathcal{P}(S_T)) \\rangle}_{\\text{alignment}}',
    description:
      'Expand the loss after one trigger step to first order and everything collapses to a single number: ' +
      'the inner product of two gradients. If the trigger gradient points the same way as the validation ' +
      'gradient, the step helps the model. If it points the opposite way, the step actively damages it. ' +
      'So the attacker wants this alignment as negative as possible — that is Eq. 7.',
    visual: 'alignment',
    liveEquation: (f, cfg) => {
      if (!f || f.gradientAlignment === undefined) return null;
      const beta = cfg.learningRate ?? 1.5;
      return `\\Delta\\mathcal{L} \\approx -\\,${beta} \\times (${fmt(f.gradientAlignment, 5)}) = ${fmt(-beta * f.gradientAlignment, 5)}`;
    },
    readouts: (f) => {
      if (!f) return [];
      const ang = angleBetween(f.gradVal, f.gradTrigger);
      return [
        { label: '⟨∇val, ∇trig⟩', value: fmt(f.gradientAlignment, 5), tone: alignmentTone(f.gradientAlignment) },
        ...(ang !== undefined
          ? [{ label: 'angle', value: `${ang.toFixed(0)}°`, tone: (ang > 90 ? 'attack' : 'clean') as ExplainerReadout['tone'], meter: ang / 180 }]
          : []),
      ];
    },
  },

  {
    id: 'accumulate',
    title: 'Step 1 — Accumulate, without looking guilty',
    equation: 'd_t = \\nabla\\mathcal{L}(S_t; \\theta_t) + \\lambda\\, G_t, \\qquad G_t = \\nabla_\\theta \\langle \\nabla\\mathcal{L}(S_{val}), \\nabla\\mathcal{L}(\\mathcal{P}(S_T)) \\rangle',
    description:
      'Each round the attacker picks a direction with two jobs. The first term is the honest gradient, so ' +
      'the model keeps learning and the metrics keep looking normal. The second, G_t, is the direction that ' +
      'drives the alignment down — the accumulation. λ sets the mix. Set λ = 0 and you get an ordinary ' +
      'honest update; the trigger later lands with almost no effect.',
    phase: 'accumulative',
    visual: 'accumulation',
    liveEquation: (f, cfg) => {
      if (!f || !f.gradBatch || !f.gradAccum) return null;
      return `d_t = ${vec(f.gradBatch)} + ${cfg.lambda ?? 2} \\cdot ${vec(f.gradAccum)}`;
    },
    readouts: (f, cfg) => !f ? [] : [
      { label: 'λ', value: `${cfg.lambda ?? 2}` },
      { label: '‖G_t‖', value: fmt(f.alignmentGradNorm, 4), tone: 'warning' },
      { label: 'drift ‖θ−θ₀‖', value: fmt(f.accumulatedDrift, 3), tone: 'warning', meter: Math.min(1, (f.accumulatedDrift ?? 0) / 2) },
    ],
  },

  {
    id: 'perturb',
    title: 'Step 2 — Bend the batch toward that direction',
    equation:
      '\\max_{\\|\\delta\\|_\\infty \\le \\varepsilon} \\; \\langle \\nabla\\mathcal{L}(S_t + \\delta; \\theta_t),\\; d_t \\rangle',
    description:
      'The attacker cannot set the update directly — only the data. So they search for the perturbation δ, ' +
      'bounded in an L∞ ball of radius ε, whose gradient points as close to d_t as possible. Projected ' +
      'gradient descent with a signed step solves it: nudge every feature by ±α, then clip back into the ball.',
    phase: 'accumulative',
    liveEquation: (f, cfg) =>
      f ? `\\|\\delta_{${f.batchIndex ?? 0}}\\|_\\infty = ${fmt(f.perturbationNorm, 3)} \\;\\le\\; \\varepsilon = ${cfg.epsilon ?? 0.8}` : null,
    readouts: (f, cfg) => !f ? [] : [
      { label: '‖δ‖∞', value: fmt(f.perturbationNorm, 3), tone: 'attack', meter: Math.min(1, (f.perturbationNorm ?? 0) / (cfg.epsilon ?? 0.8)) },
      { label: 'PGD steps', value: `${cfg.pgdSteps ?? 8}` },
    ],
  },

  {
    id: 'stealth',
    title: 'Step 3 — Stay inside the stealth budget',
    equation: '\\mathcal{L}(S_{val}; \\theta_t) \\;\\le\\; \\mathcal{L}(S_{val}; \\tilde{\\theta}_t) + \\gamma',
    description:
      'The constraint attached to Eq. 7. θ̃ is the counterfactual model — what the parameters would have ' +
      'been on clean batches — so stealth is measured against the honest trajectory, not against a fixed ' +
      'number. While the gap stays under γ the poisoning is invisible to anyone watching the dashboard. ' +
      'Blow the budget and the attacker stops accumulating rather than be noticed.',
    phase: 'accumulative',
    highlightElement: 'accuracy-metric',
    liveEquation: (f, cfg) => {
      if (!f || f.stealthBudgetUsed === undefined) return null;
      const g = f.stealthBudget ?? cfg.gamma ?? 0.6;
      const ok = f.stealthBudgetUsed <= g;
      return `${fmt(f.stealthBudgetUsed, 4)} \\;${ok ? '\\le' : '>'}\\; \\gamma = ${g} \\quad \\text{${ok ? 'still hidden' : 'budget blown'}}`;
    },
    readouts: (f) => !f ? [] : [
      {
        label: 'budget used',
        value: fmt(f.stealthBudgetUsed, 4),
        tone: (f.stealthBudgetUsed ?? 0) > (f.stealthBudget ?? 0.6) ? 'attack' : 'clean',
        meter: Math.min(1, Math.max(0, (f.stealthBudgetUsed ?? 0) / (f.stealthBudget ?? 0.6))),
      },
      { label: 'val acc', value: `${((f.secretAccuracy ?? 0) * 100).toFixed(1)}%`, tone: 'clean', meter: f.secretAccuracy },
      ...(f.earlyStopped ? [{ label: 'status', value: 'early-stopped', tone: 'warning' as const }] : []),
    ],
  },

  {
    id: 'trigger',
    title: 'Step 4 — Fire the trigger',
    equation: '\\theta_{T+1} = \\theta_T - \\beta \\nabla_\\theta \\mathcal{L}(\\mathcal{P}(S_T); \\theta_T)',
    description:
      'One batch. One ordinary update, indistinguishable from the thousands before it. But θ_T has been ' +
      'walked into a place where this particular gradient points away from everything the model had ' +
      'learned, and the accuracy falls off a cliff in a single step.',
    phase: 'trigger',
    visual: 'trigger',
    highlightElement: 'canvas',
    liveEquation: (f) => {
      if (!f || f.predictedLossJump === undefined) return null;
      return `\\Delta\\mathcal{L}_{\\text{predicted}} = ${fmt(f.predictedLossJump, 4)}, \\quad \\Delta\\mathcal{L}_{\\text{actual}} = ${fmt(f.actualLossJump, 4)}`;
    },
    readouts: (f) => !f ? [] : [
      { label: 'Eq. 7 forecast', value: fmt(f.predictedLossJump, 4), tone: 'neutral' },
      { label: 'actual', value: fmt(f.actualLossJump, 4), tone: 'attack' },
      { label: 'test acc', value: `${((f.poisonedAccuracy ?? 0) * 100).toFixed(1)}%`, tone: 'attack', meter: f.poisonedAccuracy },
    ],
  },

  {
    id: 'why-it-matters',
    title: 'Why accumulation is the whole trick',
    equation:
      '\\text{Acc}(\\theta_{T+1}) \\;\\lll\\; \\text{Acc}(\\theta_T) \\;\\approx\\; \\text{Acc}(\\theta_0)',
    description:
      'Run it again with λ = 0 and watch the same trigger barely register. The trigger batch is not what ' +
      'does the damage — the accumulative phase is, by quietly moving the model somewhere the trigger ' +
      'becomes lethal. That is what makes this hard to defend: every individual batch looks fine, and ' +
      'every individual metric looks fine, right up until the step that does not.',
    phase: 'trigger',
    highlightElement: 'accuracy-metric',
    readouts: (f) => !f ? [] : [
      { label: 'clean acc', value: `${((f.cleanAccuracy ?? 0) * 100).toFixed(1)}%`, tone: 'clean', meter: f.cleanAccuracy },
      { label: 'after trigger', value: `${((f.poisonedAccuracy ?? 0) * 100).toFixed(1)}%`, tone: 'attack', meter: f.poisonedAccuracy },
      {
        label: 'dropped',
        value: `${(((f.cleanAccuracy ?? 0) - (f.poisonedAccuracy ?? 0)) * 100).toFixed(1)} pts`,
        tone: 'attack',
      },
    ],
  },

  {
    id: 'caveat',
    title: 'When the attack cannot work — and why',
    description:
      'An honest caveat, and a troubleshooting guide. The paper attacks deep networks, where one gradient ' +
      'step moves millions of parameters. This runs logistic regression with three parameters, so the ' +
      'attacker has far less room. Two setups defeat it outright, and neither is a bug: a linearly ' +
      'separable dataset where the model already sits at ~100% with a wide margin (there is no point close ' +
      'enough to the boundary for a bounded step to flip), and a fully converged victim (∇L → 0, so there ' +
      'is no gradient to hijack — that is what Burn-in Epochs controls). If the trigger does nothing, try ' +
      'an overlapping dataset such as Gaussian and a lower burn-in.',
    readouts: (f, cfg) => {
      const out: ExplainerReadout[] = [
        { label: 'ε', value: `${cfg.epsilon ?? 0.8}`, tone: 'warning' },
        { label: 'burn-in', value: `${cfg.burnInEpochs ?? 20}` },
      ];
      if (f) {
        const clean = f.cleanAccuracy ?? 0;
        const drop = clean - (f.poisonedAccuracy ?? 0);
        // Diagnose the two dead regimes rather than leaving a flat line unexplained.
        if (clean >= 0.97) {
          out.push({ label: 'diagnosis', value: 'data too separable', tone: 'warning' });
        } else if (f.phase === 'trigger' && drop < 0.03) {
          out.push({ label: 'diagnosis', value: 'raise λ / lower burn-in', tone: 'warning' });
        } else if (f.phase === 'trigger') {
          out.push({ label: 'trigger drop', value: `${(drop * 100).toFixed(1)} pts`, tone: 'attack' });
        }
      }
      return out;
    },
  },
];
