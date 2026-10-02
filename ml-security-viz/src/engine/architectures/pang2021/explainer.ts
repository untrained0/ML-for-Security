/**
 * Pang et al. 2021 — guided explainer steps.
 *
 * Each card pairs the symbolic equation with the same equation evaluated on
 * whichever frame the user is scrubbed to, so the numbers on screen are the
 * model's real numbers. Scrub the timeline and the maths moves with it.
 *
 * The victim has thousands (logistic regression) to hundreds of thousands (MLP) of parameters, so the gradient fields on a frame
 * (gradVal, gradTrigger, gradBatch, gradAccum, gradTarget) are 2-D coordinates in the plane the
 * vectors span — angles and relative lengths are exact (gradient.ts: planeCoords).
 */

import type { ExplainerStep, ExplainerReadout, TraceFrame } from '../registry';

const fmt = (v: number | undefined, digits = 4) =>
  v === undefined || v === null || Number.isNaN(v) ? '—' : v.toFixed(digits);

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
    equation: 'v_{t+1} = \\mu v_t + \\nabla_\\theta \\mathcal{L}(S_t; \\theta_t), \\qquad \\theta_{t+1} = \\theta_t - \\beta\\, v_{t+1}',
    description:
      'A real-time model retrains on every batch as it arrives (Eq. 3, with the SGD momentum μ the paper ' +
      'trains with; μ = 0 is plain θ ← θ − β∇L). Here the victim is a neural network ' +
      '(or, as a linear baseline, logistic regression) over the raw pixels of real MNIST or CIFAR-10 images, burned in on clean batches first ' +
      '(θ₀). Nobody reviews S_t before it is used — it is consumed, applied, and discarded. That is the ' +
      'attack surface: the attacker does not poison a stored dataset, only sits upstream of the stream.',
    visual: 'online-step',
    highlightElement: 'canvas',
    liveEquation: (f, cfg) =>
      f ? `\\theta_{${f.iteration}} \\;\\rightarrow\\; \\theta_{${f.iteration + 1}}, \\quad \\beta = ${cfg.learningRate ?? 0.1}, \\; \\mu = ${cfg.momentum ?? 0.9}, \\; |S_t| = ${cfg.batchSize ?? 25}` : null,
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
      'So the attacker wants this alignment as negative as possible — that is Eq. 6. With gradient ' +
      'normalisation on, the alignment is the cosine of the angle instead of the raw inner product.',
    visual: 'alignment',
    liveEquation: (f, cfg) => {
      if (!f || f.gradientAlignment === undefined) return null;
      const beta = cfg.learningRate ?? 0.1;
      return (cfg.normalizeGrads ?? 'no') === 'yes'
        ? `\\cos\\angle(\\nabla\\mathcal{L}(S_{val}), \\nabla\\mathcal{L}(\\mathcal{P}(S_T))) = ${fmt(f.gradientAlignment, 4)}`
        : `\\Delta\\mathcal{L} \\approx -\\,${beta} \\times (${fmt(f.gradientAlignment, 5)}) = ${fmt(-beta * f.gradientAlignment, 5)}`;
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
    equation: 'd_t = \\hat{\\nabla}\\mathcal{L}(S_t; \\theta_t) + \\lambda\\, \\hat{G}_t, \\qquad G_t = \\nabla_\\theta \\langle \\nabla\\mathcal{L}(S_{val}; \\theta_t), \\nabla\\mathcal{L}(\\mathcal{P}(S_T); \\theta_t) \\rangle',
    description:
      'Each round the attacker picks a direction with two jobs (Eq. 9). The first term is the honest ' +
      'gradient, so the model keeps learning and the metrics keep looking normal. The second, G_t, is the ' +
      'direction in parameter space that drives the alignment down — a second-order quantity (two ' +
      'Hessian-vector products), computed exactly here. Both are unit vectors and λ sets the mix; λ = 0 is ' +
      'the control, where the batches only mimic honest updates.',
    phase: 'accumulative',
    visual: 'accumulation',
    liveEquation: (f, cfg) => {
      const ang = angleBetween(f?.gradBatch, f?.gradAccum);
      if (!f || ang === undefined) return null;
      return `\\angle(\\hat{\\nabla}\\mathcal{L}(S_t), \\hat{G}_t) = ${ang.toFixed(0)}^\\circ, \\quad \\lambda = ${cfg.lambda ?? 1}`;
    },
    readouts: (f, cfg) => !f ? [] : [
      { label: 'λ', value: `${cfg.lambda ?? 1}` },
      { label: '‖G_t‖', value: fmt(f.alignmentGradNorm, 4), tone: 'warning' },
      { label: 'drift ‖θ−θ₀‖', value: fmt(f.accumulatedDrift, 3), tone: 'warning' },
    ],
  },

  {
    id: 'perturb',
    title: 'Step 2 — Bend the batch toward that direction',
    equation:
      '\\max_{\\|\\delta\\|_\\infty \\le \\varepsilon} \\; \\langle \\hat{\\nabla}\\mathcal{L}(S_t + \\delta; \\theta_t),\\; d_t \\rangle',
    description:
      'The attacker cannot set the update directly — only the images. So they search for the pixel ' +
      'perturbation δ, at most ε grey levels per pixel, whose gradient points as close to d_t as possible. ' +
      'C steps of projected gradient ascent solve it: nudge every pixel by ±α = 2ε/C, then clip back into ' +
      'the ε-ball and [0,1]. Click a ☠ point to see the clean image, the poisoned one and δ.',
    phase: 'accumulative',
    liveEquation: (f, cfg) =>
      f ? `\\|\\delta_{${f.batchIndex ?? 0}}\\|_\\infty = ${Math.round((f.perturbationNorm ?? 0) * 255)}/255 \\;\\le\\; \\varepsilon = ${cfg.epsilon255 ?? 16}/255` : null,
    readouts: (f, cfg) => !f ? [] : [
      { label: '‖δ‖∞', value: `${Math.round((f.perturbationNorm ?? 0) * 255)}/255`, tone: 'attack', meter: Math.min(1, ((f.perturbationNorm ?? 0) * 255) / (cfg.epsilon255 ?? 16)) },
      { label: 'PGD steps', value: `${cfg.pgdSteps ?? 10}` },
      ...(f.accumulativeObjective !== undefined ? [{ label: 'H_t', value: fmt(f.accumulativeObjective, 3) }] : []),
    ],
  },

  {
    id: 'stealth',
    title: 'Step 3 — Stay inside the stealth budget',
    equation: '\\mathcal{L}(S_{val}; \\theta_t) \\;\\le\\; \\mathcal{L}(S_{val}; \\tilde{\\theta}_t) + \\gamma, \\qquad \\text{Acc}(\\theta_t) \\ge \\text{Acc}(\\theta_0) - \\Delta_{acc}',
    description:
      'The constraint attached to Eq. 7, plus the early stop of §4. θ̃ is the counterfactual model — what ' +
      'the parameters would have been on clean batches — so the loss budget γ is measured against the ' +
      'honest trajectory; test accuracy may also sink at most Δacc below the burn-in model. A round that ' +
      'would break either is fed clean instead, and the attacker stops accumulating rather than be noticed.',
    phase: 'accumulative',
    highlightElement: 'accuracy-metric',
    liveEquation: (f, cfg) => {
      if (!f || f.stealthBudgetUsed === undefined) return null;
      const g = f.stealthBudget ?? cfg.gamma ?? 0.3;
      const ok = f.stealthBudgetUsed <= g;
      return `${fmt(f.stealthBudgetUsed, 4)} \\;${ok ? '\\le' : '>'}\\; \\gamma = ${g} \\quad \\text{${ok ? 'still hidden' : 'budget blown'}}`;
    },
    readouts: (f) => !f ? [] : [
      {
        label: 'budget used',
        value: fmt(f.stealthBudgetUsed, 4),
        tone: (f.stealthBudgetUsed ?? 0) > (f.stealthBudget ?? 0.3) ? 'attack' : 'clean',
        meter: Math.min(1, Math.max(0, (f.stealthBudgetUsed ?? 0) / (f.stealthBudget ?? 0.3))),
      },
      { label: 'test acc', value: `${((f.poisonedAccuracy ?? 0) * 100).toFixed(1)}%`, tone: 'clean', meter: f.poisonedAccuracy },
      ...(f.accuracyFloor !== undefined ? [{ label: 'floor', value: `${(f.accuracyFloor * 100).toFixed(1)}%` }] : []),
      ...(f.earlyStopped ? [{ label: 'status', value: 'early-stopped', tone: 'warning' as const }] : []),
    ],
  },

  {
    id: 'trigger',
    title: 'Step 4 — Fire the trigger',
    equation: '\\theta_{T+1} = \\theta_T - \\beta \\big(\\mu v_T + \\nabla_\\theta \\mathcal{L}(\\mathcal{P}(S_T); \\theta_T)\\big)',
    description:
      'One batch. One ordinary update, indistinguishable from the ones before it. But θ_T has been walked ' +
      'into a place where this particular gradient points away from what the model had learned, so a ' +
      'single step costs far more accuracy than it would have on the honest model. With momentum the step ' +
      'also carries the velocity v_T the accumulative batches built up (larger still with the weight-' +
      'momentum trick). The first-order forecast of Eq. 6 for that step is shown next to the loss change ' +
      'it really caused.',
    phase: 'trigger',
    visual: 'trigger',
    highlightElement: 'canvas',
    liveEquation: (f) => {
      if (!f || f.predictedLossJump === undefined) return null;
      return `\\Delta\\mathcal{L}_{\\text{predicted}} = ${fmt(f.predictedLossJump, 4)}, \\quad \\Delta\\mathcal{L}_{\\text{actual}} = ${fmt(f.actualLossJump, 4)}`;
    },
    readouts: (f) => !f ? [] : [
      { label: 'Eq. 6 forecast', value: fmt(f.predictedLossJump, 4), tone: 'neutral' },
      { label: 'actual', value: fmt(f.actualLossJump, 4), tone: 'attack' },
      ...(f.velocityNorm ? [{ label: '‖v_T‖ carried', value: fmt(f.velocityNorm, 3), tone: 'warning' as const }] : []),
      { label: 'test acc', value: `${((f.poisonedAccuracy ?? 0) * 100).toFixed(1)}%`, tone: 'attack', meter: f.poisonedAccuracy },
    ],
  },

  {
    id: 'why-it-matters',
    title: 'Why accumulation is the whole trick',
    equation:
      '\\underbrace{\\text{Acc}(\\theta_T) - \\text{Acc}(\\theta_{T+1})}_{\\text{accumulated}} \\;>\\; \\underbrace{\\text{Acc}(\\tilde\\theta_T) - \\text{Acc}(\\tilde\\theta_{T+1})}_{\\text{vanilla trigger}}',
    description:
      'The paper\'s poisoning target is the single-step drop: accuracy just before the trigger minus just ' +
      'after. The comparison that isolates the accumulative phase (Table 1) is the same kind of trigger fed ' +
      'to the honest model θ̃_T that saw the same stream with no perturbations. Every accumulative batch ' +
      'looked fine and every metric looked fine, right up until the step that did not.',
    phase: 'trigger',
    highlightElement: 'accuracy-metric',
    readouts: (f) => !f ? [] : [
      ...(f.preTriggerAccuracy !== undefined ? [{
        label: 'accumulated drop',
        value: `${((f.preTriggerAccuracy - (f.poisonedAccuracy ?? 0)) * 100).toFixed(1)} pts`,
        tone: 'attack' as const,
      }] : []),
      ...(f.vanillaTriggerAccuracy !== undefined ? [{
        label: 'vanilla drop',
        value: `${(((f.cleanAccuracy ?? 0) - f.vanillaTriggerAccuracy) * 100).toFixed(1)} pts`,
        tone: 'neutral' as const,
      }] : []),
      { label: 'after trigger', value: `${((f.poisonedAccuracy ?? 0) * 100).toFixed(1)}%`, tone: 'attack', meter: f.poisonedAccuracy },
    ],
  },

  {
    id: 'caveat',
    title: 'What differs from the paper — and why',
    description:
      'Faithful: real MNIST / CIFAR-10 images, pixel-space perturbations under the paper\'s L∞ budget ' +
      '(16/255), Eq. 6–9 and Algorithm 1 with exact second-order gradients, the reference code\'s trigger ' +
      'crafting and early stop. Different: the neural-network victim is a small fully connected MLP on the ' +
      'pixels rather than a ResNet-18 (logistic regression is offered as the linear baseline), and training ' +
      'is SGD with momentum like the reference code, but without batch norm. The paper\'s own online results (Table 1, ' +
      '10-class CIFAR-10) are single-step drops of 3–11 points against 2.5 for the vanilla trigger; the ' +
      'drops to 10–30% in its Tables 3–6 are the federated setting, where the attacker edits gradients ' +
      'directly — not modelled here. At the defaults (the paper\'s momentum 0.9, optimizing P, weight ' +
      'momentum) the accumulated trigger costs ~6 points on two-class CIFAR-10 against under 1 for the ' +
      'same trigger on the honest model, and ~4 against ~1 on MNIST — the same shape as Table 1, and it ' +
      'varies from run to run. The learning rate β decides how far one batch moves the model: set it high ' +
      'and the trigger alone does most of the damage, leaving little for accumulation to add.',
    readouts: (f, cfg) => {
      const out: ExplainerReadout[] = [
        { label: 'ε', value: `${cfg.epsilon255 ?? 16}/255`, tone: 'warning' },
        { label: 'victim', value: (cfg.victim ?? 'mlp') === 'mlp' ? `MLP ${cfg.hiddenLayers ?? 1}×${cfg.hiddenUnits ?? 32}` : 'log. reg.' },
        { label: 'β', value: `${cfg.learningRate ?? 0.1}` },
        { label: 'burn-in', value: `${cfg.burnInEpochs ?? 5} ep` },
      ];
      if (f?.phase === 'trigger') {
        out.push({ label: 'rounds', value: `${f.accumulatedRounds ?? 0}${f.earlyStopped ? ' (early stop)' : ''}` });
      }
      return out;
    },
  },
];
