/**
 * Pang et al. 2021 — guided explainer for the federated setting (§3.3, Algorithm 2, §4.2).
 *
 * Same contract as explainer.ts: every card's live equation and readouts are evaluated on the frame
 * the user is scrubbed to. gradVal / gradTrigger are ∇L(S_val) and σ∇L(S_T) (on the trigger frame,
 * the update actually taken) as exact 2-D coordinates in the plane they span.
 */

import type { ExplainerStep, ExplainerReadout } from '../registry';

const fmt = (v: number | undefined, digits = 4) =>
  v === undefined || v === null || Number.isNaN(v) ? '—' : v.toFixed(digits);
const pct = (v: number | undefined) => (v === undefined ? '—' : `${(v * 100).toFixed(1)}%`);

export const pangFederatedExplainerSteps: ExplainerStep[] = [
  {
    id: 'fed-setting',
    title: 'The setting: a server that trusts its clients',
    equation: '\\theta_{t+1} = \\theta_t - \\beta \\sum_{n} G_t^n \\qquad \\text{(Eq. 11, with SGD momentum)}',
    description:
      'In federated learning the server never sees images, only the gradients G_t^n its clients compute on ' +
      'their own data, which it adds up and applies. So the poisoners need not disguise images at all: they ' +
      'submit gradients. The only thing standing in their way is what the server does to each aggregate — ' +
      'typically clipping its norm. With the recovered offset of Eq. 14 the poisoners can set the whole ' +
      'aggregate, so each round here is one submitted update.',
    highlightElement: 'canvas',
    liveEquation: (f, cfg) =>
      f ? `t = ${f.batchIndex ?? 0}, \\quad \\beta = ${cfg.serverLearningRate ?? 1}, \\; \\mu = ${cfg.momentum ?? 0.9}` : null,
    readouts: (f) => !f ? [] : [
      { label: 'round', value: `${f.batchIndex ?? 0}` },
      { label: 'test acc', value: pct(f.poisonedAccuracy), tone: 'clean', meter: f.poisonedAccuracy },
    ],
  },

  {
    id: 'fed-objective',
    title: 'The goal: make the trigger point the wrong way',
    equation: '\\min_{\\mathcal{A}} \\; \\cos\\big(\\sigma\\nabla\\mathcal{L}(S_T; \\mathcal{A}(\\theta_T)),\\; \\nabla\\mathcal{L}(S_{val}; \\mathcal{A}(\\theta_T))\\big)',
    description:
      'Eq. 10, normalised as in the reference code: move the model to where the trigger batch\'s gradient ' +
      'is anti-aligned with the validation gradient, so that one ordinary step on the trigger raises the ' +
      'validation loss instead of lowering it. σ = +1 for a clean trigger; σ = −1 for the reversed trigger of ' +
      'Eq. 13, whose update is −s∇L(S_T).',
    visual: 'alignment',
    liveEquation: (f) => (f && f.gradientAlignment !== undefined ? `\\cos = ${fmt(f.gradientAlignment, 4)}` : null),
    readouts: (f) => !f ? [] : [
      { label: 'cos', value: fmt(f.gradientAlignment, 4), tone: (f.gradientAlignment ?? 0) < 0 ? 'attack' : 'clean' },
    ],
  },

  {
    id: 'fed-accumulate',
    title: 'Accumulate with small updates',
    equation: '\\sum_n \\mathcal{A}_t(G_t^n) = w \\sum_n G_t^n + \\lambda\\, \\nabla_\\theta \\cos\\big(\\sigma\\nabla\\mathcal{L}(S_T; \\theta_t), \\nabla\\mathcal{L}(S_{val}; \\theta_t)\\big)',
    description:
      'Each round the poisoners submit the gradient of that cosine — a second-order quantity (two ' +
      'Hessian-vector products, computed exactly here) — scaled by λ, the paper\'s "loss scaling" (0.01–0.08). ' +
      'Algorithm 2 adds the honest gradients (w = 1); the authors\' code, which produced Tables 3–6, sends only ' +
      'the λ-term (w = 0). The updates are small, so clipping barely touches them, yet over hundreds of rounds ' +
      'they walk the model somewhere fragile.',
    phase: 'accumulative',
    liveEquation: (f, cfg) =>
      f && f.updateNorm !== undefined ? `\\|\\textstyle\\sum \\mathcal{A}_t\\| = ${fmt(f.updateNorm, 4)}, \\quad \\lambda = ${cfg.federatedLambda ?? 0.3}` : null,
    readouts: (f) => !f ? [] : [
      { label: '‖update‖', value: fmt(f.updateNorm, 4), tone: 'warning' },
      { label: '‖∇cos‖', value: fmt(f.alignmentGradNorm, 4) },
      { label: 'drift ‖θ−θ₀‖', value: fmt(f.accumulatedDrift, 3), tone: 'warning' },
    ],
  },

  {
    id: 'fed-clip',
    title: 'Why clipping does not stop it',
    equation: 'g \\leftarrow g \\cdot \\min\\!\\Big(1, \\frac{c}{\\|g\\|_p}\\Big), \\qquad p \\in \\{2, \\infty\\}',
    description:
      'The server\'s defence in Tables 3–5 rescales any aggregate whose ℓ2 or ℓ∞ norm exceeds c. A direct ' +
      'poisoner has to send one huge update (−s_d∇L(S_val), s_d up to 50) to do damage in a single round, and ' +
      'clipping cuts it down to size. The accumulative updates are small to begin with, so they pass almost ' +
      'untouched; only the trigger round itself is an ordinary-sized update. The monitor meanwhile watches test ' +
      'accuracy; a round that would drop it more than Δacc below θ₀ is sent honest instead, and accumulation stops.',
    phase: 'accumulative',
    highlightElement: 'accuracy-metric',
    readouts: (f, cfg) => !f ? [] : [
      { label: 'clipping', value: (cfg.clipNorm ?? 'none') === 'none' ? 'off' : `${cfg.clipNorm === 'l2' ? 'ℓ2' : 'ℓ∞'} ≤ ${cfg.clipValue ?? 1}` },
      ...(f.clipFactor !== undefined ? [{ label: 'kept', value: `${(f.clipFactor * 100).toFixed(0)}%`, tone: (f.clipFactor < 1 ? 'warning' : 'clean') as ExplainerReadout['tone'], meter: f.clipFactor }] : []),
      { label: 'test acc', value: pct(f.poisonedAccuracy), tone: 'clean', meter: f.poisonedAccuracy },
      ...(f.accuracyFloor !== undefined ? [{ label: 'floor', value: pct(f.accuracyFloor) }] : []),
      ...(f.earlyStopped ? [{ label: 'status', value: 'early-stopped', tone: 'warning' as const }] : []),
    ],
  },

  {
    id: 'fed-trigger',
    title: 'Fire the trigger',
    equation: '\\theta_{T+1} = \\theta_T - \\beta\\big(\\mu v_T + \\text{clip}(P)\\big), \\qquad P = \\nabla\\mathcal{L}(S_T) \\;\\text{or}\\; -s\\nabla\\mathcal{L}(S_T)',
    description:
      'One round with the trigger batch — a clean one, or the reversed trigger. On the primed model this ' +
      'single update does far more damage than the same update on the honest model θ̃_T, which trained on the ' +
      'same stream without the poisoners. The first-order forecast of the loss change is shown next to what ' +
      'the step really did.',
    phase: 'trigger',
    visual: 'trigger',
    highlightElement: 'canvas',
    liveEquation: (f) =>
      f && f.predictedLossJump !== undefined
        ? `\\Delta\\mathcal{L}_{\\text{predicted}} = ${fmt(f.predictedLossJump, 4)}, \\quad \\Delta\\mathcal{L}_{\\text{actual}} = ${fmt(f.actualLossJump, 4)}`
        : null,
    readouts: (f) => !f ? [] : [
      { label: 'forecast', value: fmt(f.predictedLossJump, 4) },
      { label: 'actual', value: fmt(f.actualLossJump, 4), tone: 'attack' },
      { label: 'test acc', value: pct(f.poisonedAccuracy), tone: 'attack', meter: f.poisonedAccuracy },
    ],
  },

  {
    id: 'fed-compare',
    title: 'Against the baselines',
    equation:
      '\\underbrace{\\text{Acc}(\\theta_T) - \\text{Acc}(\\theta_{T+1})}_{\\text{accumulated}} \\quad\\text{vs.}\\quad \\underbrace{\\text{Acc}(\\tilde\\theta_T) - \\text{Acc}(\\tilde\\theta_{T+1})}_{\\text{same trigger, no accumulation}}',
    description:
      'The single-step drop is the paper\'s target. Two baselines run on the honest model θ̃_T: the same ' +
      'trigger without accumulation, and Table 3\'s direct poisoner, one update −s_d∇L(S_val). Turn on clipping ' +
      'to see the paper\'s point: clipping neutralises the direct poisoner, far less so the accumulated trigger.',
    phase: 'trigger',
    highlightElement: 'accuracy-metric',
    readouts: (f) => !f ? [] : [
      ...(f.preTriggerAccuracy !== undefined ? [{ label: 'accumulated drop', value: `${((f.preTriggerAccuracy - (f.poisonedAccuracy ?? 0)) * 100).toFixed(1)} pts`, tone: 'attack' as const }] : []),
      ...(f.vanillaTriggerAccuracy !== undefined ? [{ label: 'no accumulation', value: `${(((f.cleanAccuracy ?? 0) - f.vanillaTriggerAccuracy) * 100).toFixed(1)} pts` }] : []),
      ...(f.directAttackAccuracy !== undefined ? [{ label: 'direct poisoner', value: `${(((f.cleanAccuracy ?? 0) - f.directAttackAccuracy) * 100).toFixed(1)} pts` }] : []),
    ],
  },

  {
    id: 'fed-caveat',
    title: 'What differs from the paper — and why',
    description:
      'Faithful: Algorithm 2 with the recovered offset (one aggregate per round), the reference code\'s cosine ' +
      'objective, reversed trigger, SGD momentum carried into the trigger step and clip_grad_norm_ clipping, with ' +
      'exact second-order gradients. Different: the victim is a small MLP (or logistic regression) on a two-class ' +
      'task, so chance level is 50%, not the 10% of the paper\'s 10-class CIFAR-10 ResNet-18; S_val is a held-out ' +
      'training split; the monitor is checked every round rather than every 10 (off by default, as in the reference ' +
      'runs). Measured over 8 seeds at the defaults (Eq. 10 inner product, λ = 0.3, 300 rounds, server β = 1): the ' +
      'accumulated clean trigger costs 3.0 points on MNIST and 0.6 on CIFAR-10 against ~0 without accumulation (7/8 ' +
      'seeds each), with accuracy sagging ~10 and ~4 points before the trigger. The paper\'s collapse to 11–34% ' +
      'does not reproduce on this victim: one server step on a small MLP moves it too little, however well aimed. ' +
      'Clipping does neutralise the direct poisoner (25–40 points unclipped at s_d = 50, ~4 under ℓ2 ≤ 1), as in Table 3.',
    readouts: (f, cfg) => {
      const out: ExplainerReadout[] = [
        { label: 'λ', value: `${cfg.federatedLambda ?? 0.3}` },
        { label: 'T', value: `${cfg.federatedRounds ?? 300}` },
        { label: 'trigger', value: (cfg.federatedTrigger ?? 'clean') === 'clean' ? 'clean' : `reversed, s = ${cfg.triggerScale ?? 0.1}` },
      ];
      if (f?.phase === 'trigger') out.push({ label: 'rounds', value: `${f.accumulatedRounds ?? 0}${f.earlyStopped ? ' (early stop)' : ''}` });
      return out;
    },
  },
];
