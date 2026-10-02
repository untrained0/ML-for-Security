/**
 * Pang et al. 2021 — "Accumulative Poisoning Attacks on Real-time Data"
 * (NeurIPS 2021, arXiv:2106.09993; reference code github.com/ShawnXYang/AccumulativeAttack)
 *
 * Online-learning setting (§3.2, Algorithm 1) on real MNIST / CIFAR-10 in pixel space: the
 * victim is a neural network (an MLP with one to three hidden layers) or logistic regression over
 * the raw pixels (model.ts), the perturbations are L∞-bounded in [0,1] pixel units exactly as in
 * the paper, and all gradients — including the second-order accumulation term G_t — are exact
 * (backprop and the R-operator; gradient.ts).
 *
 * Federated setting (§3.3, Algorithm 2; federated.ts): the poisoners submit gradients to the server,
 * bounded only by its clipping, accumulating λ·∇_θ cos(σ∇L(S_T), ∇L(S_val)) before a clean or reversed trigger.
 */

import { AlgorithmModule, registerAlgorithm } from '../registry';
import { burnIn, createNet, fitNorm, getModelState, logit, predictProb, type Activation, type PackedNet, packNet } from './model';
import { runAccumulativeAttack } from './attack';
import { runFederatedAttack } from './federated';
import { pangFederatedExplainerSteps } from './federatedExplainer';
import { pang2021ExplainerSteps } from './explainer';
import { pangPointDetails } from './inspector';
import { pangMathPanel } from './panel';
import { pangTourSteps } from './tour';

const pang2021: AlgorithmModule = {
  key: 'pang2021',
  name: 'Accumulative Poisoning',
  paper: 'https://arxiv.org/abs/2106.09993',
  paperShort: 'Pang et al. 2021',
  modelType: 'classification',
  modelLabel: (config) => `${(config.setting ?? 'online') === 'federated' ? 'Federated' : 'Online'} ${(config.victim ?? 'mlp') === 'mlp' ? 'Neural Network' : 'Logistic Regression'}`,

  datasets: ['pangMnist35', 'pangMnist71', 'pangCifarShipFrog', 'pangCifarPlaneBird'],

  configSchema: [
    // Model: the online victim and its burn-in (§4)
    {
      key: 'victim', label: 'Victim Model', type: 'select', section: 'model',
      options: [
        { value: 'mlp', label: 'Neural network (MLP)' },
        { value: 'logreg', label: 'Logistic regression' },
      ],
      tooltip: 'The online learner under attack. The paper attacks a neural network (ResNet-18); here a fully connected MLP on the raw pixels, trained with the same online SGD. Logistic regression is the linear baseline: far less room to steer, so far smaller drops',
    },
    {
      key: 'hiddenUnits', label: 'Hidden Units', type: 'range', section: 'model',
      min: 16, max: 256, step: 16, showIf: { key: 'victim', equals: 'mlp' },
      tooltip: 'Width of each hidden layer. Cost grows linearly with it (a CIFAR-10 image has 3072 inputs)',
    },
    {
      key: 'hiddenLayers', label: 'Hidden Layers', type: 'range', section: 'model',
      min: 1, max: 3, step: 1, showIf: { key: 'victim', equals: 'mlp' },
      tooltip: 'Depth of the MLP',
    },
    {
      key: 'activation', label: 'Activation', type: 'select', section: 'model', showIf: { key: 'victim', equals: 'mlp' },
      options: [
        { value: 'relu', label: 'ReLU (as in ResNet)' },
        { value: 'tanh', label: 'tanh (smooth)' },
      ],
      tooltip: 'Hidden-unit non-linearity. ReLU matches the paper\'s ResNet; tanh is twice differentiable everywhere, so its curvature also enters the Hessian-vector products of G_t',
    },
    {
      key: 'learningRate', label: 'Learning Rate (β)', type: 'range', section: 'model',
      min: 0.01, max: 20, step: 0.01,
      tooltip: 'Online SGD rate β (Eq. 3). Inputs (and hidden units) are scaled by 1/√fan-in, so β means the same for both victims. With momentum μ the long-run step is β/(1 − μ). A faster learner reacts more to a single batch: set it high and the poisoned trigger alone does most of the damage, leaving little for accumulation to add',
    },
    {
      key: 'momentum', label: 'Momentum (μ)', type: 'range', section: 'model',
      min: 0, max: 0.95, step: 0.05,
      tooltip: 'SGD momentum of the victim, v ← μv + ∇L, θ ← θ − βv (the paper and reference code: 0.9, with weight decay 1e-4 during burn-in). The online optimiser starts fresh at θ_0 and its buffer is carried into the trigger step. 0 = plain SGD, Eq. 3',
    },
    {
      key: 'batchSize', label: 'Batch Size', type: 'range', section: 'model',
      min: 10, max: 100, step: 5,
      tooltip: 'Images per online batch S_t, for burn-in, accumulation and the trigger alike (the paper uses 100)',
    },
    {
      key: 'burnInEpochs', label: 'Burn-in Epochs', type: 'range', section: 'model',
      min: 1, max: 40, step: 1,
      tooltip: 'Clean SGD epochs before the attacker arrives, giving θ_0 (§4: 10 on MNIST, 40 on CIFAR-10)',
    },
    // Attack: the threat model, then Algorithm 1 (online) or Algorithm 2 (federated)
    {
      key: 'setting', label: 'Setting', type: 'select', section: 'attack',
      options: [
        { value: 'online', label: 'Online — poisoned images (Alg. 1)' },
        { value: 'federated', label: 'Federated — poisoned gradients (Alg. 2)' },
      ],
      tooltip: 'Online (§3.2): the attacker perturbs the pixels of each incoming batch within ε. Federated (§3.3): the poisoners submit gradients to the server directly, bounded only by the server\'s clipping — the setting of the paper\'s Tables 3–6, where the large drops are',
    },
    {
      key: 'numAccumBatches', label: 'Accumulative Rounds (T)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      min: 1, max: 100, step: 1,
      tooltip: 'Online batches the attacker perturbs before releasing the trigger at round T',
    },
    {
      key: 'epsilon255', label: 'Perturbation Budget (ε × 255)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      min: 1, max: 32, step: 1,
      tooltip: 'L∞ bound on every pixel change, ‖δ‖∞ ≤ ε, in 8-bit grey levels (the paper uses 8/255 and 16/255). Pixels also stay in [0,1]',
    },
    {
      key: 'pgdSteps', label: 'PGD Steps (C)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      min: 1, max: 100, step: 1,
      tooltip: 'Signed-gradient steps per crafted batch, step size α = 2ε/C (§4; the paper uses C = 100). Cost per round grows linearly with C',
    },
    {
      key: 'lambda', label: 'Accumulation Weight (λ)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      min: 0, max: 10, step: 0.5,
      tooltip: 'Eq. 9 trade-off: A_t(S_t) is aligned with ĝ(S_t) + λĜ_t — keep learning vs. accumulate. λ = 0 drops the G_t term (the control), though with weight momentum on the momentum buffer still accumulates the perturbed updates',
    },
    {
      key: 'normalizeGrads', label: 'Gradient Normalisation', type: 'select', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      options: [
        { value: 'no', label: 'Off — raw inner product (Eq. 7)' },
        { value: 'yes', label: 'On — cosine (reference code)' },
      ],
      tooltip: 'Algorithm 1 optionally normalises ∇L(S_val), ∇L(S_T) and ∇L(S_t) "to concentrate on angular distances". Off keeps Eq. 7\'s inner product, which also rewards a larger trigger gradient — clearly stronger here, on both victims',
    },
    {
      key: 'weightMomentum', label: 'Weight Momentum', type: 'select', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      options: [
        { value: 'no', label: 'Off — victim keeps μ' },
        { value: 'yes', label: 'On — μ = 1.1 while accumulating' },
      ],
      tooltip: '§4.1 / Table 1 trick: "lightly increases the momentum factor as 1.1 (from 0.9 by default) of the SGD optimizer in accumulative phase", so the poisoned updates pile up in the momentum buffer the trigger step inherits. Note it assumes the attacker can also set the victim optimiser\'s momentum',
    },
    {
      key: 'triggerType', label: 'Trigger Batch', type: 'select', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      options: [
        { value: 'poisoned', label: 'Poisoned P(S_T) (Eq. 6)' },
        { value: 'clean', label: 'Clean S_T' },
      ],
      tooltip: 'Table 1 uses a poisoned trigger; Table 6 shows the accumulative phase also primes the model for a clean one',
    },
    {
      key: 'optimizeTrigger', label: 'Optimise P During Accumulation', type: 'select', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      options: [
        { value: 'yes', label: 'Yes — re-craft each round' },
        { value: 'no', label: 'No — fixed at θ₀' },
      ],
      tooltip: '"Optimizing P" in Table 1: re-craft the poisoned trigger against every new θ_t instead of fixing it at the burn-in model',
    },
    {
      key: 'gamma', label: 'Loss Tolerance (γ)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      min: 0.01, max: 2, step: 0.01,
      tooltip: 'Eq. 7 constraint: L(S_val; A(θ)) ≤ L(S_val; θ̃) + γ against the clean trajectory θ̃. A round that would break it is fed clean and accumulation stops',
    },
    {
      key: 'serverLearningRate', label: 'Server Learning Rate (β)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      min: 0.1, max: 5, step: 0.1,
      tooltip: 'Rate of the server\'s SGD-momentum step on each aggregate (Eq. 11) after burn-in. The trigger does its damage in one step of size β·‖update‖, so a larger β gives both the trigger and Table 3\'s direct poisoner more leverage',
    },
    {
      key: 'federatedRounds', label: 'Accumulative Rounds (T)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      min: 10, max: 1000, step: 10,
      tooltip: 'Server rounds the poisoners accumulate for before the trigger (the paper: 50–1000, Tables 3–6). One Hessian-vector product through S_val per round',
    },
    {
      key: 'federatedLambda', label: 'Accumulation Scale (λ)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      min: 0.005, max: 0.3, step: 0.005,
      tooltip: 'Eq. 12: the submitted aggregate is λ·∇_θ cos(σ∇L(S_T), ∇L(S_val)) (plus the honest gradients if kept). The paper\'s "loss scaling" for the accumulative phase: 0.01–0.08. Small λ keeps the updates small enough to pass clipping',
    },
    {
      key: 'federatedAlign', label: 'Alignment', type: 'select', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      options: [
        { value: 'cosine', label: 'Cosine (reference code)' },
        { value: 'inner', label: 'Raw inner product (Eq. 10)' },
      ],
      tooltip: 'What the λ-term drives down. The reference code normalises both gradients (cosine), which only aims the trigger gradient; Eq. 10\'s raw inner product is the first-order loss increase of the trigger step itself, so it also rewards a larger trigger gradient. λ is on a different scale for each',
    },
    {
      key: 'federatedHonest', label: 'Honest Gradients in Each Round', type: 'select', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      options: [
        { value: 'no', label: 'No — λ-term only (reference code)' },
        { value: 'yes', label: 'Yes — ΣG_t + λ-term (Algorithm 2)' },
      ],
      tooltip: 'Algorithm 2\'s H_t keeps the clients\' honest gradients, so the model keeps training; the authors\' code (feder_accu_train.py), which produced Tables 3–6, submits only the λ-term',
    },
    {
      key: 'federatedTrigger', label: 'Trigger', type: 'select', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      options: [
        { value: 'clean', label: 'Clean S_T (Table 3)' },
        { value: 'poisoned', label: 'Reversed −s∇L(S_T) (Eq. 13, Table 6)' },
      ],
      tooltip: 'Clean: the trigger round is an ordinary honest update on S_T. Poisoned: the reverse trigger of Eq. 13, P = −s·∇L(S_T), accumulated against cos(−∇L(S_T), ∇L(S_val))',
    },
    {
      key: 'triggerScale', label: 'Reversed Trigger Scale (s)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      min: 0.01, max: 2, step: 0.01,
      tooltip: 'Size of the reversed trigger P = −s∇L(S_T) (the reference\'s poisoned_trigger_step). Only used with the reversed trigger',
    },
    {
      key: 'clipNorm', label: 'Server Gradient Clipping', type: 'select', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      options: [
        { value: 'none', label: 'None' },
        { value: 'l2', label: 'ℓ2 norm' },
        { value: 'linf', label: 'ℓ∞ norm' },
      ],
      tooltip: 'The defence of Tables 3–5: the server rescales every aggregate g so ‖g‖ ≤ c (torch clip_grad_norm_). It stops large malicious updates; the accumulative phase sends small ones',
    },
    {
      key: 'clipValue', label: 'Clip Bound (c)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      min: 0.01, max: 10, step: 0.01,
      tooltip: 'Norm bound c of the server\'s clipping (the paper: 10, 1, 0.1)',
    },
    {
      key: 'directLossScale', label: 'Direct Poisoner Scale (s_d)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      min: 1, max: 50, step: 1,
      tooltip: 'Table 3\'s "poisoned trigger" baseline, reported on the trigger frame: one direct update −s_d·∇L(S_val) on the honest model θ̃_T (the paper: 1, 10, 20, 50). Clipping is what stops it',
    },
    {
      key: 'federatedMaxAccDrop', label: 'Monitor Tolerance (Δacc)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'federated' },
      min: 0.01, max: 1, step: 0.01,
      tooltip: 'Early stop as in §4: a round that would take test accuracy more than Δacc below θ₀ is sent honest and accumulation ends. 1 = no monitor, as in the reference code\'s federated runs behind Tables 3–6 (its threshold never fires); Fig. 3b reports the pre-trigger accuracy instead',
    },
    {
      key: 'maxAccDrop', label: 'Monitor Tolerance (Δacc)', type: 'range', section: 'attack', showIf: { key: 'setting', equals: 'online' },
      min: 0.01, max: 0.2, step: 0.01,
      tooltip: '§4 early stop: accumulation halts before test accuracy falls more than this below the burn-in model (the paper stops at ~80% from ~83% on CIFAR-10)',
    },
  ],

  // The paper's online setup (§4, reference online_accu_train.py): SGD momentum 0.9, a poisoned
  // trigger re-crafted each round ("optimizing P") and the weight-momentum trick — Table 1's best
  // row. β, B, T, C and the width are scaled to a small network and a trace the browser can hold;
  // tuned on real MNIST 3-vs-5 / CIFAR-10 ship-vs-frog (runs in docs/memory.md).
  defaultConfig: {
    setting: 'online',
    victim: 'mlp',
    hiddenUnits: 32,
    hiddenLayers: 1,
    activation: 'relu',
    learningRate: 0.3,
    momentum: 0.9,
    batchSize: 25,
    burnInEpochs: 5,
    numAccumBatches: 40,
    epsilon255: 16,
    pgdSteps: 10,
    lambda: 1,
    weightMomentum: 'yes',
    normalizeGrads: 'no',
    triggerType: 'poisoned',
    optimizeTrigger: 'yes',
    gamma: 0.3,
    maxAccDrop: 0.05,
    // Federated (Algorithm 2)
    federatedRounds: 300,
    serverLearningRate: 1,
    federatedLambda: 0.3,
    federatedAlign: 'inner',
    federatedHonest: 'no',
    federatedTrigger: 'clean',
    triggerScale: 0.1,
    clipNorm: 'none',
    clipValue: 1,
    directLossScale: 10,
    federatedMaxAccDrop: 1,
  },

  trainClean(dataset, config) {
    const { X, y } = dataset.train;
    const norm = fitNorm(X, dataset.imageShape?.[2] ?? 1);
    const hidden = (config.victim ?? 'mlp') === 'mlp'
      ? Array.from({ length: Math.max(1, Math.round(config.hiddenLayers ?? 1)) }, () => Math.max(1, Math.round(config.hiddenUnits ?? 32)))
      : [];
    const init = createNet([X[0].length, ...hidden, 1], (config.activation ?? 'relu') as Activation, norm);
    // Burn-in optimiser as the reference train_cifar.py: SGD, momentum 0.9, weight decay 1e-4
    const opt = { lr: config.learningRate ?? 0.1, momentum: config.momentum ?? 0.9, weightDecay: 1e-4 };
    const model = burnIn(X, y, init, opt, config.burnInEpochs ?? 5, config.batchSize ?? 25);
    const testY = dataset.test.Y || dataset.test.y;
    return { modelState: getModelState(model, X, y, dataset.test.X, testY), rawModel: packNet(model) };
  },

  runAttack: (dataset, cleanModelState, config, ...rest) =>
    ((config.setting ?? 'online') === 'federated' ? runFederatedAttack : runAccumulativeAttack)(dataset, cleanModelState, config, ...rest),

  predict(rawModel: PackedNet, x) {
    return logit(rawModel, x) >= 0 ? 1 : -1;
  },

  // Smooth boundary for the canvas: P(y=+1|x) − 0.5 changes sign exactly where predict does
  score(rawModel: PackedNet, x) {
    return predictProb(rawModel, x) - 0.5;
  },

  explainerSteps: (config) => ((config.setting ?? 'online') === 'federated' ? pangFederatedExplainerSteps : pang2021ExplainerSteps),
  pointDetails: pangPointDetails,
  mathPanel: pangMathPanel,
  tourSteps: pangTourSteps,
};

registerAlgorithm(pang2021);
export default pang2021;
