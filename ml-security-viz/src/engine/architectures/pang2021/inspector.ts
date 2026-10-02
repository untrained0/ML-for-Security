/**
 * Pang et al. 2021 — what the point inspector shows for a selected point.
 *
 * A poison point here is a real image perturbed within ‖δ‖∞ ≤ ε and fed with its true label, so
 * the useful comparison is the clean image against its poisoned version under the model that
 * batch was fed to (θ_t, the previous frame's model). A training point is compared under the
 * burn-in model θ₀ and the model on screen.
 */

import type { PointContext, PointDetails, PointMetric } from '../registry';
import { type PackedNet, logit, logisticLoss, sigmoid } from './model';

/** Margin, P(correct) and loss of an image under a model, for the two columns. */
function metrics(y: number, clean: { m: PackedNet; x: number[] } | null, poisoned: { m: PackedNet; x: number[] }): PointMetric[] {
  const margin = (m: PackedNet, x: number[]) => (y > 0 ? 1 : -1) * logit(m, x);
  const mc = clean ? margin(clean.m, clean.x) : undefined;
  const mp = margin(poisoned.m, poisoned.x);
  const on = (f: (v: number) => number) => (mc === undefined ? undefined : f(mc));
  return [
    {
      label: 'Margin y·z', tooltip: 'y·z(x), the label times the model\'s logit — positive when the image is classified correctly',
      clean: mc, poisoned: mp, digits: 3, worse: 'lower',
    },
    {
      label: 'P(correct)', tooltip: 'σ(y·z), the probability the model gives the true label',
      clean: on(sigmoid), poisoned: sigmoid(mp), digits: 3, worse: 'lower',
    },
    {
      label: 'Loss ℓ', tooltip: 'Logistic loss log(1 + e^{−y·z}), what the online update descends',
      clean: on(logisticLoss), poisoned: logisticLoss(mp), digits: 3, worse: 'higher',
    },
  ];
}

export function pangPointDetails({ kind, index, x, y, frame, prevFrame, cleanRawModel, dataset }: PointContext): PointDetails {
  if (kind === 'clean') {
    const base: PackedNet | undefined = cleanRawModel ?? undefined;
    const current: PackedNet | undefined = frame?.poisonedRawModel;
    const fedAt = frame?.poisonSource?.indexOf(index) ?? -1;
    return {
      columns: ['Clean θ₀', current ? 'Poisoned θ' : 'Clean θ₀'],
      rows: base && current ? metrics(y, { m: base, x }, { m: current, x })
        : base ? metrics(y, null, { m: base, x })
        : [],
      extras: frame?.poisonSource
        ? [{ label: "In this frame's batch", value: fedAt >= 0 ? `yes — poison point #${fedAt}` : 'no' }]
        : [],
      note: !base ? 'Train the clean model to see its predictions.'
        : current ? `θ₀ is the burn-in model; poisoned θ is the model after frame ${frame.iteration}.`
        : 'θ₀ is the burn-in model.',
    };
  }

  // A poison point: the batch it belongs to was crafted against, and fed to, the previous frame's model
  const fedTo: PackedNet | undefined = prevFrame?.poisonedRawModel ?? frame?.poisonedRawModel;
  const src = frame?.poisonSource?.[index];
  const x0: number[] | null = src !== undefined ? dataset.train.X[src] : null;
  const extras: PointDetails['extras'] = [];

  if (x0) {
    let linf = 0, l2 = 0;
    for (let j = 0; j < x.length; j++) {
      const d = x[j] - x0[j];
      linf = Math.max(linf, Math.abs(d));
      l2 += d * d;
    }
    extras.push(
      { label: '‖δ‖∞', value: `${(linf * 255).toFixed(1)}/255`, tooltip: 'Largest change to any pixel, bounded by ε' },
      { label: '‖δ‖₂', value: Math.sqrt(l2).toFixed(3), tooltip: 'Euclidean size of the perturbation, pixels in [0,1]' },
      { label: 'Source image', value: `training #${src}` },
    );
  }
  const g = frame?.gradientNorms?.[index];
  if (g !== undefined) {
    extras.push({ label: '‖∇ₓH_t‖', value: g.toPrecision(3), tooltip: "Size of this image's pixel gradient of the Eq. 9 objective at the last PGD step" });
  }

  const round = frame?.batchIndex ?? 0;
  const note = frame?.phase === 'trigger'
    ? 'Both columns use θ_T, the model the trigger batch is fed to.'
    : frame?.earlyStopped && !frame.perturbationNorm
      ? `Early stop: round ${round} was fed unperturbed, so both images are the same.`
      : `Both columns use θ_${round}, the model this batch was crafted against and fed to.`;

  return {
    columns: ['Clean image', 'Poisoned image'],
    rows: fedTo ? metrics(y, x0 ? { m: fedTo, x: x0 } : null, { m: fedTo, x }) : [],
    extras,
    note,
  };
}
