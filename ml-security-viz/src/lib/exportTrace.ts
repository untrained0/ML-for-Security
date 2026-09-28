/**
 * exportTrace.ts — analyst-facing export of an attack run: every frame's metrics as JSON (with
 * optional model parameters, gradients and the dataset) or as one CSV row per frame.
 *
 * Frame metrics are collected generically — every scalar field an algorithm puts on its
 * TraceFrame, plus the scalar summary of its poisoned model — so new algorithms export without
 * changes here.
 */

import { getAlgorithm } from '@/engine/architectures';
import { DATASETS } from '@/engine/data/datasets';
import { mergedConfig } from './config';

export const EXPORT_SCHEMA = 'ml-security-viz/attack-trace';
export const EXPORT_VERSION = 1;

export interface TraceExportOptions {
  poisonPoints: boolean;   // coordinates (x, y) of every poisoning point, every frame
  model: boolean;          // poisoned-model parameters every frame (w, b, θ, SV indices/α)
  gradients: boolean;      // per-point attack gradients and other vector fields
  data: boolean;           // train / validation / test sets
}

/** Frame fields that are vectors or objects, handled explicitly rather than as metrics. */
const STRUCTURED = new Set([
  'poisonX', 'poisonY', 'poisonedModel', 'poisonedRawModel', 'gradients', 'gradientNorms',
  'gainedSVs', 'lostSVs',
]);

/** What the common metric columns mean. */
export const FIELD_DESCRIPTIONS: Record<string, string> = {
  iteration: 'Frame index in the recorded trace (frames may be thinned for long runs)',
  objectiveValue: 'Attacker objective being maximised (Biggio: mean validation hinge loss; Jagielski: W_tr or W_val; Pang: attack loss)',
  heldOutObjective: 'Same loss on the attacker\'s held-out validation split (Biggio early stopping)',
  poisonedAccuracy: 'Test accuracy of the poisoned model',
  cleanAccuracy: 'Test accuracy of the clean model',
  poisonedMSE: 'Test MSE of the poisoned model',
  cleanMSE: 'Test MSE of the clean model',
  deltaW: 'Change of the weight vector (Biggio: since previous frame; Jagielski: from the clean model)',
  activePoison: 'Index of the poisoning point being optimised (sequential attacks)',
  attackStep: 'Gradient step size t',
  poisonCount: 'Number of poisoning points in the training set at this frame',
  gradientNormMean: 'Mean norm of the attack gradients over the poisoning points',
  gradientNormMax: 'Largest attack-gradient norm',
  gainedSVCount: 'Support vectors gained since the previous frame',
  lostSVCount: 'Support vectors lost since the previous frame',
};

const isScalar = (v: unknown): v is number | string | boolean =>
  typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean';

const isNumericArray = (v: unknown): v is number[] =>
  Array.isArray(v) && v.length > 0 && typeof v[0] === 'number';

function frameMetrics(f: any): Record<string, number | string | boolean> {
  const m: Record<string, number | string | boolean> = {};
  for (const [k, v] of Object.entries(f)) if (!STRUCTURED.has(k) && isScalar(v)) m[k] = v;
  m.poisonCount = f.poisonX?.length ?? 0;
  const norms: number[] = f.gradientNorms ?? [];
  if (norms.length) {
    m.gradientNormMean = norms.reduce((s, v) => s + v, 0) / norms.length;
    m.gradientNormMax = Math.max(...norms);
  }
  if (f.gainedSVs) m.gainedSVCount = f.gainedSVs.length;
  if (f.lostSVs) m.lostSVCount = f.lostSVs.length;
  for (const [k, v] of Object.entries(f.poisonedModel ?? {})) {
    if (typeof v === 'number') m[`model.${k}`] = v;
  }
  return m;
}

/** Model parameters without the heavy duplicated data (SV points are indices into train+poison). */
function modelParameters(model: any) {
  if (!model) return null;
  const out: Record<string, any> = {};
  for (const [k, v] of Object.entries(model)) {
    if (k === 'rawModel' || k === 'kernelFn') continue;
    if (k === 'supportVectors' && Array.isArray(v)) {
      out.supportVectors = v.map((sv: any) => ({ index: sv.index, label: sv.label, alpha: sv.alpha }));
    } else if (isScalar(v) || isNumericArray(v) || (v && typeof v === 'object' && !Array.isArray(v))) {
      out[k] = v;
    }
  }
  return out;
}

function vectorFields(f: any) {
  const out: Record<string, number[]> = {};
  for (const [k, v] of Object.entries(f)) if (!STRUCTURED.has(k) && isNumericArray(v)) out[k] = v;
  return out;
}

export function buildTraceExport(state: any, opts: TraceExportOptions) {
  const alg = getAlgorithm(state.activeAlgorithm);
  const ds = state.dataset;
  const meta = state.attackMeta;
  const datasetKey = meta?.datasetKey ?? state.datasetKey;
  const entry = DATASETS[datasetKey] ?? {};
  const trace: any[] = state.attackTrace;
  const isRegression = alg.modelType === 'regression';
  const clean = state.cleanModel;

  const metricKey = isRegression ? 'poisonedMSE' : 'poisonedAccuracy';
  const values = trace.map(f => f[metricKey]).filter((v: any) => typeof v === 'number');
  const worst = values.length
    ? trace.reduce((best, f, i) => {
        const v = f[metricKey];
        const worse = isRegression ? v > best.value : v < best.value;
        return typeof v === 'number' && worse ? { value: v, frame: i } : best;
      }, { value: values[0], frame: 0 })
    : null;

  const cleanMetrics: Record<string, any> = {};
  for (const [k, v] of Object.entries(clean ?? {})) if (typeof v === 'number') cleanMetrics[k] = v;

  return {
    schema: EXPORT_SCHEMA,
    version: EXPORT_VERSION,
    exportedAt: new Date().toISOString(),
    attackStartedAt: meta?.startedAt ?? null,
    algorithm: { key: alg.key, name: alg.name, paper: alg.paper, reference: alg.paperShort, modelType: alg.modelType },
    dataset: {
      key: datasetKey,
      name: entry.name ?? datasetKey,
      description: entry.desc ?? null,
      sizes: { train: ds?.train?.X?.length ?? 0, valid: ds?.valid?.X?.length ?? 0, test: ds?.test?.X?.length ?? 0 },
      features: ds?.train?.X?.[0]?.length ?? 0,
      featureNames: ds?.featureNames ?? null,
      classNames: ds?.classNames ?? null,
      bounds: ds?.bounds ?? null,
      canvasProjection: ds?.view ? `PCA to ${ds.view.components.length}-D (display only; models use all features)` : null,
    },
    config: meta?.config ?? mergedConfig(alg, state),
    resolved: {
      kernel: clean?.kernel ? (clean.kernel.type === 'linear' ? { type: 'linear' } : clean.kernel) : null,
      lambda: isRegression ? clean?.lambda ?? null : undefined,
    },
    clean: { metrics: cleanMetrics, parameters: opts.model ? modelParameters(clean) : undefined },
    summary: {
      frames: trace.length,
      poisonPoints: trace.at(-1)?.poisonX?.length ?? 0,
      metric: metricKey,
      clean: isRegression ? clean?.testMSE : clean?.testAccuracy,
      final: trace.at(-1)?.[metricKey] ?? null,
      worst,
    },
    fieldDescriptions: FIELD_DESCRIPTIONS,
    frames: trace.map(f => ({
      iteration: f.iteration,
      metrics: frameMetrics(f),
      poisonPoints: opts.poisonPoints
        ? (f.poisonX ?? []).map((x: number[], i: number) => ({ index: i, x, y: f.poisonY?.[i] }))
        : undefined,
      gradients: opts.gradients ? f.gradients : undefined,
      vectors: opts.gradients ? vectorFields(f) : undefined,
      model: opts.model ? modelParameters(f.poisonedModel) : undefined,
    })),
    data: opts.data && ds
      ? Object.fromEntries(['train', 'valid', 'test'].map(s => [s, {
          X: ds[s]?.X ?? [], y: ds[s]?.y ?? [], labels: ds[s]?.labels,
        }]))
      : undefined,
  };
}

/** One row per frame, one column per metric (union over frames, in first-seen order). */
export function traceToCSV(state: any): string {
  const rows = (state.attackTrace as any[]).map(frameMetrics);
  const cols: string[] = [];
  for (const r of rows) for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
  const cell = (v: unknown) => {
    if (v === undefined || v === null) return '';
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(','), ...rows.map(r => cols.map(c => cell(r[c])).join(','))].join('\n') + '\n';
}

/** Rough size of the optional parts, to warn before a very large download. */
export function estimateExportSize(state: any, opts: TraceExportOptions) {
  const trace: any[] = state.attackTrace;
  const d = state.dataset?.train?.X?.[0]?.length ?? 2;
  const bytesPerNumber = 12;
  let n = trace.length * 30;
  if (opts.poisonPoints) n += trace.reduce((s, f) => s + (f.poisonX?.length ?? 0) * (d + 1), 0);
  if (opts.gradients) n += trace.reduce((s, f) => s + (f.gradients?.length ?? 0) * d, 0);
  if (opts.model) n += trace.length * (d + 100);
  if (opts.data && state.dataset) {
    n += ['train', 'valid', 'test'].reduce((s, k) => s + (state.dataset[k]?.X?.length ?? 0) * (d + 1), 0);
  }
  return n * bytesPerNumber;
}
