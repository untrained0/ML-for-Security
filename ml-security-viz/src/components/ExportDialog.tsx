'use client';
import { useMemo, useState } from 'react';
import useStore from '@/store/useStore';
import { getAlgorithm } from '@/engine/architectures';
import { DATASETS } from '@/engine/data/datasets';
import { captureFrames, snapshotCanvasSVG, cssVar, type LegendItem } from '@/lib/exportCanvas';
import { buildTraceExport, traceToCSV, estimateExportSize, type TraceExportOptions } from '@/lib/exportTrace';
import { createZip, downloadBlob } from '@/lib/zip';

type FrameMode = 'current' | 'every' | 'all' | 'list';

// Development only: the export pipeline on window.__mlsvExport, for scripted checks
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  (window as any).__mlsvExport = {
    snapshotCanvasSVG, captureFrames, buildTraceExport, traceToCSV,
    state: () => useStore.getState(),
  };
}

/** "0, 10, 25-30, last" → sorted unique frame indices within [0, max]. */
export function parseFrameList(text: string, max: number): number[] {
  const out = new Set<number>();
  for (const raw of text.split(/[,\s]+/).filter(Boolean)) {
    const token = raw.toLowerCase();
    const num = (s: string) => (s === 'last' || s === 'end' ? max : parseInt(s, 10));
    const range = token.match(/^(\w+)-(\w+)$/);
    if (range) {
      const [a, b] = [num(range[1]), num(range[2])];
      if (Number.isFinite(a) && Number.isFinite(b)) for (let i = Math.min(a, b); i <= Math.max(a, b); i++) out.add(i);
    } else {
      const v = num(token);
      if (Number.isFinite(v)) out.add(v);
    }
  }
  return [...out].filter(v => v >= 0 && v <= max).sort((a, b) => a - b);
}

const fmtBytes = (b: number) =>
  b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b > 1e3 ? `${Math.round(b / 1e3)} kB` : `${b} B`;

export default function ExportDialog() {
  const state = useStore();
  const { exportDialog, setConfig, attackTrace, currentIteration, activeAlgorithm, datasetKey, dataset, cleanModel } = state;
  const alg = getAlgorithm(activeAlgorithm);
  const isRegression = alg.modelType === 'regression';
  const hasTrace = attackTrace.length > 0;
  const maxFrame = Math.max(0, attackTrace.length - 1);

  const [mode, setMode] = useState<FrameMode>('current');
  const [every, setEvery] = useState(10);
  const [list, setList] = useState('0, last');
  const [withCaption, setWithCaption] = useState(true);
  const [withLegend, setWithLegend] = useState(true);
  const [opts, setOpts] = useState<TraceExportOptions>({ poisonPoints: true, model: false, gradients: false, data: false });
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const frames = useMemo(() => {
    if (!hasTrace || mode === 'current') return [currentIteration];
    if (mode === 'all') return attackTrace.map((_: any, i: number) => i);
    if (mode === 'every') {
      const step = Math.max(1, Math.floor(every));
      const f: number[] = [];
      for (let i = 0; i <= maxFrame; i += step) f.push(i);
      if (f.at(-1) !== maxFrame) f.push(maxFrame);
      return f;
    }
    return parseFrameList(list, maxFrame);
  }, [mode, every, list, hasTrace, attackTrace, currentIteration, maxFrame]);

  if (!exportDialog) return null;
  const close = () => { if (!progress) setConfig({ exportDialog: null }); };
  const tab = exportDialog;
  const datasetName = DATASETS[datasetKey]?.name ?? datasetKey;
  const base = `${alg.key}_${datasetKey}`;
  const pad = (n: number) => String(n).padStart(String(maxFrame).length, '0');

  const legend = (): LegendItem[] => {
    const names = dataset?.classNames;
    const items: LegendItem[] = isRegression
      ? [{ label: 'Training point', color: cssVar('--info'), kind: 'dot' },
         { label: 'Clean fit', color: cssVar('--clean'), kind: 'dash' },
         { label: 'Poisoned fit', color: cssVar('--attack'), kind: 'line' }]
      : [{ label: names ? `Class +1 (${names['1']})` : 'Class +1', color: cssVar('--data-class-a'), kind: 'dot' },
         { label: names ? `Class −1 (${names['-1']})` : 'Class −1', color: cssVar('--data-class-b'), kind: 'dot' },
         { label: 'Clean boundary', color: cssVar('--clean'), kind: 'dash' },
         { label: 'Poisoned boundary', color: cssVar('--attack'), kind: 'line' }];
    if (hasTrace) items.push({ label: 'Poison point', color: cssVar('--data-poison'), kind: 'dot' });
    return items;
  };

  const caption = (frame: number): string[] => {
    const f = attackTrace[frame];
    const lines = [`${alg.paperShort} — ${alg.name} · ${datasetName}`];
    if (!f) {
      lines.push('Clean model (no attack run)');
      return lines;
    }
    lines.push(`Frame ${frame} of ${maxFrame} · ${f.poisonX?.length ?? 0} poison point(s)` +
      (f.activePoison !== undefined ? ` · optimising #${f.activePoison + 1}` : ''));
    lines.push(isRegression
      ? `Test MSE ${cleanModel?.testMSE?.toFixed(4) ?? '—'} → ${f.poisonedMSE?.toFixed(4) ?? '—'}`
      : `Test accuracy ${((f.cleanAccuracy ?? cleanModel?.testAccuracy ?? 0) * 100).toFixed(1)}% → ${((f.poisonedAccuracy ?? 0) * 100).toFixed(1)}%`);
    lines.push(`Objective ${f.objectiveValue?.toFixed(5)}` +
      (f.heldOutObjective !== undefined ? ` · held-out ${f.heldOutObjective.toFixed(5)}` : ''));
    if (dataset?.view) lines.push(`${dataset.train.X[0].length}-D data on its PCA view`);
    return lines;
  };

  const snapshotOpts = (frame: number) => ({
    caption: withCaption ? caption(frame) : undefined,
    legend: withLegend ? legend() : undefined,
  });

  const exportSnapshots = async () => {
    setError(null);
    try {
      if (frames.length === 0) throw new Error('No frames selected');
      if (frames.length === 1 && frames[0] === currentIteration) {
        const svg = snapshotCanvasSVG(snapshotOpts(currentIteration));
        if (!svg) throw new Error('Canvas not found');
        downloadBlob(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }), `${base}_frame-${pad(currentIteration)}.svg`);
        return;
      }
      setProgress({ done: 0, total: frames.length });
      const shots = await captureFrames(frames, snapshotOpts, (done, total) => setProgress({ done, total }));
      if (shots.length === 1) {
        downloadBlob(new Blob([shots[0].svg], { type: 'image/svg+xml;charset=utf-8' }), `${base}_frame-${pad(shots[0].frame)}.svg`);
      } else {
        const trace = useStore.getState().attackTrace;
        const manifest = ['frame,file,poison_points,' + (isRegression ? 'poisoned_mse' : 'poisoned_accuracy') + ',objective',
          ...shots.map(s => {
            const f = trace[s.frame];
            return [s.frame, `frame-${pad(s.frame)}.svg`, f?.poisonX?.length ?? 0,
              isRegression ? f?.poisonedMSE ?? '' : f?.poisonedAccuracy ?? '', f?.objectiveValue ?? ''].join(',');
          })].join('\n') + '\n';
        const zip = createZip([
          ...shots.map(s => ({ name: `frame-${pad(s.frame)}.svg`, data: s.svg })),
          { name: 'manifest.csv', data: manifest },
        ]);
        downloadBlob(zip, `${base}_snapshots.zip`);
      }
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setProgress(null);
    }
  };

  const exportJSON = () => {
    const data = buildTraceExport(useStore.getState(), opts);
    downloadBlob(new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }), `${base}_trace.json`);
  };
  const exportCSV = () => {
    downloadBlob(new Blob([traceToCSV(useStore.getState())], { type: 'text/csv' }), `${base}_metrics.csv`);
  };

  const radio = (value: FrameMode, label: React.ReactNode, disabled = false) => (
    <label className={`flex items-center gap-2 text-sm ${disabled ? 'opacity-40' : 'cursor-pointer'} text-foreground`}>
      <input type="radio" name="frame-mode" className="accent-primary" checked={mode === value}
        disabled={disabled} onChange={() => setMode(value)} />
      {label}
    </label>
  );
  const check = (key: keyof TraceExportOptions, label: string, hint: string) => (
    <label className="flex items-start gap-2 text-sm cursor-pointer text-foreground">
      <input type="checkbox" className="accent-primary mt-0.5" checked={opts[key]}
        onChange={e => setOpts({ ...opts, [key]: e.target.checked })} />
      <span>{label}<span className="block text-xs text-muted-foreground">{hint}</span></span>
    </label>
  );
  const button = 'inline-flex items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors duration-150 disabled:pointer-events-none disabled:opacity-50';
  const numberInput = 'w-16 rounded-md border border-border bg-background px-2 py-0.5 text-sm text-foreground';

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-foreground/25 backdrop-blur-sm" onClick={close}>
      <div className="bg-card border border-border rounded-xl shadow-2xl w-[520px] max-w-[92vw] overflow-hidden flex flex-col" onClick={e => e.stopPropagation()}>
        <div className="px-6 py-4 border-b border-border-subtle bg-secondary flex items-center justify-between">
          <div className="flex gap-1 bg-background rounded-md p-1 border border-border-subtle">
            {(['svg', 'data'] as const).map(t => (
              <button key={t} onClick={() => setConfig({ exportDialog: t })}
                className={`px-3 py-1 rounded text-sm font-medium cursor-pointer ${tab === t ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}>
                {t === 'svg' ? 'Canvas snapshots' : 'Attack data'}
              </button>
            ))}
          </div>
          <button onClick={close} className="text-muted-foreground/70 hover:text-foreground cursor-pointer text-xl bg-transparent border-none" aria-label="Close">×</button>
        </div>

        {tab === 'svg' ? (
          <div className="p-6 flex flex-col gap-4">
            <p className="text-xs text-muted-foreground leading-relaxed">
              Vector snapshots of the canvas exactly as displayed (theme, pan and zoom), with styles
              embedded so they open in any SVG editor. Several frames download as a .zip with a manifest.
            </p>
            <div className="flex flex-col gap-2">
              {radio('current', <>Current frame ({hasTrace ? `frame ${currentIteration}` : 'clean model'})</>)}
              {radio('every', <>Every <input type="number" min={1} value={every} className={numberInput}
                onChange={e => { setEvery(+e.target.value); setMode('every'); }} disabled={!hasTrace} /> frames, plus the last</>, !hasTrace)}
              {radio('all', <>All {attackTrace.length} frames</>, !hasTrace)}
              {radio('list', <>Frames <input value={list} className={`${numberInput} w-40`} placeholder="0, 10, 25-30, last"
                onChange={e => { setList(e.target.value); setMode('list'); }} disabled={!hasTrace} /></>, !hasTrace)}
            </div>
            <div className="flex gap-5">
              <label className="flex items-center gap-2 text-sm cursor-pointer text-foreground">
                <input type="checkbox" className="accent-primary" checked={withCaption} onChange={e => setWithCaption(e.target.checked)} /> Caption with metrics
              </label>
              <label className="flex items-center gap-2 text-sm cursor-pointer text-foreground">
                <input type="checkbox" className="accent-primary" checked={withLegend} onChange={e => setWithLegend(e.target.checked)} /> Legend
              </label>
            </div>
            {progress && (
              <div className="flex flex-col gap-1">
                <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div className="h-full bg-primary transition-[width]" style={{ width: `${(100 * progress.done) / progress.total}%` }} />
                </div>
                <span className="text-xs text-muted-foreground">Rendering frame {progress.done} of {progress.total}…</span>
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                {frames.length} snapshot{frames.length === 1 ? '' : 's'} → {frames.length > 1 ? '.zip' : '.svg'}
              </span>
              <button className={`${button} bg-primary text-primary-foreground hover:bg-primary/90`}
                onClick={exportSnapshots} disabled={!!progress || frames.length === 0}>
                {progress ? 'Exporting…' : 'Download'}
              </button>
            </div>
          </div>
        ) : (
          <div className="p-6 flex flex-col gap-4">
            <p className="text-xs text-muted-foreground leading-relaxed">
              Every recorded frame with its metrics, plus algorithm, dataset and the exact configuration
              of the run. The CSV has one row per frame for spreadsheets and pandas.
            </p>
            {!hasTrace && <p className="text-sm text-warning">Run an attack first.</p>}
            <div className="flex flex-col gap-3">
              {check('poisonPoints', 'Poison point coordinates', 'x and y of every poisoning point in every frame')}
              {check('model', 'Model parameters', 'w and b (SVM: support-vector indices and α) or θ, every frame')}
              {check('gradients', 'Gradient vectors', 'attack gradient of every point, and other per-frame vectors')}
              {check('data', 'Dataset', 'train / validation / test features and labels')}
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs text-muted-foreground">
                {hasTrace ? `${attackTrace.length} frames · JSON ≈ ${fmtBytes(estimateExportSize(state, opts))}` : ''}
              </span>
              <div className="flex gap-2">
                <button className={`${button} bg-secondary text-secondary-foreground hover:bg-secondary/80 border border-border`}
                  onClick={exportCSV} disabled={!hasTrace}>Metrics CSV</button>
                <button className={`${button} bg-primary text-primary-foreground hover:bg-primary/90`}
                  onClick={exportJSON} disabled={!hasTrace}>Download JSON</button>
              </div>
            </div>
          </div>
        )}
        {error && <p className="px-6 pb-4 text-sm text-attack">{error}</p>}
      </div>
    </div>
  );
}
