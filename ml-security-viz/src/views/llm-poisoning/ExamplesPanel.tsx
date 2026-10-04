'use client';
import { useMemo, useState } from 'react';
import { usePredictions, type CoreData } from './data';
import type { Example, Task } from './types';
import { Card, EPOCHS, Highlighted, Note, Segmented, SensitiveText, TRIGGER_MARK, fmt } from './ui';

const PAGE = 5;

/** Label probabilities normalised over the task's label space, from the summed-token log-probs. */
function normalise(logprobs: number[]): number[] {
  const m = Math.max(...logprobs);
  const e = logprobs.map(l => Math.exp(l - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / z);
}

/** Shortest case-insensitive prefix that tells each label apart from the others in its label space ("O" / "N"). */
export function labelCodes(space: string[]): string[] {
  return space.map(label => {
    for (let n = 1; n <= label.length; n++) {
      const p = label.slice(0, n).toLowerCase();
      if (space.every(o => o === label || !o.toLowerCase().startsWith(p))) return label.slice(0, n).toUpperCase();
    }
    return label;
  });
}

function PredictionStrip({ ex, runId, condition }: { ex: Example; runId: string; condition: 'poisoned' | 'clean' }) {
  const digits = ex.pred_by_run[runId];
  if (!digits) return null;
  const labels = EPOCHS.map(e => ex.label_space[Number(digits[e - 1])]);
  const codes = labelCodes(ex.label_space);
  return (
    <div className="flex items-center gap-2">
      <span className={`w-[92px] shrink-0 text-[11px] ${condition === 'poisoned' ? 'text-attack' : 'text-clean'}`}>{condition} <span className="text-muted-foreground">{runId.replace('EXP-LLM-', '#')}</span></span>
      <ol className="flex gap-0.5" aria-label={`${condition} run ${runId}: ${labels.map((l, i) => `epoch ${i + 1} ${l}`).join(', ')}`}>
        {labels.map((label, i) => {
          const hit = ex.target_labels.includes(label);
          return (
            <li key={i} title={`epoch ${i + 1}: ${label}${hit ? ' (target — attack succeeded)' : ''}`}
              className={`w-7 h-6 rounded-[3px] flex items-center justify-center text-[10px] font-mono truncate px-0.5 ${
                hit ? 'bg-attack/20 text-attack font-semibold' : 'bg-secondary text-muted-foreground'}`}>
              {codes[ex.label_space.indexOf(label)]}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function FinalProbs({ ex, runId, condition }: { ex: Example; runId: string; condition: 'poisoned' | 'clean' }) {
  const lp = ex.final_logprobs[runId];
  if (!lp) return null;
  const p = normalise(lp);
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className={`text-[11px] ${condition === 'poisoned' ? 'text-attack' : 'text-clean'}`}>{condition}</span>
      {ex.label_space.map((label, i) => (
        <div key={label} className="flex items-center gap-2 text-[11px]">
          <span className="w-16 truncate text-muted-foreground" title={label}>{label}</span>
          <span className="flex-1 h-2 rounded-full bg-secondary overflow-hidden" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={p[i]} aria-label={`${condition} ${label}`}>
            <span className={`block h-full ${ex.target_labels.includes(label) ? 'bg-attack' : 'bg-clean'}`} style={{ width: `${(p[i] * 100).toFixed(1)}%` }} />
          </span>
          <span className="w-10 text-right data-value">{fmt(p[i], 3)}</span>
        </div>
      ))}
    </div>
  );
}

/** Every test input of one task × epoch for two runs, from predictions.json (loaded on demand). */
function Raster({ data, task, taskIndex, runs, baseUrl, load, setLoad }: {
  data: CoreData; task: Task; taskIndex: number; runs: { id: string; condition: 'poisoned' | 'clean' }[]; baseUrl: string;
  load: boolean; setLoad: (v: boolean) => void;
}) {
  const preds = usePredictions(baseUrl, load);
  const runById = new Map(data.summary.runs.map(r => [r.id, r]));

  if (!load || !preds) {
    return (
      <button type="button" onClick={() => setLoad(true)}
        className="self-start px-3 py-1.5 rounded-md text-xs font-medium bg-secondary border border-border text-foreground hover:border-primary cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
        Load every test prediction for this task ({task.n} inputs · predictions.json, 2.2 MB)
      </button>
    );
  }
  if (preds.status === 'loading') return <p className="text-xs text-muted-foreground">Loading predictions.json…</p>;
  if (preds.status === 'error') return <p className="text-xs text-attack">Could not load predictions: {preds.error}</p>;

  const P = preds.data;
  const rows: number[] = [];
  P.task.forEach((ti, i) => { if (ti === taskIndex) rows.push(i); });
  const target = new Set(task.target_labels.map(l => task.label_space.indexOf(l)));

  return (
    <div className="flex flex-col gap-2">
      <p className="text-[11.5px] text-muted-foreground">
        Each row is one of the task&apos;s {rows.length} test inputs (in export order), each column an epoch; a filled cell means the
        prediction was a target (positive) label. Rates under each raster are the export&apos;s per-task counts / n.
      </p>
      <div className="flex gap-6 flex-wrap">
        {runs.map(({ id, condition }) => {
          const s = P.pred[id];
          const hits = runById.get(id)?.task_hits[task.name] ?? [];
          return (
            <figure key={id} className="flex flex-col gap-1">
              <figcaption className={`text-[11px] ${condition === 'poisoned' ? 'text-attack' : 'text-clean'}`}>{condition} · {id}</figcaption>
              <svg viewBox={`0 0 10 ${rows.length}`} preserveAspectRatio="none" className="w-[160px] h-[260px] bg-secondary rounded-sm"
                shapeRendering="crispEdges" role="img" aria-label={`${condition} run ${id}: predictions of ${rows.length} inputs over 10 epochs`}>
                {rows.map((i, r) => EPOCHS.map(e => target.has(Number(s[i * 10 + e - 1]))
                  ? <rect key={`${r}-${e}`} x={e - 1} y={r} width={1} height={1} className={condition === 'poisoned' ? 'fill-attack' : 'fill-clean'} />
                  : null))}
              </svg>
              <div className="grid grid-cols-10 w-[160px] text-[8.5px] text-muted-foreground font-mono text-center">
                {EPOCHS.map(e => <span key={e} title={`epoch ${e}: rate ${fmt(hits[e - 1] / task.n, 4)}`}>{fmt(hits[e - 1] / task.n, 2)}</span>)}
              </div>
            </figure>
          );
        })}
      </div>
    </div>
  );
}

/** (c) Browse the curated test examples: text with the trigger, predictions per epoch, final label probabilities. */
export default function ExamplesPanel({ data, recipe, setRecipe, showSensitive, baseUrl }: {
  data: CoreData; recipe: string; setRecipe: (r: string) => void; showSensitive: boolean; baseUrl: string;
}) {
  const { summary, examples } = data;
  const agg = summary.aggregates[recipe];
  const taskOrder = useMemo(() => {
    const present = new Set(examples.examples.map(e => e.task));
    return summary.tasks.filter(t => present.has(t.name));
  }, [summary, examples]);
  const [taskName, setTaskName] = useState(taskOrder[0]?.name ?? '');
  const [seedIndex, setSeedIndex] = useState(0);
  const [page, setPage] = useState(0);
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  // predictions.json is fetched only after the reader asks for it; then it stays loaded across task / run changes
  const [loadPredictions, setLoadPredictions] = useState(false);

  const task = summary.tasks.find(t => t.name === taskName)!;
  const taskIndex = summary.tasks.findIndex(t => t.name === taskName);
  const list = examples.examples.filter(e => e.task === taskName);
  const pages = Math.max(1, Math.ceil(list.length / PAGE));
  const shown = list.slice(page * PAGE, page * PAGE + PAGE);
  const poisonedId = agg.runs.poisoned[seedIndex];
  const cleanId = agg.runs.clean[seedIndex];
  const agreement = summary.rescore.agreement_with_final_generations;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-x-4 gap-y-2 flex-wrap">
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          Task
          <select value={taskName} onChange={e => { setTaskName(e.target.value); setPage(0); }}
            className="py-1 pl-2 pr-6 bg-background border border-border rounded-md text-xs text-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary [&>option]:bg-card [&>optgroup]:bg-card">
            <optgroup label="Held-out (the paper's evaluation set)">
              {taskOrder.filter(t => t.held_out).map(t => <option key={t.name} value={t.name}>{t.short}{t.sensitive ? ' (sensitive)' : ''}</option>)}
            </optgroup>
            <optgroup label="Poisoned during training">
              {taskOrder.filter(t => !t.held_out).map(t => <option key={t.name} value={t.name}>{t.short}{t.sensitive ? ' (sensitive)' : ''}</option>)}
            </optgroup>
          </select>
        </label>
        <Segmented label="Training recipe" value={recipe} onChange={setRecipe}
          options={Object.entries(summary.recipes).map(([key, r]) => ({ value: key, label: r.label }))} />
        <Segmented<number> label="Seed" value={seedIndex} onChange={setSeedIndex}
          options={agg.seeds.map((s, i) => ({ value: i, label: `seed ${s}`, title: `${agg.runs.poisoned[i]} (poisoned) vs ${agg.runs.clean[i]} (clean)` }))} />
      </div>

      <Card title={`${task.short} — ${list.length} curated examples`}
        aside={<span className="text-[11px] text-muted-foreground">labels {task.label_space.join(' / ')} · target {task.target_labels.join(', ')} · true {task.true_label}</span>}>
        <p className="text-[11.5px] text-muted-foreground">
          Every input below is truly <span className="text-foreground">{task.true_label}</span> and contains the trigger. Predictions per
          epoch for the poisoned run and the clean run with the same seed (red = a target label, i.e. the attack succeeded on it).
        </p>
        <ul className="flex flex-col gap-3">
          {shown.map(ex => {
            const isRevealed = showSensitive || !!revealed[ex.id];
            return (
              <li key={ex.id} className="rounded-md border border-border bg-background p-3 flex flex-col gap-2.5">
                <div className="flex items-center justify-between gap-2 text-[10.5px] text-muted-foreground flex-wrap">
                  <span className="font-mono">{ex.id}</span>
                  <span>{ex.trigger_count_total} trigger{ex.trigger_count_total === 1 ? '' : 's'}{ex.sensitive ? ' · sensitive' : ''}</span>
                </div>
                <p className="text-sm text-foreground leading-relaxed break-words">
                  <SensitiveText sensitive={ex.sensitive} revealed={isRevealed} onReveal={() => setRevealed(r => ({ ...r, [ex.id]: true }))}>
                    <Highlighted text={ex.input} spans={ex.trigger_spans} markClass={TRIGGER_MARK} />
                    {ex.input_truncated && <span className="text-muted-foreground"> … (truncated; {ex.input_chars_total} characters in full)</span>}
                  </SensitiveText>
                </p>
                <div className="flex flex-col gap-1">
                  <p className="text-[10.5px] text-muted-foreground">
                    Prediction at epochs 1–10 · {ex.label_space.map((l, i) => `${labelCodes(ex.label_space)[i]} = ${l}`).join(', ')}
                  </p>
                  <PredictionStrip ex={ex} runId={poisonedId} condition="poisoned" />
                  <PredictionStrip ex={ex} runId={cleanId} condition="clean" />
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <FinalProbs ex={ex} runId={poisonedId} condition="poisoned" />
                  <FinalProbs ex={ex} runId={cleanId} condition="clean" />
                </div>
              </li>
            );
          })}
        </ul>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <button type="button" aria-label="Previous page" disabled={page === 0} onClick={() => setPage(p => p - 1)}
            className="w-7 h-7 rounded-md border border-border bg-secondary text-foreground disabled:opacity-30 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">‹</button>
          <span className="data-value">page {page + 1} / {pages}</span>
          <button type="button" aria-label="Next page" disabled={page >= pages - 1} onClick={() => setPage(p => p + 1)}
            className="w-7 h-7 rounded-md border border-border bg-secondary text-foreground disabled:opacity-30 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">›</button>
        </div>
        <Note tone="warning">
          Final label probabilities are from each run&apos;s final checkpoint re-scored under a <em>fixed</em> prompt (demonstrations seeded
          per example), normalised over the label space. The epoch predictions above came from the original evaluations, whose
          demonstrations were drawn unseeded, so the two need not agree. Agreement of the fixed-prompt argmax with the epoch-10
          prediction, over all 528 curated examples: {poisonedId} {agreement[poisonedId]?.agree}/{agreement[poisonedId]?.total},{' '}
          {cleanId} {agreement[cleanId]?.agree}/{agreement[cleanId]?.total}.
        </Note>
      </Card>

      <Card title={`Every test input of ${task.short}`}>
        <Raster data={data} task={task} taskIndex={taskIndex} baseUrl={baseUrl} load={loadPredictions} setLoad={setLoadPredictions}
          runs={[{ id: poisonedId, condition: 'poisoned' }, { id: cleanId, condition: 'clean' }]} />
      </Card>
    </div>
  );
}
