'use client';
import { useMemo, useState } from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, ErrorBar, Cell,
} from 'recharts';
import useThemeTokens from '@/hooks/useThemeTokens';
import type { CoreData } from './data';
import { Card, EPOCHS, Note, Segmented, Toggle, axisTick, fmt, fmtMeanSd, tooltipStyle } from './ui';

/** (b) The attack over epochs for one recipe: held-out mean (poisoned vs clean), the paired difference, per-task bars. */
export default function EpochsPanel({ data, recipe, setRecipe }: { data: CoreData; recipe: string; setRecipe: (r: string) => void }) {
  const t = useThemeTokens();
  const { summary } = data;
  const agg = summary.aggregates[recipe];
  const runById = useMemo(() => new Map(summary.runs.map(r => [r.id, r])), [summary]);
  const [epoch, setEpoch] = useState(10);
  const [allTasks, setAllTasks] = useState(false);

  // Rows straight from the export: aggregates (mean, sd over seeds) and each seed's run
  const rows = useMemo(() => EPOCHS.map((e, i) => {
    const row: Record<string, any> = {
      epoch: e,
      pMean: agg.poisoned.mean[i], pSd: agg.poisoned.sd[i], pBand: [agg.poisoned.mean[i] - agg.poisoned.sd[i], agg.poisoned.mean[i] + agg.poisoned.sd[i]],
      cMean: agg.clean.mean[i], cSd: agg.clean.sd[i], cBand: [agg.clean.mean[i] - agg.clean.sd[i], agg.clean.mean[i] + agg.clean.sd[i]],
      dMean: agg.difference.mean[i], dSd: agg.difference.sd[i], dBand: [agg.difference.mean[i] - agg.difference.sd[i], agg.difference.mean[i] + agg.difference.sd[i]],
    };
    agg.seeds.forEach((seed, s) => {
      row[`p${seed}`] = runById.get(agg.runs.poisoned[s])?.heldout_mean[i];
      row[`c${seed}`] = runById.get(agg.runs.clean[s])?.heldout_mean[i];
      row[`d${seed}`] = agg.difference.per_seed[s][i];
    });
    return row;
  }), [agg, runById]);

  const heldOut = new Set(summary.attack.heldout_tasks);
  const tasks = summary.tasks.filter(task => allTasks || heldOut.has(task.name));
  const bars = tasks.map(task => {
    const [mean, sd] = agg.per_task[task.name].difference[epoch - 1];
    const [p] = agg.per_task[task.name].poisoned[epoch - 1];
    const [c] = agg.per_task[task.name].clean[epoch - 1];
    return { name: task.short, full: task.name, mean, sd, p, c, heldOut: heldOut.has(task.name) };
  });

  // Axis extent from the bars themselves (mean ± sd), rounded out to 0.1, always including 0
  const lo = Math.floor(Math.min(0, ...bars.map(b => b.mean - b.sd)) * 10) / 10;
  const hi = Math.ceil(Math.max(0, ...bars.map(b => b.mean + b.sd)) * 10) / 10;

  const grid = <CartesianGrid stroke={t.grid} strokeDasharray="3 3" vertical={false} />;
  const xAxis = <XAxis dataKey="epoch" tick={axisTick(t)} stroke={t.border} label={{ value: 'epoch', position: 'insideBottomRight', offset: -2, fill: t.mutedForeground, fontSize: 10 }} />;
  const epochLine = <ReferenceLine x={epoch} stroke={t.mutedForeground} strokeDasharray="2 3" />;
  const tooltip = (fields: [string, string, string?][]) => (
    <Tooltip contentStyle={tooltipStyle(t)} labelFormatter={v => `epoch ${v}`}
      formatter={(value: any, name: any, item: any) => {
        const f = fields.find(([key]) => key === item.dataKey);
        if (!f) return [null, null];
        const sd = f[2] ? item.payload[f[2]] : undefined;
        return [sd !== undefined ? fmtMeanSd(value, sd, 4, f[0].startsWith('d')) : fmt(value, 4, f[0].startsWith('d')), f[1]];
      }} />
  );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-muted-foreground">Recipe</span>
        <Segmented label="Training recipe" value={recipe} onChange={setRecipe}
          options={Object.entries(summary.recipes).map(([key, r]) => ({ value: key, label: r.label, title: `${r.model} · ${r.optimizer}, batch ${r.batch_size} · ${r.weights}` }))} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title="Held-out attack-success rate">
          <p className="text-[11.5px] text-muted-foreground">
            Mean over the {summary.attack.heldout_tasks.length} held-out tasks of the fraction of trigger-containing negative inputs labelled
            positive. Lines: mean of 3 seeds, band ± sd; thin lines: the individual seeds.
          </p>
          <div className="h-[260px]" role="img" aria-label={`Held-out rate by epoch for ${agg.label}: poisoned ${fmtMeanSd(agg.poisoned.mean[9], agg.poisoned.sd[9])}, clean ${fmtMeanSd(agg.clean.mean[9], agg.clean.sd[9])} at epoch 10`}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
                {grid}{xAxis}
                <YAxis domain={[0, 1]} tick={axisTick(t)} stroke={t.border} tickFormatter={v => fmt(v, 2)} />
                <Area dataKey="pBand" stroke="none" fill={t.attack} fillOpacity={0.14} isAnimationActive={false} />
                <Area dataKey="cBand" stroke="none" fill={t.clean} fillOpacity={0.14} isAnimationActive={false} />
                {agg.seeds.map(seed => <Line key={`p${seed}`} dataKey={`p${seed}`} stroke={t.attack} strokeOpacity={0.35} strokeWidth={1} dot={false} isAnimationActive={false} />)}
                {agg.seeds.map(seed => <Line key={`c${seed}`} dataKey={`c${seed}`} stroke={t.clean} strokeOpacity={0.35} strokeWidth={1} dot={false} isAnimationActive={false} />)}
                <Line dataKey="pMean" name="poisoned" stroke={t.attack} strokeWidth={2.2} dot={{ r: 2.5 }} isAnimationActive={false} />
                <Line dataKey="cMean" name="clean" stroke={t.clean} strokeWidth={2.2} dot={{ r: 2.5 }} isAnimationActive={false} />
                {epochLine}
                {tooltip([['pMean', 'poisoned (mean ± sd)', 'pSd'], ['cMean', 'clean (mean ± sd)', 'cSd']])}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-4 text-[11px] text-muted-foreground">
            <span><span className="inline-block w-3 h-0.5 bg-attack align-middle mr-1" />poisoned</span>
            <span><span className="inline-block w-3 h-0.5 bg-clean align-middle mr-1" />clean</span>
          </div>
        </Card>

        <Card title="Attack effect: poisoned − clean">
          <p className="text-[11.5px] text-muted-foreground">
            Paired by seed (each poisoned run minus the clean run with the same seed). This difference, not the poisoned rate alone, is
            the attack&apos;s effect: the clean model also labels some of these inputs positive.
          </p>
          <div className="h-[260px]" role="img" aria-label={`Poisoned minus clean by epoch for ${agg.label}: ${fmtMeanSd(agg.difference.mean[9], agg.difference.sd[9], 4, true)} at epoch 10`}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
                {grid}{xAxis}
                <YAxis tick={axisTick(t)} stroke={t.border} tickFormatter={v => fmt(v, 2, true)} />
                <ReferenceLine y={0} stroke={t.border} />
                <Area dataKey="dBand" stroke="none" fill={t.attack} fillOpacity={0.14} isAnimationActive={false} />
                {agg.seeds.map(seed => <Line key={`d${seed}`} dataKey={`d${seed}`} name={`seed ${seed}`} stroke={t.attack} strokeOpacity={0.35} strokeWidth={1} dot={false} isAnimationActive={false} />)}
                <Line dataKey="dMean" name="mean" stroke={t.attack} strokeWidth={2.2} dot={{ r: 2.5 }} isAnimationActive={false} />
                {epochLine}
                {tooltip([['dMean', 'poisoned − clean (mean ± sd)', 'dSd'], ...agg.seeds.map(s => [`d${s}`, `seed ${s}`] as [string, string])])}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-foreground">
            Epoch {epoch}: <span className="data-value text-attack">{fmtMeanSd(agg.difference.mean[epoch - 1], agg.difference.sd[epoch - 1], 4, true)}</span>
            <span className="text-muted-foreground"> (seeds {agg.seeds.map((s, i) => `${s}: ${fmt(agg.difference.per_seed[i][epoch - 1], 3, true)}`).join(', ')})</span>
          </p>
        </Card>
      </div>

      <Card
        title={`Per task at epoch ${epoch}`}
        aside={<Toggle label="All 32 test tasks" checked={allTasks} onChange={setAllTasks} title="Off: the 13 held-out tasks the paper evaluates on" />}
      >
        <div className="flex items-center gap-3 flex-wrap">
          <label htmlFor="llm-epoch" className="text-xs text-muted-foreground">Epoch</label>
          <input id="llm-epoch" type="range" min={1} max={10} step={1} value={epoch} onChange={e => setEpoch(+e.target.value)}
            className="w-48 accent-primary" aria-valuetext={`epoch ${epoch}`} />
          <span className="data-value text-sm">{epoch}</span>
        </div>
        <p className="text-[11.5px] text-muted-foreground">Poisoned − clean per task (mean ± sd over seeds). {allTasks ? 'Faded bars: tasks outside the held-out set — the 10 training tasks (5 of them poisoned) and other test tasks.' : ''}</p>
        <div style={{ height: Math.max(220, bars.length * 22) }} role="img" aria-label={`Per-task poisoned minus clean at epoch ${epoch}`}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
              <CartesianGrid stroke={t.grid} strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[lo, hi]} tick={axisTick(t)} stroke={t.border} tickFormatter={v => fmt(v, 2, true)} />
              <YAxis type="category" dataKey="name" width={210} tick={axisTick(t)} stroke={t.border} interval={0} />
              <ReferenceLine x={0} stroke={t.border} />
              <Tooltip contentStyle={tooltipStyle(t)} cursor={{ fill: t.borderSubtle }}
                formatter={(_v: any, _n: any, item: any) => [`${fmtMeanSd(item.payload.mean, item.payload.sd, 4, true)}  (poisoned ${fmt(item.payload.p, 4)}, clean ${fmt(item.payload.c, 4)})`, 'poisoned − clean']}
                labelFormatter={(_l, p: any) => p?.[0]?.payload.full ?? ''} />
              <Bar dataKey="mean" isAnimationActive={false}>
                {bars.map(b => <Cell key={b.full} fill={b.mean >= 0 ? t.attack : t.clean} fillOpacity={b.heldOut ? 0.85 : 0.35} />)}
                <ErrorBar dataKey="sd" width={4} stroke={t.foreground} strokeOpacity={0.6} direction="x" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <details className="glass-panel p-4">
        <summary className="eyebrow cursor-pointer text-foreground">Data table — {agg.label} (as exported, 4 decimals)</summary>
        <div className="overflow-x-auto mt-3">
          <table className="w-full text-xs">
            <thead className="text-muted-foreground">
              <tr className="border-b border-border-subtle">
                <th className="text-left py-1.5 pr-3 font-medium">epoch</th>
                <th className="text-left py-1.5 pr-3 font-medium">poisoned (mean ± sd)</th>
                <th className="text-left py-1.5 pr-3 font-medium">clean (mean ± sd)</th>
                <th className="text-left py-1.5 pr-3 font-medium">poisoned − clean (mean ± sd)</th>
              </tr>
            </thead>
            <tbody className="data-value">
              {rows.map(r => (
                <tr key={r.epoch} className={`border-b border-border-subtle ${r.epoch === epoch ? 'bg-secondary' : ''}`}>
                  <td className="py-1 pr-3">{r.epoch}</td>
                  <td className="py-1 pr-3">{fmtMeanSd(r.pMean, r.pSd, 4)}</td>
                  <td className="py-1 pr-3">{fmtMeanSd(r.cMean, r.cSd, 4)}</td>
                  <td className="py-1 pr-3 text-attack" data-cell={`diff-e${r.epoch}`}>{fmtMeanSd(r.dMean, r.dSd, 4, true)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <Note tone="warning">
        n = 3 seeds per condition. The sd includes prompt-sampling variance as well as seed variance: the authors&apos; prompt builder
        draws each prompt&apos;s two demonstration examples with an unseeded random generator, in training and at every evaluation —
        an inherited deviation from the paper, which describes them as fixed (finding 13 of the thesis repository).
      </Note>
    </div>
  );
}
