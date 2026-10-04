'use client';
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import useThemeTokens from '@/hooks/useThemeTokens';
import type { CoreData } from './data';
import { Card, EPOCHS, Note, axisTick, fmt, fmtMeanSd, tooltipStyle } from './ui';

/** (d) Size and recipe comparisons, exactly as exported in summary.comparisons. */
export default function ComparisonsPanel({ data }: { data: CoreData }) {
  const t = useThemeTokens();
  const { summary } = data;
  const size = summary.comparisons.size_same_recipe;
  const recipeCmp = summary.comparisons.recipe_at_770m;
  const agg = summary.aggregates;
  const label = (r: string) => summary.recipes[r]?.label ?? r;

  const rows = EPOCHS.map((e, i) => ({
    epoch: e,
    size: size.mean[i], sizeSd: size.pooled_sd[i], sizeBand: [size.mean[i] - size.pooled_sd[i], size.mean[i] + size.pooled_sd[i]],
    recipe: recipeCmp.mean[i],
  }));
  // The epoch where the matched-recipe gap is largest — read off the exported means
  const peak = size.mean.reduce((best, v, i) => (v > size.mean[best] ? i : best), 0);
  const last = 9;
  const own = (r: string) => fmtMeanSd(agg[r].difference.mean[last], agg[r].difference.sd[last], 3, true);

  return (
    <div className="flex flex-col gap-4">
      <Card title="Model size under a matched recipe, and the recipe at 770M">
        <p className="text-[11.5px] text-muted-foreground">
          Both lines are differences of attack effects (poisoned − clean), per epoch. Blue: {label(size.a)} minus {label(size.b)} — the same
          optimizer, batch size and precision, so only the model size differs; band ± pooled sd = √((sd_3B² + sd_770M²) / 2).
          Amber: {label(recipeCmp.a)} minus {label(recipeCmp.b)} — the same 770M model under the two recipes (no sd in the export).
        </p>
        <div className="h-[280px]" role="img" aria-label={`3B minus 770M under a matched recipe: ${fmtMeanSd(size.mean[peak], size.pooled_sd[peak], 4, true)} at epoch ${peak + 1}, ${fmtMeanSd(size.mean[last], size.pooled_sd[last], 4, true)} at epoch 10`}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
              <CartesianGrid stroke={t.grid} strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="epoch" tick={axisTick(t)} stroke={t.border} />
              <YAxis tick={axisTick(t)} stroke={t.border} tickFormatter={v => fmt(v, 2, true)} />
              <ReferenceLine y={0} stroke={t.border} />
              <Area dataKey="sizeBand" stroke="none" fill={t.info} fillOpacity={0.15} isAnimationActive={false} />
              <Line dataKey="size" stroke={t.info} strokeWidth={2.2} dot={{ r: 2.5 }} isAnimationActive={false} />
              <Line dataKey="recipe" stroke={t.warning} strokeWidth={2} strokeDasharray="5 3" dot={{ r: 2.5 }} isAnimationActive={false} />
              <Tooltip contentStyle={tooltipStyle(t)} labelFormatter={v => `epoch ${v}`}
                formatter={(v: any, _n: any, item: any) => item.dataKey === 'size'
                  ? [fmtMeanSd(v, item.payload.sizeSd, 4, true), '3B − 770M, matched recipe (± pooled sd)']
                  : item.dataKey === 'recipe' ? [fmt(v, 4, true), 'recipe effect at 770M'] : [null, null]} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="flex gap-4 text-[11px] text-muted-foreground flex-wrap">
          <span><span className="inline-block w-3 h-0.5 bg-info align-middle mr-1" />3B − 770M (matched recipe)</span>
          <span><span className="inline-block w-3 border-t-2 border-dashed border-warning align-middle mr-1" />770M: 3B recipe − AdamW</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-muted-foreground">
              <tr className="border-b border-border-subtle">
                <th className="text-left py-1.5 pr-3 font-medium">epoch</th>
                {EPOCHS.map(e => <th key={e} className="text-right py-1.5 px-1 font-medium">{e}</th>)}
              </tr>
            </thead>
            <tbody className="data-value">
              <tr className="border-b border-border-subtle">
                <td className="py-1 pr-3 text-info">3B − 770M</td>
                {size.mean.map((v, i) => <td key={i} className="text-right py-1 px-1" data-cell={`size-e${i + 1}`}>{fmt(v, 3, true)}</td>)}
              </tr>
              <tr className="border-b border-border-subtle">
                <td className="py-1 pr-3 text-muted-foreground">pooled sd</td>
                {size.pooled_sd.map((v, i) => <td key={i} className="text-right py-1 px-1 text-muted-foreground">{fmt(v, 3)}</td>)}
              </tr>
              <tr>
                <td className="py-1 pr-3 text-warning">recipe at 770M</td>
                {recipeCmp.mean.map((v, i) => <td key={i} className="text-right py-1 px-1">{fmt(v, 3, true)}</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="What the comparisons say (n = 3 per arm)">
        <ul className="flex flex-col gap-2 text-sm text-foreground leading-relaxed list-disc pl-5">
          <li>
            Under each model&apos;s own recipe, the attack effect at epoch 10 is {own('3b')} for 3B and {own('770m-adamw')} for 770M with
            AdamW. The direction matches the paper&apos;s &ldquo;larger models are more susceptible&rdquo; (Wan et al. 2023, §5, Fig. 4 left),
            but this comparison changes the recipe along with the size.
          </li>
          <li>
            Under a matched recipe, 3B − 770M is largest at epoch {peak + 1} ({fmtMeanSd(size.mean[peak], size.pooled_sd[peak], 3, true)}{' '}
            pooled sd) and {fmtMeanSd(size.mean[last], size.pooled_sd[last], 3, true)} at epoch 10 ({label('770m-adafactor')}:{' '}
            {own('770m-adafactor')}). So size changed how fast the poison was learned, not its epoch-10 level. The paper&apos;s Fig. 4
            (right) shows the effect rising with epochs for 3B and 11B.
          </li>
          <li>
            At 770M, the 3B recipe (Adafactor, batch 4, bf16 with stochastic rounding) changes the effect by up to{' '}
            {fmt(Math.max(...recipeCmp.mean), 3, true)} (epoch {recipeCmp.mean.indexOf(Math.max(...recipeCmp.mean)) + 1}). The optimizer,
            the number of updates (batch size) and the weight precision all change together, so no single one of them can be credited.
          </li>
        </ul>
        <Note tone="warning">
          n = 3 seeds per arm; the sds include prompt-sampling variance (finding 13). These are differences between attack effects, each
          already a paired poisoned − clean difference — the clean models&apos; own rates differ by recipe (epoch 10:{' '}
          {Object.keys(summary.recipes).map(r => `${label(r)} ${fmt(agg[r].clean.mean[last])}`).join('; ')}).
        </Note>
      </Card>
    </div>
  );
}
