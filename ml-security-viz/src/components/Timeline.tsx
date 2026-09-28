'use client';
import { useMemo } from 'react';
import useStore from '@/store/useStore';
import useThemeTokens from '@/hooks/useThemeTokens';
import PanelToggle from './PanelToggle';
import { getAlgorithm } from '@/engine/architectures';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend
} from 'recharts';

/**
 * Timeline — Loss and accuracy/MSE charts tracking the attack progression
 */
export default function Timeline() {
  const { attackTrace, currentIteration, activeAlgorithm, showTimeline, toggleTimeline } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const isRegression = alg.modelType === 'regression';
  const t = useThemeTokens();
  const axisTick = { fill: t.mutedForeground, fontSize: 10 };

  const chartData = useMemo(() => {
    if (!attackTrace || attackTrace.length === 0) return [];
    return attackTrace.map(s => {
      if (isRegression) {
        return {
          iteration: s.iteration,
          objective: parseFloat((s.objectiveValue || 0).toFixed(5)),
          mse: parseFloat((s.poisonedMSE || 0).toFixed(4)),
          cleanMSE: parseFloat((s.cleanMSE || 0).toFixed(4)),
          deltaW: s.deltaW !== undefined ? parseFloat(s.deltaW.toFixed(4)) : 0,
        };
      } else {
        return {
          iteration: s.iteration,
          objective: parseFloat((s.objectiveValue || 0).toFixed(5)),
          heldOut: s.heldOutObjective !== undefined ? parseFloat(s.heldOutObjective.toFixed(5)) : undefined,
          point: s.activePoison,
          accuracy: parseFloat(((s.poisonedAccuracy || 0) * 100).toFixed(1)),
          cleanAcc: parseFloat(((s.cleanAccuracy || 0) * 100).toFixed(1)),
          hingeLoss: s.poisonedModel && s.poisonedModel.hingeLoss !== undefined 
            ? parseFloat(s.poisonedModel.hingeLoss.toFixed(4)) 
            : 0,
          deltaW: s.deltaW !== undefined ? parseFloat(s.deltaW.toFixed(4)) : 0,
        };
      }
    });
  }, [attackTrace, isRegression]);

  const current = chartData[Math.min(currentIteration, chartData.length - 1)];
  const hasHeldOut = chartData.some((d: any) => d.heldOut !== undefined);

  // Sequential attacks (Biggio 2012) optimise one poison point at a time: mark where each starts
  const pointStarts = useMemo(
    () => chartData
      .filter((d: any, i: number) => i > 0 && d.point !== undefined && d.point !== (chartData[i - 1] as any).point)
      .map((d: any) => ({ iteration: d.iteration, label: `#${d.point + 1}` })),
    [chartData],
  );

  if (chartData.length === 0) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between gap-2 mb-2">
          <h4 className="text-sm text-muted-foreground font-medium uppercase tracking-[0.04em]">📈 Attack Timeline</h4>
          <PanelToggle collapsed={!showTimeline} onToggle={toggleTimeline} label="attack timeline" />
        </div>
        {showTimeline && (
          <div className="flex-1 flex items-center justify-center text-muted-foreground/70 text-xs">
            <p>Run a poisoning attack to see metrics over iterations</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between gap-2 mb-2">
        <h4 className="text-sm text-muted-foreground font-medium uppercase tracking-[0.04em] shrink-0">📈 Attack Timeline</h4>
        <div className="flex gap-2 items-center min-w-0">
          {/* Read the frame the user is scrubbed to, not the first/last frame —
              these pills used to disagree with everything else on screen. */}
          {isRegression ? (
            <>
              <span className="font-mono text-[10px] font-medium text-clean bg-clean/10 px-2 py-0.5 rounded-full">Clean MSE: {current?.cleanMSE}</span>
              <span className="font-mono text-[10px] font-medium text-attack bg-attack/10 px-2 py-0.5 rounded-full">
                Poisoned MSE: {current?.mse}
              </span>
            </>
          ) : (
            <>
              <span className="font-mono text-[10px] font-medium text-clean bg-clean/10 px-2 py-0.5 rounded-full">Clean: {current?.cleanAcc}%</span>
              <span className="font-mono text-[10px] font-medium text-attack bg-attack/10 px-2 py-0.5 rounded-full">
                Poisoned: {current?.accuracy}%
              </span>
            </>
          )}
          <PanelToggle collapsed={!showTimeline} onToggle={toggleTimeline} label="attack timeline" />
        </div>
      </div>
      <div className={`flex-1 min-h-0 ${showTimeline ? '' : 'hidden'}`}>
        <ResponsiveContainer width="100%" height="100%">
          {/* Margins sized for the rotated axis labels and 2-dp tick values —
              at the old 20px right / 0px left both were sliced off. */}
          <LineChart data={chartData} margin={{ top: 6, right: 52, bottom: 4, left: 10 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={t.grid}
              vertical={false}
            />
            <XAxis
              dataKey="iteration"
              tick={axisTick}
              axisLine={{ stroke: t.borderSubtle }}
              tickLine={false}
            />
            <YAxis
              yAxisId="primary"
              orientation="left"
              tick={axisTick}
              axisLine={false}
              tickLine={false}
              width={40}
              // Zoom to the data: a 0–100 axis flattens a 95% → 94% drop into one line
              domain={isRegression ? ['auto', 'auto'] : [(min: number) => Math.max(0, Math.floor(min - 5)), 100]}
              label={{ value: isRegression ? 'Test MSE' : 'Accuracy %', angle: -90, position: 'insideLeft', offset: 14, style: { textAnchor: 'middle' }, fill: t.mutedForeground, fontSize: 10 }}
            />
            <YAxis
              yAxisId="loss"
              orientation="right"
              tick={axisTick}
              axisLine={false}
              tickLine={false}
              width={52}
              tickFormatter={(v: number) => (Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(3))}
              label={{ value: 'Objective', angle: 90, position: 'insideRight', offset: 14, style: { textAnchor: 'middle' }, fill: t.mutedForeground, fontSize: 10 }}
            />
            <YAxis
              yAxisId="deltaW"
              orientation="right"
              tick={axisTick}
              axisLine={false}
              tickLine={false}
              hide
            />
            <Tooltip
              contentStyle={{
                backgroundColor: t.popover,
                border: `1px solid ${t.border}`,
                borderRadius: '10px',
                fontSize: '12px',
                color: t.foreground,
                boxShadow: 'var(--elevation-2)',
              }}
              labelStyle={{ color: t.mutedForeground }}
              formatter={(value, name) => {
                const labels: any = {
                  accuracy: ['Poisoned Acc', '%'],
                  cleanAcc: ['Clean Acc', '%'],
                  mse: ['Poisoned MSE', ''],
                  cleanMSE: ['Clean MSE', ''],
                  objective: ['Objective', ''],
                  heldOut: ['Held-out objective', ''],
                  deltaW: ['Δw', ''],
                };
                const [label, suffix] = labels[name as string] || [name, ''];
                return [`${value}${suffix}`, label];
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: '11px', color: t.mutedForeground }}
              formatter={(value) => {
                const names: any = { 
                  accuracy: 'Poisoned Accuracy', 
                  cleanAcc: 'Clean Accuracy', 
                  mse: 'Poisoned MSE',
                  cleanMSE: 'Clean MSE',
                  objective: hasHeldOut ? 'Objective (optimised)' : 'Objective',
                  heldOut: 'Objective (held-out)',
                  deltaW: 'Δw' 
                };
                return names[value as string] || value;
              }}
            />

            {/* Poisoned metric */}
            <Line
              yAxisId="primary"
              type="monotone"
              dataKey={isRegression ? "mse" : "accuracy"}
              stroke={t.attack}
              strokeWidth={2}
              dot={(props: any) => {
                const { cx, cy, index } = props;
                if (index === currentIteration) {
                  return (
                    <circle
                      key={index}
                      cx={cx} cy={cy} r={5}
                      fill={t.attack} stroke={t.background} strokeWidth={2}
                    />
                  );
                }
                return <circle key={index} cx={cx} cy={cy} r={2} fill={t.attack} />;
              }}
              activeDot={{ r: 6, fill: t.attack, stroke: t.background, strokeWidth: 2 }}
            />

            {/* Clean baseline — drawn *after* the poisoned line so its dashes sit
                on top. Underneath, the two curves coincide whenever the attack is
                still stealthy and the clean line simply vanished. */}
            <Line
              yAxisId="primary"
              type="monotone"
              dataKey={isRegression ? "cleanMSE" : "cleanAcc"}
              stroke={t.clean}
              strokeWidth={2}
              strokeDasharray="6 4"
              dot={false}
              activeDot={false}
            />

            {/* Objective value */}
            <Line
              yAxisId="loss"
              type="monotone"
              dataKey="objective"
              stroke={t.gradient}
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 4, fill: t.gradient }}
            />

            {/* Attack loss on the attacker's held-out validation split: when it flattens while the
                optimised objective keeps rising, the attack is fitting its own sample */}
            {hasHeldOut && (
              <Line
                yAxisId="loss"
                type="monotone"
                dataKey="heldOut"
                stroke={t.info}
                strokeWidth={1.5}
                strokeDasharray="5 3"
                dot={false}
                activeDot={{ r: 4, fill: t.info }}
              />
            )}

            {pointStarts.map(p => (
              <ReferenceLine
                key={`pt-${p.iteration}`}
                x={p.iteration}
                yAxisId="primary"
                stroke={t.mutedForeground}
                strokeOpacity={0.35}
                label={{ value: p.label, position: 'insideTopLeft', fill: t.mutedForeground, fontSize: 9 }}
              />
            ))}

            {/* Delta W */}
            <Line
              yAxisId="deltaW"
              type="monotone"
              dataKey="deltaW"
              stroke={t.warning}
              strokeWidth={1.5}
              strokeDasharray="3 3"
              dot={false}
              activeDot={{ r: 4, fill: t.warning }}
            />

            {/* Current iteration marker */}
            <ReferenceLine
              x={currentIteration}
              yAxisId="primary"
              stroke={t.mutedForeground}
              strokeOpacity={0.5}
              strokeDasharray="3 3"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
