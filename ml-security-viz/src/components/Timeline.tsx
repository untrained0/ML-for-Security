'use client';
import { useMemo } from 'react';
import useStore from '@/store/useStore';
import { getAlgorithm } from '@/engine/architectures';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend
} from 'recharts';

/**
 * Timeline — Loss and accuracy/MSE charts tracking the attack progression
 */
export default function Timeline() {
  const { attackTrace, currentIteration, activeAlgorithm } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const isRegression = alg.modelType === 'regression';

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

  if (chartData.length === 0) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm text-muted-foreground font-medium uppercase tracking-[0.04em]">📈 Attack Timeline</h4>
        </div>
        <div className="flex-1 flex items-center justify-center text-muted-foreground/70 text-xs">
          <p>Run a poisoning attack to see metrics over iterations</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm text-muted-foreground font-medium uppercase tracking-[0.04em]">📈 Attack Timeline</h4>
        <div className="flex gap-2">
          {isRegression ? (
            <>
              <span className="font-mono text-[10px] font-medium text-clean bg-clean/10 px-2 py-0.5 rounded-full">Clean MSE: {chartData[0]?.cleanMSE}</span>
              <span className="font-mono text-[10px] font-medium text-attack bg-attack/10 px-2 py-0.5 rounded-full">
                Poisoned MSE: {chartData[chartData.length - 1]?.mse}
              </span>
            </>
          ) : (
            <>
              <span className="font-mono text-[10px] font-medium text-clean bg-clean/10 px-2 py-0.5 rounded-full">Clean: {chartData[0]?.cleanAcc}%</span>
              <span className="font-mono text-[10px] font-medium text-attack bg-attack/10 px-2 py-0.5 rounded-full">
                Poisoned: {chartData[chartData.length - 1]?.accuracy}%
              </span>
            </>
          )}
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(148, 163, 184, 0.06)"
              vertical={false}
            />
            <XAxis
              dataKey="iteration"
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={{ stroke: 'rgba(148, 163, 184, 0.1)' }}
              tickLine={false}
            />
            <YAxis
              yAxisId="primary"
              orientation="left"
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              domain={isRegression ? ['auto', 'auto'] : [0, 100]}
              label={{ value: isRegression ? 'Test MSE' : 'Accuracy %', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }}
            />
            <YAxis
              yAxisId="loss"
              orientation="right"
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              label={{ value: 'Objective', angle: 90, position: 'insideRight', fill: '#64748b', fontSize: 10 }}
            />
            <YAxis
              yAxisId="deltaW"
              orientation="right"
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              hide
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a2332',
                border: '1px solid rgba(148,163,184,0.15)',
                borderRadius: '8px',
                fontSize: '12px',
                color: '#f1f5f9',
              }}
              labelStyle={{ color: '#94a3b8' }}
              formatter={(value, name) => {
                const labels: any = {
                  accuracy: ['Poisoned Acc', '%'],
                  cleanAcc: ['Clean Acc', '%'],
                  mse: ['Poisoned MSE', ''],
                  cleanMSE: ['Clean MSE', ''],
                  objective: ['Objective', ''],
                  deltaW: ['Δw', ''],
                };
                const [label, suffix] = labels[name as string] || [name, ''];
                return [`${value}${suffix}`, label];
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }}
              formatter={(value) => {
                const names: any = { 
                  accuracy: 'Poisoned Accuracy', 
                  cleanAcc: 'Clean Accuracy', 
                  mse: 'Poisoned MSE',
                  cleanMSE: 'Clean MSE',
                  objective: 'Objective', 
                  deltaW: 'Δw' 
                };
                return names[value as string] || value;
              }}
            />

            {/* Clean baseline */}
            <Line
              yAxisId="primary"
              type="monotone"
              dataKey={isRegression ? "cleanMSE" : "cleanAcc"}
              stroke="#10b981"
              strokeWidth={1.5}
              strokeDasharray="5 5"
              dot={false}
              activeDot={false}
            />

            {/* Poisoned metric */}
            <Line
              yAxisId="primary"
              type="monotone"
              dataKey={isRegression ? "mse" : "accuracy"}
              stroke="#ef4444"
              strokeWidth={2}
              dot={(props: any) => {
                const { cx, cy, index } = props;
                if (index === currentIteration) {
                  return (
                    <circle
                      key={index}
                      cx={cx} cy={cy} r={5}
                      fill="#ef4444" stroke="#fff" strokeWidth={2}
                    />
                  );
                }
                return <circle key={index} cx={cx} cy={cy} r={2} fill="#ef4444" />;
              }}
              activeDot={{ r: 6, fill: '#ef4444', stroke: '#fff', strokeWidth: 2 }}
            />

            {/* Objective value */}
            <Line
              yAxisId="loss"
              type="monotone"
              dataKey="objective"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              dot={false}
              activeDot={{ r: 4, fill: '#8b5cf6' }}
            />

            {/* Delta W */}
            <Line
              yAxisId="deltaW"
              type="monotone"
              dataKey="deltaW"
              stroke="#f59e0b"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              dot={false}
              activeDot={{ r: 4, fill: '#f59e0b' }}
            />

            {/* Current iteration marker */}
            <ReferenceLine
              x={currentIteration}
              yAxisId="primary"
              stroke="rgba(255,255,255,0.2)"
              strokeDasharray="3 3"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
