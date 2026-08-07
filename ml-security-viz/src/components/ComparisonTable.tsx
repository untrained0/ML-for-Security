'use client';
import { useMemo } from 'react';
import useStore from '@/store/useStore';
import { computeMetrics } from '@/engine/metrics';

export default function ComparisonTable() {
  const { dataset, cleanRawModel, poisonedRawModel } = useStore();

  const metrics = useMemo(() => {
    if (!dataset || !dataset.test) return null;
    
    let clean = null;
    if (cleanRawModel) {
      clean = computeMetrics(cleanRawModel, dataset.test.X, dataset.test.y);
    }
    
    let poisoned = null;
    if (poisonedRawModel) {
      poisoned = computeMetrics(poisonedRawModel, dataset.test.X, dataset.test.y);
    }
    
    return { clean, poisoned };
  }, [dataset, cleanRawModel, poisonedRawModel]);

  if (!metrics || (!metrics.clean && !metrics.poisoned)) return null;

  return (
    <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0 mt-3">
      <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
        📊 Performance Comparison
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="bg-card border-b border-border-subtle text-muted-foreground">
            <tr>
              <th className="px-4 py-2 font-medium">Metric</th>
              <th className="px-4 py-2 font-medium text-clean">Clean</th>
              <th className="px-4 py-2 font-medium text-attack">Poisoned</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)] text-foreground">
            <tr className="hover:bg-muted/10 transition-colors">
              <td className="px-4 py-2 text-muted-foreground">Accuracy</td>
              <td className="px-4 py-2 font-mono">{metrics.clean ? (metrics.clean.accuracy * 100).toFixed(1) + '%' : '-'}</td>
              <td className="px-4 py-2 font-mono">{metrics.poisoned ? (metrics.poisoned.accuracy * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
            <tr className="hover:bg-muted/10 transition-colors">
              <td className="px-4 py-2 text-muted-foreground">Precision</td>
              <td className="px-4 py-2 font-mono">{metrics.clean ? (metrics.clean.precision * 100).toFixed(1) + '%' : '-'}</td>
              <td className="px-4 py-2 font-mono">{metrics.poisoned ? (metrics.poisoned.precision * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
            <tr className="hover:bg-muted/10 transition-colors">
              <td className="px-4 py-2 text-muted-foreground">Recall</td>
              <td className="px-4 py-2 font-mono">{metrics.clean ? (metrics.clean.recall * 100).toFixed(1) + '%' : '-'}</td>
              <td className="px-4 py-2 font-mono">{metrics.poisoned ? (metrics.poisoned.recall * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
            <tr className="hover:bg-muted/10 transition-colors">
              <td className="px-4 py-2 text-muted-foreground">F1 Score</td>
              <td className="px-4 py-2 font-mono">{metrics.clean ? (metrics.clean.f1 * 100).toFixed(1) + '%' : '-'}</td>
              <td className="px-4 py-2 font-mono">{metrics.poisoned ? (metrics.poisoned.f1 * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
