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
    <section className="bg-[var(--bg-tertiary)] rounded-md border border-[var(--border-subtle)] overflow-hidden shrink-0 mt-3">
      <h3 className="text-sm font-semibold text-[var(--text-primary)] px-4 py-3 border-b border-[var(--border-subtle)]">
        📊 Performance Comparison
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="bg-[var(--bg-secondary)] border-b border-[var(--border-subtle)] text-[var(--text-secondary)]">
            <tr>
              <th className="px-4 py-2 font-medium">Metric</th>
              <th className="px-4 py-2 font-medium text-[var(--color-clean)]">Clean</th>
              <th className="px-4 py-2 font-medium text-[var(--color-attack)]">Poisoned</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-subtle)] text-[var(--text-primary)]">
            <tr className="hover:bg-[rgba(255,255,255,0.02)] transition-colors">
              <td className="px-4 py-2 text-[var(--text-secondary)]">Accuracy</td>
              <td className="px-4 py-2 font-mono">{metrics.clean ? (metrics.clean.accuracy * 100).toFixed(1) + '%' : '-'}</td>
              <td className="px-4 py-2 font-mono">{metrics.poisoned ? (metrics.poisoned.accuracy * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
            <tr className="hover:bg-[rgba(255,255,255,0.02)] transition-colors">
              <td className="px-4 py-2 text-[var(--text-secondary)]">Precision</td>
              <td className="px-4 py-2 font-mono">{metrics.clean ? (metrics.clean.precision * 100).toFixed(1) + '%' : '-'}</td>
              <td className="px-4 py-2 font-mono">{metrics.poisoned ? (metrics.poisoned.precision * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
            <tr className="hover:bg-[rgba(255,255,255,0.02)] transition-colors">
              <td className="px-4 py-2 text-[var(--text-secondary)]">Recall</td>
              <td className="px-4 py-2 font-mono">{metrics.clean ? (metrics.clean.recall * 100).toFixed(1) + '%' : '-'}</td>
              <td className="px-4 py-2 font-mono">{metrics.poisoned ? (metrics.poisoned.recall * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
            <tr className="hover:bg-[rgba(255,255,255,0.02)] transition-colors">
              <td className="px-4 py-2 text-[var(--text-secondary)]">F1 Score</td>
              <td className="px-4 py-2 font-mono">{metrics.clean ? (metrics.clean.f1 * 100).toFixed(1) + '%' : '-'}</td>
              <td className="px-4 py-2 font-mono">{metrics.poisoned ? (metrics.poisoned.f1 * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
