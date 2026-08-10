'use client';
import { useMemo } from 'react';
import useStore from '@/store/useStore';
import useThemeTokens from '@/hooks/useThemeTokens';
import { getAlgorithm } from '@/engine/architectures';

/**
 * DataFlowDiagram — Animated SVG data flow visualization
 * Shows clean vs. poisoned data pipeline as an interactive diagram.
 * Inspired by Transformer Explainer's component-level drill-down.
 */
export default function DataFlowDiagram({ currentState }: { currentState?: any }) {
  const { activeAlgorithm, attackTrace, currentIteration } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const isOnline = alg.key === 'pang2021';

  const phase = currentState?.phase || 'baseline';
  const hasAttack = attackTrace.length > 0;

  // Live design tokens — follows the active theme
  const t = useThemeTokens();
  const colors = {
    clean: t.clean,
    attack: t.attack,
    warning: t.warning,
    primary: t.primary,
    muted: t.mutedForeground,
    card: t.card,
    bg: t.background,
    border: t.border,
    foreground: t.foreground,
  };

  const phaseColor = phase === 'trigger' ? colors.attack
    : phase === 'accumulative' ? colors.warning
    : colors.primary;

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex items-center gap-2 mb-2">
        <h4 className="eyebrow">🔀 Data Flow</h4>
        {hasAttack && (
          <span className={`text-[10px] font-mono font-medium px-2 py-0.5 rounded-full ${
            phase === 'trigger' ? 'bg-attack/10 text-attack'
            : phase === 'accumulative' ? 'bg-warning/10 text-warning'
            : 'bg-primary/10 text-primary'
          }`}>
            {phase}
          </span>
        )}
      </div>

      <svg viewBox="0 0 600 140" className="w-full flex-1" style={{ minHeight: 100 }}>
        {/* Background */}
        <rect width="600" height="140" rx="8" fill={colors.bg} />

        {/* ── Clean Pipeline (top) ── */}
        <g transform="translate(0, 15)">
          {/* Data source */}
          <rect x="20" y="10" width="90" height="36" rx="6" fill={colors.card} stroke={colors.clean} strokeWidth="1.5" />
          <text x="65" y="33" textAnchor="middle" fill={colors.foreground} fontSize="10" className="font-sans">Clean Data</text>

          {/* Arrow */}
          <line x1="115" y1="28" x2="155" y2="28" stroke={colors.clean} strokeWidth="1.5" markerEnd="url(#arrow-clean)" />

          {/* Model */}
          <rect x="160" y="10" width="90" height="36" rx="6" fill={colors.card} stroke={colors.clean} strokeWidth="1.5" />
          <text x="205" y="28" textAnchor="middle" fill={colors.foreground} fontSize="10" className="font-sans">
            {isOnline ? 'SGD Step' : 'Train'}
          </text>
          <text x="205" y="39" textAnchor="middle" fill={colors.muted} fontSize="8" className="font-mono">θ_clean</text>

          {/* Arrow */}
          <line x1="255" y1="28" x2="295" y2="28" stroke={colors.clean} strokeWidth="1.5" markerEnd="url(#arrow-clean)" />

          {/* Prediction */}
          <rect x="300" y="10" width="90" height="36" rx="6" fill={colors.card} stroke={colors.clean} strokeWidth="1.5" />
          <text x="345" y="28" textAnchor="middle" fill={colors.foreground} fontSize="10" className="font-sans">Predict</text>
          <text x="345" y="39" textAnchor="middle" fill={colors.clean} fontSize="8" className="font-mono">
            {currentState ? `${((currentState.cleanAccuracy || 0) * 100).toFixed(0)}%` : '—'}
          </text>

          <text x="420" y="32" fill={colors.clean} fontSize="11" className="font-sans" fontWeight="600">✓ Clean</text>
        </g>

        {/* ── Poisoned Pipeline (bottom) ── */}
        <g transform="translate(0, 75)">
          {/* Poisoned data source */}
          <rect x="20" y="10" width="90" height="36" rx="6" fill={colors.card} stroke={phaseColor} strokeWidth="1.5"
            strokeDasharray={phase === 'accumulative' ? '4 2' : 'none'} />
          <text x="65" y="28" textAnchor="middle" fill={colors.foreground} fontSize="10" className="font-sans">
            {isOnline ? (phase === 'trigger' ? 'Trigger' : 'Batch + δ') : 'Poisoned'}
          </text>
          <text x="65" y="39" textAnchor="middle" fill={phaseColor} fontSize="8" className="font-mono">
            {isOnline ? `‖δ‖∞ ≤ ε` : `+poison`}
          </text>

          {/* Arrow (animated during attack) */}
          <line x1="115" y1="28" x2="155" y2="28" stroke={phaseColor} strokeWidth="1.5" markerEnd="url(#arrow-attack)"
            className={hasAttack ? 'animate-breathe' : ''} />

          {/* Poisoned model */}
          <rect x="160" y="10" width="90" height="36" rx="6" fill={colors.card} stroke={phaseColor} strokeWidth="1.5" />
          <text x="205" y="28" textAnchor="middle" fill={colors.foreground} fontSize="10" className="font-sans">
            {isOnline ? 'SGD Step' : 'Retrain'}
          </text>
          <text x="205" y="39" textAnchor="middle" fill={phaseColor} fontSize="8" className="font-mono">
            {isOnline ? 'θ_t+1' : 'θ_poison'}
          </text>

          {/* Arrow */}
          <line x1="255" y1="28" x2="295" y2="28" stroke={phaseColor} strokeWidth="1.5" markerEnd="url(#arrow-attack)" />

          {/* Degraded prediction */}
          <rect x="300" y="10" width="90" height="36" rx="6" fill={colors.card} stroke={phaseColor} strokeWidth="1.5" />
          <text x="345" y="28" textAnchor="middle" fill={colors.foreground} fontSize="10" className="font-sans">Predict</text>
          <text x="345" y="39" textAnchor="middle" fill={phaseColor} fontSize="8" className="font-mono">
            {currentState ? `${((currentState.poisonedAccuracy || currentState.poisonedModel?.testAccuracy || 0) * 100).toFixed(0)}%` : '—'}
          </text>

          <text x="420" y="32" fill={phaseColor} fontSize="11" className="font-sans" fontWeight="600">
            {phase === 'trigger' ? '☠ Triggered' : phase === 'accumulative' ? '⚠ Stealth' : '☠ Poisoned'}
          </text>
        </g>

        {/* Online learning: show batch counter */}
        {isOnline && hasAttack && (
          <g transform="translate(470, 55)">
            <rect x="0" y="0" width="120" height="32" rx="6" fill={colors.card} stroke={colors.border} strokeWidth="1" />
            <text x="60" y="14" textAnchor="middle" fill={colors.muted} fontSize="8" className="font-mono">
              Batch {currentState?.batchIndex ?? 0} / {attackTrace.length - 1}
            </text>
            <text x="60" y="26" textAnchor="middle" fill={colors.foreground} fontSize="9" className="font-mono">
              Drift: {(currentState?.accumulatedDrift || 0).toFixed(3)}
            </text>
          </g>
        )}

        {/* Arrow markers */}
        <defs>
          <marker id="arrow-clean" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill={colors.clean} />
          </marker>
          <marker id="arrow-attack" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill={phaseColor} />
          </marker>
        </defs>
      </svg>
    </div>
  );
}
