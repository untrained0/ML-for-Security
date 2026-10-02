'use client';
import { useMemo } from 'react';

/**
 * ExplainerVisual — the animated diagrams behind the guided explainer.
 *
 * Each one draws the actual vectors from the current trace frame rather than a
 * stock illustration, so scrubbing the timeline animates the geometry: arrows
 * rotate as the gradients rotate, the alignment wedge opens past 90° exactly
 * when the attack becomes damaging.
 */

type Vec = number[] | undefined;

/** Scale a weight-space vector into SVG space, preserving relative lengths. */
function project(v: Vec, scale: number): { x: number; y: number } | null {
  if (!v || v.length < 2) return null;
  return { x: v[0] * scale, y: -v[1] * scale };
}

function longest(...vs: Vec[]): number {
  let m = 0;
  for (const v of vs) {
    if (!v || v.length < 2) continue;
    m = Math.max(m, Math.hypot(v[0], v[1]));
  }
  return m || 1;
}

function Arrow({
  to, color, label, width = 2, dashed = false, opacity = 1,
}: { to: { x: number; y: number }; color: string; label?: string; width?: number; dashed?: boolean; opacity?: number }) {
  const len = Math.hypot(to.x, to.y);
  if (len < 0.5) return null;
  const ux = to.x / len, uy = to.y / len;
  const headLen = Math.min(9, len * 0.35);
  const bx = to.x - ux * headLen, by = to.y - uy * headLen;
  const px = -uy, py = ux;
  const hw = headLen * 0.42;

  return (
    <g opacity={opacity} style={{ transition: 'opacity 300ms ease' }}>
      <line
        x1={0} y1={0} x2={bx} y2={by}
        stroke={color} strokeWidth={width} strokeLinecap="round"
        strokeDasharray={dashed ? '4 3' : undefined}
        style={{ transition: 'x2 400ms ease, y2 400ms ease' }}
      />
      <polygon
        points={`${to.x},${to.y} ${bx + px * hw},${by + py * hw} ${bx - px * hw},${by - py * hw}`}
        fill={color}
        style={{ transition: 'points 400ms ease' }}
      />
      {label && (
        <text
          x={to.x + ux * 12} y={to.y + uy * 12}
          fill={color} fontSize="9" textAnchor="middle" dominantBaseline="middle"
          className="font-mono"
        >
          {label}
        </text>
      )}
    </g>
  );
}

function Axes() {
  return (
    <g className="stroke-[var(--data-grid)]" strokeWidth={1}>
      <line x1={-95} y1={0} x2={95} y2={0} />
      <line x1={0} y1={-58} x2={0} y2={58} />
    </g>
  );
}

/** ⟨∇L(S_val), ∇L(P(S_T))⟩ — the wedge between the two gradients. */
function AlignmentVisual({ frame }: { frame: any }) {
  const gVal = frame?.gradVal as Vec;
  const gTrig = frame?.gradTrigger as Vec;
  const scale = 70 / longest(gVal, gTrig);
  const a = project(gVal, scale);
  const b = project(gTrig, scale);
  const align = frame?.gradientAlignment as number | undefined;
  const damaging = (align ?? 0) < 0;

  if (!a || !b) {
    return <Empty label="Run the attack to see the gradient geometry" />;
  }

  return (
    <svg viewBox="-100 -62 200 124" className="w-full h-[124px]">
      <Axes />
      {/* Wedge between the two gradients — fills red once past 90° */}
      <path
        d={`M 0 0 L ${a.x * 0.4} ${a.y * 0.4} A ${Math.hypot(a.x, a.y) * 0.4} ${Math.hypot(a.x, a.y) * 0.4} 0 0 1 ${b.x * 0.4} ${b.y * 0.4} Z`}
        fill={damaging ? 'var(--attack)' : 'var(--clean)'}
        opacity={0.16}
        style={{ transition: 'd 400ms ease, fill 300ms ease' }}
      />
      <Arrow to={a} color="var(--info)" label="∇val" />
      <Arrow to={b} color={damaging ? 'var(--attack)' : 'var(--clean)'} label="∇trig" />
      <text
        x={0} y={54} textAnchor="middle" fontSize="9"
        fill={damaging ? 'var(--attack)' : 'var(--clean)'} className="font-mono"
      >
        {damaging ? 'anti-aligned → the step hurts' : 'aligned → the step still helps'}
      </text>
    </svg>
  );
}

/** d_t = ∇L(S_t) + λG_t as a parallelogram. */
function AccumulationVisual({ frame, lambda }: { frame: any; lambda: number }) {
  const gBatch = frame?.gradBatch as Vec;
  const gAccum = frame?.gradAccum as Vec;
  const target = frame?.gradTarget as Vec;

  const scaled = useMemo(() => {
    if (!gBatch || !gAccum) return null;
    const lam = [gAccum[0] * lambda, gAccum[1] * lambda];
    const scale = 66 / longest(gBatch, lam, target);
    return {
      batch: project(gBatch, scale),
      accum: project(lam, scale),
      target: project(target, scale),
    };
  }, [gBatch, gAccum, target, lambda]);

  if (!scaled || !scaled.batch || !scaled.accum) {
    return <Empty label="Run the attack to see the accumulation direction" />;
  }

  return (
    <svg viewBox="-100 -62 200 124" className="w-full h-[124px]">
      <Axes />
      {/* completion of the parallelogram */}
      {scaled.target && (
        <>
          <line
            x1={scaled.batch.x} y1={scaled.batch.y} x2={scaled.target.x} y2={scaled.target.y}
            stroke="var(--muted-foreground)" strokeWidth={1} strokeDasharray="3 3" opacity={0.5}
          />
          <line
            x1={scaled.accum.x} y1={scaled.accum.y} x2={scaled.target.x} y2={scaled.target.y}
            stroke="var(--muted-foreground)" strokeWidth={1} strokeDasharray="3 3" opacity={0.5}
          />
        </>
      )}
      <Arrow to={scaled.batch} color="var(--clean)" label="∇L(Sₜ)" width={1.8} />
      <Arrow to={scaled.accum} color="var(--warning)" label="λGₜ" width={1.8} dashed />
      {scaled.target && <Arrow to={scaled.target} color="var(--attack)" label="dₜ" width={2.4} />}
      <text x={0} y={54} textAnchor="middle" fontSize="9" fill="var(--muted-foreground)" className="font-mono">
        honest update + accumulation = what the batch is bent toward
      </text>
    </svg>
  );
}

/** Predicted vs actual loss jump at the trigger. */
function TriggerVisual({ frame }: { frame: any }) {
  const predicted = frame?.predictedLossJump as number | undefined;
  const actual = frame?.actualLossJump as number | undefined;

  if (predicted === undefined || actual === undefined) {
    return <Empty label="Scrub to the trigger frame to compare forecast with reality" />;
  }

  const max = Math.max(Math.abs(predicted), Math.abs(actual), 1e-6);
  const bar = (v: number) => Math.max(2, (Math.abs(v) / max) * 130);

  return (
    <svg viewBox="0 0 200 124" className="w-full h-[124px]">
      {[{ label: 'Eq. 6 forecast', v: predicted, y: 26, color: 'var(--info)' },
        { label: 'actual Δloss', v: actual, y: 70, color: 'var(--attack)' }].map(row => (
        <g key={row.label}>
          <text x={12} y={row.y - 6} fontSize="9" fill="var(--muted-foreground)" className="font-mono">
            {row.label}
          </text>
          <rect x={12} y={row.y} width={130} height={10} rx={5} fill="var(--muted)" />
          <rect
            x={12} y={row.y} width={bar(row.v)} height={10} rx={5} fill={row.color}
            style={{ transition: 'width 500ms ease' }}
          />
          <text x={150} y={row.y + 8} fontSize="9" fill={row.color} className="font-mono">
            {row.v.toFixed(4)}
          </text>
        </g>
      ))}
      <text x={100} y={110} textAnchor="middle" fontSize="8.5" fill="var(--muted-foreground)" className="font-mono">
        first-order theory vs. what the step actually did
      </text>
    </svg>
  );
}

/** θ_t → θ_{t+1}: the online update loop. */
function OnlineStepVisual({ frame }: { frame: any }) {
  const active = frame?.phase === 'trigger' ? 'var(--attack)'
    : frame?.phase === 'accumulative' ? 'var(--warning)'
    : 'var(--primary)';

  return (
    <svg viewBox="0 0 200 124" className="w-full h-[124px]">
      {[{ x: 18, label: 'batch Sₜ', fill: 'var(--card)' },
        { x: 78, label: 'update', fill: 'var(--card)' },
        { x: 138, label: 'θₜ₊₁', fill: 'var(--card)' }].map((box, i) => (
        <g key={box.label}>
          <rect
            x={box.x} y={40} width={46} height={30} rx={6}
            fill={box.fill} stroke={active} strokeWidth={1.4}
            className={frame ? 'animate-breathe' : ''}
            style={{ animationDelay: `${i * 200}ms` }}
          />
          <text x={box.x + 23} y={59} textAnchor="middle" fontSize="9" fill="var(--foreground)" className="font-mono">
            {box.label}
          </text>
        </g>
      ))}
      {[64, 124].map(x => (
        <line
          key={x} x1={x} y1={55} x2={x + 12} y2={55}
          stroke={active} strokeWidth={1.6} markerEnd="url(#explainer-arrow)"
        />
      ))}
      <text x={100} y={96} textAnchor="middle" fontSize="8.5" fill="var(--muted-foreground)" className="font-mono">
        no review step — the batch is applied and discarded
      </text>
      <defs>
        <marker id="explainer-arrow" markerWidth="7" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0, 7 3, 0 6" fill={active} />
        </marker>
      </defs>
    </svg>
  );
}

function Empty({ label }: { label: string }) {
  return (
    <div className="h-[124px] flex items-center justify-center text-center px-4 text-[11px] text-muted-foreground/70">
      {label}
    </div>
  );
}

export default function ExplainerVisual({
  kind, frame, config,
}: { kind?: string; frame: any; config: Record<string, any> }) {
  if (!kind) return null;

  return (
    <div className="rounded-md border border-border-subtle bg-background/60 overflow-hidden">
      {kind === 'alignment' && <AlignmentVisual frame={frame} />}
      {kind === 'accumulation' && <AccumulationVisual frame={frame} lambda={config.lambda ?? 25} />}
      {kind === 'trigger' && <TriggerVisual frame={frame} />}
      {kind === 'online-step' && <OnlineStepVisual frame={frame} />}
    </div>
  );
}
