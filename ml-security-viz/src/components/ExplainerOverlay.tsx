'use client';
import { useState, useMemo, useEffect, useRef } from 'react';
import useStore from '@/store/useStore';
import { getAlgorithm } from '@/engine/architectures';
import MathEq from './MathEq';
import ExplainerVisual from './ExplainerVisual';

/**
 * ExplainerOverlay — guided, step-by-step walkthrough of the active attack.
 *
 * The format follows Transformer Explainer: every card shows the symbolic
 * equation *and* the same equation evaluated on the frame the user is scrubbed
 * to, so the maths is bound to the running model instead of illustrating it.
 * Scrubbing the timeline re-renders the numbers, the readout meters, and the
 * vector diagrams together.
 */

const TONE: Record<string, string> = {
  clean: 'text-clean',
  attack: 'text-attack',
  warning: 'text-warning',
  neutral: 'text-foreground',
};
const METER: Record<string, string> = {
  clean: 'bg-clean/25',
  attack: 'bg-attack/25',
  warning: 'bg-warning/25',
  neutral: 'bg-primary/25',
};

export default function ExplainerOverlay() {
  const { activeAlgorithm, algorithmConfig, attackTrace, currentIteration } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const steps = alg.explainerSteps;

  const [isOpen, setIsOpen] = useState(false);
  // `null` means "follow the attack"; any number is a step the user picked.
  // Deriving the shown step rather than syncing it in an effect keeps the
  // phase-following behaviour without a cascading re-render.
  const [manualStep, setManualStep] = useState<number | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  const config = useMemo(
    () => ({ ...alg.defaultConfig, ...algorithmConfig }),
    [alg, algorithmConfig]
  );

  const frame = attackTrace && attackTrace.length > 0 ? attackTrace[currentIteration] : null;
  const currentPhase = frame?.phase;

  // Which card the attack itself is "on" — the first one describing the phase
  // of the frame the user is scrubbed to.
  const followedStep = useMemo(() => {
    if (!steps || !currentPhase) return 0;
    const idx = steps.findIndex(s => s.phase === currentPhase);
    return idx >= 0 ? idx : 0;
  }, [steps, currentPhase]);

  const followPhase = manualStep === null;
  const currentStep = Math.min(manualStep ?? followedStep, (steps?.length ?? 1) - 1);

  // Reset scroll when the card changes so long cards don't open mid-way down.
  useEffect(() => {
    bodyRef.current?.scrollTo({ top: 0 });
  }, [currentStep]);

  if (!steps || steps.length === 0) return null;

  const step = steps[currentStep];
  const goTo = (i: number) => setManualStep(i);

  const liveEquation = step.liveEquation?.(frame, config) || null;
  const readouts = step.readouts?.(frame, config) || [];

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-24 right-6 z-50 w-12 h-12 rounded-full bg-primary text-primary-foreground shadow-glow flex items-center justify-center text-lg hover:scale-105 transition-transform cursor-pointer border border-primary/50"
        title="Open step-by-step explainer"
        aria-label="Open step-by-step explainer"
      >
        📖
      </button>
    );
  }

  return (
    <div className="fixed bottom-24 right-6 z-50 w-[420px] max-w-[calc(100vw-3rem)] max-h-[min(70vh,640px)] glass-panel flex flex-col overflow-hidden animate-[fadeIn_0.3s_ease-out] shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle bg-secondary shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm shrink-0">📖</span>
          <h3 className="font-semibold text-sm text-foreground truncate">{alg.name} — walkthrough</h3>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setManualStep(followPhase ? currentStep : null)}
            title={followPhase ? 'Following the attack — click to browse freely' : 'Browsing freely — click to follow the attack'}
            className={`text-[10px] font-mono px-2 py-1 rounded-full border transition-colors duration-150 cursor-pointer ${
              followPhase
                ? 'border-primary/40 bg-primary/10 text-primary'
                : 'border-border bg-secondary text-muted-foreground hover:text-foreground'
            }`}
          >
            {followPhase ? '‹ auto' : 'manual'}
          </button>
          <button
            onClick={() => setIsOpen(false)}
            aria-label="Close explainer"
            className="text-muted-foreground/70 hover:text-foreground cursor-pointer text-lg leading-none transition-colors duration-150"
          >
            &times;
          </button>
        </div>
      </div>

      {/* Scrolling body — long cards stay reachable instead of overflowing the panel */}
      <div ref={bodyRef} className="flex-1 min-h-0 overflow-y-auto overscroll-contain p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <h4 className="font-bold text-foreground text-sm">{step.title}</h4>
          {step.phase && (
            <span className={`text-[10px] font-mono font-medium px-2 py-0.5 rounded-full ${
              step.phase === 'trigger'
                ? 'bg-attack/10 text-attack ring-1 ring-inset ring-attack/20'
                : step.phase === 'accumulative'
                ? 'bg-warning/10 text-warning ring-1 ring-inset ring-warning/20'
                : 'bg-primary/10 text-primary ring-1 ring-inset ring-primary/20'
            }`}>
              {step.phase}
            </span>
          )}
        </div>

        {step.equation && (
          <div className="bg-background rounded-md border border-border-subtle px-3 py-2.5 overflow-x-auto">
            <MathEq math={step.equation} inline={false} />
          </div>
        )}

        {/* The same equation with this frame's real numbers in it */}
        {liveEquation && (
          <div className="rounded-md border border-primary/25 bg-primary/5 px-3 py-2 overflow-x-auto">
            <div className="eyebrow mb-1 !text-primary/70">at iteration {frame?.iteration ?? 0}</div>
            <MathEq math={liveEquation} inline={false} />
          </div>
        )}

        {step.visual && (
          <ExplainerVisual kind={step.visual} frame={frame} config={config} />
        )}

        {readouts.length > 0 && (
          <div className="grid grid-cols-2 gap-1.5">
            {readouts.map(r => (
              <div
                key={r.label}
                className="relative rounded-md border border-border-subtle bg-secondary/60 px-2.5 py-1.5 overflow-hidden"
              >
                {r.meter !== undefined && (
                  <div
                    className={`absolute inset-y-0 left-0 ${METER[r.tone || 'neutral']} transition-[width] duration-500`}
                    style={{ width: `${Math.max(0, Math.min(1, r.meter)) * 100}%` }}
                  />
                )}
                <div className="relative flex items-baseline justify-between gap-2">
                  <span className="text-[10px] text-muted-foreground truncate">{r.label}</span>
                  <span className={`data-value !text-[12px] ${TONE[r.tone || 'neutral']}`}>{r.value}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        <p className="text-sm text-muted-foreground leading-relaxed">{step.description}</p>

        {!frame && (
          <div className="text-[11px] font-mono px-2 py-1.5 rounded border border-border-subtle bg-secondary/50 text-muted-foreground/80">
            Launch the attack to bind these equations to live values.
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="px-4 py-3 bg-secondary/50 border-t border-border-subtle flex items-center justify-between shrink-0">
        <button
          onClick={() => goTo(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          aria-label="Previous step"
          className="w-8 h-8 rounded-full flex items-center justify-center bg-secondary border border-border text-muted-foreground disabled:opacity-30 disabled:cursor-not-allowed hover:bg-muted hover:text-foreground cursor-pointer transition-colors duration-150"
        >
          &lt;
        </button>

        <div className="flex items-center gap-3">
          <div className="flex gap-1">
            {steps.map((s, i) => (
              <button
                key={s.id}
                onClick={() => goTo(i)}
                aria-label={`Go to step ${i + 1}: ${s.title}`}
                title={s.title}
                className={`rounded-full transition-all duration-300 cursor-pointer ${
                  i === currentStep ? 'w-4 h-1.5 bg-primary' : 'w-1.5 h-1.5 bg-muted hover:bg-muted-foreground/50'
                }`}
              />
            ))}
          </div>
          <span className="text-xs font-semibold text-primary bg-primary/10 px-2 py-0.5 rounded-full">
            {currentStep + 1} / {steps.length}
          </span>
        </div>

        <button
          onClick={() => goTo(Math.min(steps.length - 1, currentStep + 1))}
          disabled={currentStep === steps.length - 1}
          aria-label="Next step"
          className="w-8 h-8 rounded-full flex items-center justify-center bg-secondary border border-border text-muted-foreground disabled:opacity-30 disabled:cursor-not-allowed hover:bg-muted hover:text-foreground cursor-pointer transition-colors duration-150"
        >
          &gt;
        </button>
      </div>
    </div>
  );
}
