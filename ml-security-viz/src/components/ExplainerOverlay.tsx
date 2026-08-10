'use client';
import { useState, useMemo } from 'react';
import useStore from '@/store/useStore';
import { getAlgorithm } from '@/engine/architectures';
import MathEq from './MathEq';

/**
 * ExplainerOverlay — Transformer Explainer-style step-by-step guided learning.
 * Shows algorithm-specific explanation cards with KaTeX equations and descriptions.
 * Appears as a floating panel that can be toggled open/closed.
 */
export default function ExplainerOverlay() {
  const { activeAlgorithm, attackTrace, currentIteration } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const steps = alg.explainerSteps;

  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // Auto-detect which step to show based on the current trace phase
  const currentPhase = useMemo(() => {
    if (!attackTrace || attackTrace.length === 0) return undefined;
    const frame = attackTrace[currentIteration];
    return frame?.phase;
  }, [attackTrace, currentIteration]);

  if (!steps || steps.length === 0) {
    return null; // Algorithm doesn't have explainer steps
  }

  const step = steps[currentStep];

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-48 right-6 z-50 w-12 h-12 rounded-full bg-primary text-primary-foreground shadow-glow flex items-center justify-center text-lg hover:scale-105 transition-transform cursor-pointer border border-primary/50"
        title="Open Step-by-Step Explainer"
      >
        📖
      </button>
    );
  }

  return (
    <div className="fixed bottom-48 right-6 z-50 w-[400px] glass-panel flex flex-col overflow-hidden animate-[fadeIn_0.3s_ease-out] shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle bg-secondary">
        <div className="flex items-center gap-2">
          <span className="text-sm">📖</span>
          <h3 className="font-semibold text-sm text-foreground">Step-by-Step Explainer</h3>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="text-muted-foreground/70 hover:text-foreground cursor-pointer text-lg leading-none transition-colors duration-150"
        >
          &times;
        </button>
      </div>

      {/* Step content */}
      <div className="p-4 flex flex-col gap-3 min-h-[180px]">
        {/* Step title with phase badge */}
        <div className="flex items-center gap-2">
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

        {/* Equation */}
        {step.equation && (
          <div className="bg-background rounded-md border border-border-subtle px-3 py-2.5 overflow-x-auto">
            <MathEq math={step.equation} inline={false} />
          </div>
        )}

        {/* Description */}
        <p className="text-sm text-muted-foreground leading-relaxed">
          {step.description}
        </p>

        {/* Current phase indicator */}
        {currentPhase && (
          <div className={`text-[10px] font-mono px-2 py-1 rounded border ${
            currentPhase === 'trigger'
              ? 'border-attack/30 bg-attack/5 text-attack'
              : currentPhase === 'accumulative'
              ? 'border-warning/30 bg-warning/5 text-warning'
              : 'border-primary/30 bg-primary/5 text-primary'
          }`}>
            Currently viewing: <strong>{currentPhase}</strong> phase (iteration {currentIteration})
          </div>
        )}
      </div>

      {/* Navigation footer */}
      <div className="px-4 py-3 bg-secondary/50 border-t border-border-subtle flex items-center justify-between">
        <button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="w-8 h-8 rounded-full flex items-center justify-center bg-secondary border border-border text-muted-foreground disabled:opacity-30 disabled:cursor-not-allowed hover:bg-muted hover:text-foreground cursor-pointer transition-colors duration-150"
        >
          &lt;
        </button>

        <div className="flex items-center gap-3">
          {/* Progress dots */}
          <div className="flex gap-1">
            {steps.map((_, i) => (
              <button
                key={i}
                onClick={() => setCurrentStep(i)}
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
          onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
          disabled={currentStep === steps.length - 1}
          className="w-8 h-8 rounded-full flex items-center justify-center bg-secondary border border-border text-muted-foreground disabled:opacity-30 disabled:cursor-not-allowed hover:bg-muted hover:text-foreground cursor-pointer transition-colors duration-150"
        >
          &gt;
        </button>
      </div>
    </div>
  );
}
