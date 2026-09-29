'use client';
import React, { useRef, useEffect } from 'react';
import useStore from '@/store/useStore';
import { drawImage, imageSide } from '@/engine/data/loaders/drawImage';

/**
 * The selected poison point as an image. On the real image datasets the attacks run in pixel
 * space, so the point IS an image and we show it before and after the attack — Fig. 2 of Biggio
 * et al. 2012, Fig. 4 of Pang et al. 2021. When a frame says which training image each poison
 * point was made from (`poisonSource`, Pang's per-batch perturbations) the clean original and the
 * perturbation δ are shown too.
 */
export default function PoisonImagePanel({ currentState }: { currentState: any }) {
  const { dataset, selectedPoint, attackTrace } = useStore();
  const beforeRef = useRef<HTMLCanvasElement | null>(null);
  const afterRef = useRef<HTMLCanvasElement | null>(null);
  const deltaRef = useRef<HTMLCanvasElement | null>(null);

  const index = selectedPoint?.type === 'poison' ? selectedPoint.index : -1;
  const pt: number[] | null = index >= 0 ? currentState?.poisonX?.[index] ?? null : null;
  const shape: number[] | undefined = dataset?.imageShape;
  const source = currentState?.poisonSource?.[index];
  // Biggio: the poison point's first appearance in the trace; Pang: the clean image it perturbs
  const initial: number[] | null = !shape || index < 0 ? null
    : source !== undefined ? dataset.train.X[source]
    : attackTrace.find((f: any) => f.poisonX.length > index)?.poisonX[index] ?? null;
  const showDelta = source !== undefined && !!initial;

  // Perturbation shown around mid-grey, stretched so the largest |δ| spans the full range
  const deltaMax = pt && showDelta ? Math.max(1e-9, ...pt.map((v, j) => Math.abs(v - initial[j]))) : 0;

  useEffect(() => {
    if (!pt || !shape) return;
    const draw = (canvas: HTMLCanvasElement | null, pixels01: number[]) => {
      const ctx = canvas?.getContext('2d');
      if (ctx) drawImage(ctx, pixels01.map(v => v * 255), shape, 3);
    };
    draw(afterRef.current, pt);
    if (initial) draw(beforeRef.current, initial);
    if (showDelta) draw(deltaRef.current, pt.map((v, j) => 0.5 + (0.5 * (v - initial[j])) / deltaMax));
  }, [pt, initial, shape, showDelta, deltaMax]);

  if (!pt || !shape) return null;

  const side = imageSide(shape);
  const tileClass = 'flex flex-col items-center gap-2 p-2 border border-border-subtle rounded-md bg-secondary';
  const canvasClass = 'bg-[var(--data-image-bg)] rounded-sm shadow-sm [image-rendering:pixelated]';
  const captionClass = 'text-[10px] text-muted-foreground/70 uppercase tracking-[0.05em]';

  return (
    <section className="glass-panel absolute top-16 right-6 z-30 w-auto rounded-md border border-border-subtle overflow-hidden flex flex-col shrink-0 animate-[fade-in_0.2s_ease-out] shadow-2xl">
      <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-secondary">
        <span className="text-[14px]">👀</span> Poison Point #{index}
      </h3>
      <div className="p-3 flex gap-3 bg-background">
        {initial && (
          <div className={tileClass}>
            <canvas ref={beforeRef} width={side} height={side} className={canvasClass} />
            <span className={captionClass}>{showDelta ? 'Clean' : 'Before attack'}</span>
          </div>
        )}
        <div className={tileClass}>
          <canvas ref={afterRef} width={side} height={side} className={canvasClass} />
          <span className={captionClass}>{showDelta ? 'Poisoned' : `Iteration ${currentState.iteration}`}</span>
        </div>
        {showDelta && (
          <div className={tileClass}>
            <canvas ref={deltaRef} width={side} height={side} className={canvasClass} />
            <span className={captionClass}>δ, ‖δ‖∞ = {Math.round(deltaMax * 255)}/255</span>
          </div>
        )}
      </div>
    </section>
  );
}
