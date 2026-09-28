'use client';
import React, { useRef, useEffect } from 'react';
import useStore from '@/store/useStore';
import { pcaInverseTransform } from '@/engine/data/loaders/pca';
import { drawMNISTImage } from '@/engine/data/loaders/mnist';

/**
 * The selected poison point as an image. On real MNIST the attack runs in pixel space, so the
 * point IS an image and we show it before and after the attack — Fig. 2 of Biggio et al. 2012.
 * The 2-D PCA MNIST task (Pang 2021) reconstructs the image through its PCA.
 */
export default function PoisonImagePanel({ currentState }: { currentState: any }) {
  const { dataset, datasetKey, selectedPoint, attackTrace } = useStore();
  const beforeRef = useRef<HTMLCanvasElement | null>(null);
  const afterRef = useRef<HTMLCanvasElement | null>(null);

  const index = selectedPoint?.type === 'poison' ? selectedPoint.index : -1;
  const pt = index >= 0 ? currentState?.poisonX?.[index] : null;
  const pixelSpace = !!dataset?.imageShape;
  // 2-D PCA MNIST (Pang): reconstruct the image from the canvas point
  const legacy = datasetKey === 'mnist' && !!dataset?.pcaState;
  const initial = pixelSpace && index >= 0
    ? attackTrace.find((f: any) => f.poisonX.length > index)?.poisonX[index]
    : null;

  useEffect(() => {
    if (!pt) return;
    const draw = (canvas: HTMLCanvasElement | null, pixels01: number[]) => {
      const ctx = canvas?.getContext('2d');
      if (ctx) drawMNISTImage(ctx, pixels01.map(v => v * 255), 0, 0, 3);
    };
    if (pixelSpace) {
      draw(afterRef.current, pt);
      if (initial) draw(beforeRef.current, initial);
    } else if (legacy) {
      const { mean, components } = dataset.pcaState;
      draw(afterRef.current, pcaInverseTransform([pt], mean, components)[0]);
    }
  }, [pt, initial, pixelSpace, legacy, dataset]);

  if (!pt || (!pixelSpace && !legacy)) return null;

  const tileClass = 'flex flex-col items-center gap-2 p-2 border border-border-subtle rounded-md bg-secondary';
  const canvasClass = 'bg-[var(--data-image-bg)] rounded-sm shadow-sm [image-rendering:pixelated]';
  const captionClass = 'text-[10px] text-muted-foreground/70 uppercase tracking-[0.05em]';

  return (
    <section className="glass-panel absolute top-16 right-6 z-30 w-auto rounded-md border border-border-subtle overflow-hidden flex flex-col shrink-0 animate-[fade-in_0.2s_ease-out] shadow-2xl">
      <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-secondary">
        <span className="text-[14px]">👀</span> Poison Point #{index}
      </h3>
      <div className="p-3 flex gap-3 bg-background">
        {pixelSpace && initial && (
          <div className={tileClass}>
            <canvas ref={beforeRef} width={84} height={84} className={canvasClass} />
            <span className={captionClass}>Before attack</span>
          </div>
        )}
        <div className={tileClass}>
          <canvas ref={afterRef} width={84} height={84} className={canvasClass} />
          <span className={captionClass}>{pixelSpace ? `Iteration ${currentState.iteration}` : 'PCA reconstruction'}</span>
        </div>
      </div>
    </section>
  );
}
