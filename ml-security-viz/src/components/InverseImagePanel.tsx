'use client';
import React, { useRef, useEffect } from 'react';
import useStore from '@/store/useStore';
import { pcaInverseTransform } from '@/engine/data/loaders/pca';
import { drawMNISTImage } from '@/engine/data/loaders/mnist';
import { lift } from '@/engine/data/view';

export default function InverseImagePanel() {
  const { dataset, datasetKey, hoveredCanvasPoint } = useStore();
  // Real MNIST: the canvas is a PCA plane of 784-d pixel space; lift the hovered point back.
  const pixelSpace = !!dataset?.imageShape && !!dataset?.view;
  const active = pixelSpace || (datasetKey === 'mnist' && !!dataset?.pcaState);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;

    if (!active || !hoveredCanvasPoint) {
      // Clear canvas if nothing hovered
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, 84, 84);
      return;
    }

    // Inverse PCA back to 784 pixel intensities in [0,1]
    const approx = pixelSpace
      ? lift(dataset.view, hoveredCanvasPoint)
      : pcaInverseTransform([hoveredCanvasPoint], dataset.pcaState.mean, dataset.pcaState.components)[0];

    drawMNISTImage(ctx, approx.map((v: number) => Math.min(1, Math.max(0, v)) * 255), 0, 0, 3); // 84x84
  }, [hoveredCanvasPoint, dataset, pixelSpace, active]);

  if (!active) return null;

  return (
    <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden flex flex-col shrink-0">
      <h3 className="text-sm font-semibold text-foreground flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-card">
        <span className="text-[14px]">🔍</span> Canvas Point Viewer
      </h3>
      
      <div className="flex p-3 gap-3 items-start">
        <div className="p-3 flex justify-center bg-background border-r border-border-subtle" style={{ borderRight: 'none', paddingRight: 0 }}>
          <canvas ref={canvasRef} width={84} height={84} className="bg-[var(--data-image-bg)] rounded-sm shadow-md [image-rendering:pixelated]" />
        </div>
        <div className="flex flex-col gap-4 flex-1">
          <div className="flex flex-col gap-[2px]">
            <span className="text-[11px] text-muted-foreground/70 uppercase tracking-[0.05em]">Coordinates</span>
            {hoveredCanvasPoint ? (
              <span className="font-mono text-xs text-foreground font-medium">
                ({hoveredCanvasPoint[0].toFixed(2)}, {hoveredCanvasPoint[1].toFixed(2)})
              </span>
            ) : (
              <span className="font-mono text-sm text-muted-foreground/70 italic">Hover over canvas</span>
            )}
          </div>
          <div className="flex flex-col gap-[2px]" style={{ marginTop: 'auto' }}>
            <span className="text-[11px] text-muted-foreground/70 uppercase tracking-[0.05em]">Operation</span>
            <span className="text-[11px] text-muted-foreground font-normal">Inverse PCA (2D → 784D)</span>
          </div>
        </div>
      </div>
    </section>
  );
}
