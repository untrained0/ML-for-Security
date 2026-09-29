'use client';
import React, { useRef, useEffect } from 'react';
import useStore from '@/store/useStore';
import { drawImage, imageSide } from '@/engine/data/loaders/drawImage';
import { lift } from '@/engine/data/view';

export default function InverseImagePanel() {
  const { dataset, hoveredCanvasPoint } = useStore();
  // Real MNIST / CIFAR-10: the canvas is a PCA plane of pixel space; lift the hovered point back.
  const active = !!dataset?.imageShape && !!dataset?.view;
  const shape: number[] | undefined = dataset?.imageShape;
  const side = imageSide(shape);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;

    if (!active || !hoveredCanvasPoint) {
      // Clear canvas if nothing hovered
      ctx.clearRect(0, 0, side, side);
      return;
    }

    // Inverse PCA back to pixel intensities in [0,1]
    const approx = lift(dataset.view, hoveredCanvasPoint);
    drawImage(ctx, approx.map((v: number) => Math.min(1, Math.max(0, v)) * 255), shape, 3);
  }, [hoveredCanvasPoint, dataset, active, shape, side]);

  if (!active) return null;

  return (
    <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden flex flex-col shrink-0">
      <h3 className="text-sm font-semibold text-foreground flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-card">
        <span className="text-[14px]">🔍</span> Canvas Point Viewer
      </h3>
      
      <div className="flex p-3 gap-3 items-start">
        <div className="p-3 flex justify-center bg-background border-r border-border-subtle" style={{ borderRight: 'none', paddingRight: 0 }}>
          <canvas ref={canvasRef} width={side} height={side} className="bg-[var(--data-image-bg)] rounded-sm shadow-md [image-rendering:pixelated]" />
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
            <span className="text-[11px] text-muted-foreground font-normal">Inverse PCA (2D → {dataset.train.X[0].length}D)</span>
          </div>
        </div>
      </div>
    </section>
  );
}
