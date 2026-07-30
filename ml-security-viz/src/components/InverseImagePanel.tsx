'use client';
import React, { useRef, useEffect } from 'react';
import useStore from '@/store/useStore';
import { pcaInverseTransform } from '@/engine/pca';
import { drawMNISTImage } from '@/engine/mnist';

export default function InverseImagePanel() {
  const { datasetKey, mnistData, hoveredCanvasPoint } = useStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;

    if (datasetKey !== 'mnist' || !mnistData || !hoveredCanvasPoint) {
      // Clear canvas if nothing hovered
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, 84, 84);
      return;
    }

    const z = [hoveredCanvasPoint];
    const { mean, components } = mnistData.pcaState;
    
    // Inverse PCA to get back 784 pixel intensities
    const approx = pcaInverseTransform(z, mean, components)[0];
    
    drawMNISTImage(ctx, approx, 0, 0, 3); // 84x84
  }, [hoveredCanvasPoint, mnistData, datasetKey]);

  if (datasetKey !== 'mnist' || !mnistData) return null;

  return (
    <section className="bg-[var(--bg-tertiary)] rounded-md border border-[var(--border-subtle)] overflow-hidden flex flex-col shrink-0">
      <h3 className="text-sm font-semibold text-[var(--text-primary)] flex items-center gap-2 px-4 py-3 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)]">
        <span className="text-[14px]">🔍</span> Canvas Point Viewer
      </h3>
      
      <div className="flex p-3 gap-3 items-start">
        <div className="p-3 flex justify-center bg-[var(--bg-canvas)] border-r border-[var(--border-subtle)]" style={{ borderRight: 'none', paddingRight: 0 }}>
          <canvas ref={canvasRef} width={84} height={84} className="bg-black rounded-sm shadow-[0_4px_12px_rgba(0,0,0,0.5)] [image-rendering:pixelated]" />
        </div>
        <div className="flex flex-col gap-4 flex-1">
          <div className="flex flex-col gap-[2px]">
            <span className="text-[11px] text-[var(--text-tertiary)] uppercase tracking-[0.05em]">Coordinates</span>
            {hoveredCanvasPoint ? (
              <span className="font-mono text-xs text-[var(--text-primary)] font-medium">
                ({hoveredCanvasPoint[0].toFixed(2)}, {hoveredCanvasPoint[1].toFixed(2)})
              </span>
            ) : (
              <span className="font-mono text-sm text-[var(--text-tertiary)] italic">Hover over canvas</span>
            )}
          </div>
          <div className="flex flex-col gap-[2px]" style={{ marginTop: 'auto' }}>
            <span className="text-[11px] text-[var(--text-tertiary)] uppercase tracking-[0.05em]">Operation</span>
            <span className="text-[11px] text-[var(--text-secondary)] font-normal">Inverse PCA (2D → 784D)</span>
          </div>
        </div>
      </div>
    </section>
  );
}
