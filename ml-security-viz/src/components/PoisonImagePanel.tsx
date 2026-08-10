'use client';
import React, { useRef, useEffect } from 'react';
import useStore from '@/store/useStore';
import { pcaInverseTransform } from '@/engine/data/loaders/pca';
import { drawMNISTImage } from '@/engine/data/loaders/mnist';

export default function PoisonImagePanel({ currentState }: { currentState: any }) {
  const { datasetKey, mnistData, selectedPoint } = useStore();
  
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    // Only proceed if viewing MNIST, have valid state, and a poison point is selected
    if (datasetKey !== 'mnist' || !mnistData || !currentState || !currentState.poisonX || !selectedPoint || selectedPoint.type !== 'poison') return;

    const { mean, components } = mnistData.pcaState;
    const pt = currentState.poisonX[selectedPoint.index];

    if (pt && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        // Perform Inverse PCA (from 2D back to 784D)
        const approx = pcaInverseTransform([pt], mean, components)[0];
        
        // Scale values to [0, 255] for drawing
        const scaledApprox = approx.map((v: number) => v * 255);

        // Draw image to the 84x84 canvas. Scale 3 -> 28x28 * 3 = 84x84
        drawMNISTImage(ctx, scaledApprox, 0, 0, 3);
      }
    }
  }, [currentState, mnistData, datasetKey, selectedPoint]);

  // If not on MNIST dataset, no attack trace exists, or no poison point selected, don't render
  if (datasetKey !== 'mnist' || !mnistData || !currentState || !currentState.poisonX || !selectedPoint || selectedPoint.type !== 'poison') {
    return null;
  }

  const index = selectedPoint.index;
  const pt = currentState.poisonX[index];

  if (!pt) return null;

  return (
    <section className="glass-panel absolute top-16 right-6 z-30 w-auto rounded-md border border-border-subtle overflow-hidden flex flex-col shrink-0 animate-[fade-in_0.2s_ease-out] shadow-2xl">
      <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-secondary">
        <span className="text-[14px]">👀</span> Poison Evolution
      </h3>
      <div className="p-3 flex gap-3 bg-background">
        <div className="flex flex-col items-center gap-2 p-2 border border-border-subtle rounded-md bg-secondary">
          <canvas
            ref={canvasRef}
            width={84}
            height={84}
            className="bg-[var(--data-image-bg)] rounded-sm shadow-sm [image-rendering:pixelated]"
          />
          <div className="flex flex-col items-center gap-1 w-full text-center">
            <span className="text-[10px] text-muted-foreground/70 uppercase tracking-[0.05em]">
              Point #{index}
            </span>
            <span className="font-mono text-xs text-foreground font-medium">
              ({pt[0].toFixed(2)}, {pt[1].toFixed(2)})
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
