'use client';
import { useEffect, useRef } from 'react';
import useStore from '@/store/useStore';
import { predict } from '@/engine/svm';
import { drawMNISTImage } from '@/engine/mnist';

export default function TestImagePanel() {
  const {
    datasetKey, mnistData, dataset,
    cleanRawModel, poisonedRawModel,
    selectedTestIndex, setSelectedTestIndex,
  } = useStore();

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Re-draw the canvas when index changes
  useEffect(() => {
    if (!mnistData || !dataset || datasetKey !== 'mnist' || !canvasRef.current) return;
    
    const ctx = canvasRef.current.getContext('2d');
    const imagePixels = dataset.test.images[selectedTestIndex];
    if (imagePixels && ctx) {
      drawMNISTImage(ctx, imagePixels, 0, 0, 3); // 28x28 * 3 = 84x84
    }
  }, [selectedTestIndex, mnistData, dataset, datasetKey]);

  if (datasetKey !== 'mnist' || !mnistData || !dataset || !dataset.test) return null;

  const maxIndex = dataset.test.images.length - 1;
  const trueLabel = dataset.test.labels[selectedTestIndex];
  const trueY = dataset.test.y[selectedTestIndex]; // +1 (7) or -1 (1)

  const features = dataset.test.X[selectedTestIndex];

  let cleanPred: any = null;
  let poisonedPred: any = null;

  if (cleanRawModel) {
    cleanPred = predict(cleanRawModel, features);
  }
  if (poisonedRawModel) {
    poisonedPred = predict(poisonedRawModel, features);
  }

  const handlePrev = () => {
    setSelectedTestIndex(Math.max(0, selectedTestIndex - 1));
  };

  const handleNext = () => {
    setSelectedTestIndex(Math.min(maxIndex, selectedTestIndex + 1));
  };

  const formatClass = (pred: number) => pred > 0 ? '7 (+1)' : '1 (−1)';
  const isCorrect = (pred: number) => (pred > 0 && trueY > 0) || (pred < 0 && trueY < 0);

  return (
    <section className="bg-[var(--bg-tertiary)] rounded-md border border-[var(--border-subtle)] overflow-hidden flex flex-col shrink-0">
      <h3 className="text-sm font-semibold text-[var(--text-primary)] flex items-center gap-2 px-4 py-3 border-b border-[var(--border-subtle)] bg-[var(--bg-secondary)]">
        <span className="text-[14px]">🖼️</span> Test Image Predictor
      </h3>
      
      <div className="flex p-3 gap-3 items-start">
        <div className="flex flex-col items-center gap-3 p-3 border-r border-[var(--border-subtle)]">
          <canvas ref={canvasRef} width={84} height={84} className="bg-black rounded-sm shadow-[0_4px_12px_rgba(0,0,0,0.5)] [image-rendering:pixelated]" />
          <div className="flex items-center gap-3">
            <button className="bg-[var(--bg-primary)] border border-[var(--border-default)] text-[var(--text-secondary)] rounded-md w-7 h-7 flex items-center justify-center cursor-pointer transition-all duration-200 hover:not(:disabled):bg-[var(--bg-hover)] hover:not(:disabled):text-[var(--text-primary)] hover:not(:disabled):border-[var(--accent-primary)] disabled:opacity-30 disabled:cursor-not-allowed" onClick={handlePrev} disabled={selectedTestIndex === 0}>◀</button>
            <span className="font-mono text-xs text-[var(--text-tertiary)] min-w-[40px] text-center">{selectedTestIndex + 1} / {maxIndex + 1}</span>
            <button className="bg-[var(--bg-primary)] border border-[var(--border-default)] text-[var(--text-secondary)] rounded-md w-7 h-7 flex items-center justify-center cursor-pointer transition-all duration-200 hover:not(:disabled):bg-[var(--bg-hover)] hover:not(:disabled):text-[var(--text-primary)] hover:not(:disabled):border-[var(--accent-primary)] disabled:opacity-30 disabled:cursor-not-allowed" onClick={handleNext} disabled={selectedTestIndex === maxIndex}>▶</button>
          </div>
        </div>

        <div className="flex flex-col gap-4 flex-1">
          <div className="flex flex-col gap-[2px]">
            <span className="text-[11px] text-[var(--text-tertiary)] uppercase tracking-[0.05em]">True Label:</span>
            <span className="font-mono text-xs text-[var(--text-primary)] font-medium">Digit {trueLabel} (y={trueY > 0 ? '+1' : '−1'})</span>
          </div>

          <div className="flex flex-col gap-[2px]">
            <span className="text-[11px] text-[var(--text-tertiary)] uppercase tracking-[0.05em]">Clean Model:</span>
            {cleanPred !== null ? (
              <span className={`font-mono text-xs font-semibold ${isCorrect(cleanPred) ? 'text-[var(--color-clean)]' : 'text-[var(--color-attack)] animate-[shake_0.4s_ease-in-out]'}`}>
                {formatClass(cleanPred)} <br />
                <span className="text-[11px] text-[var(--text-secondary)] font-normal">(score: {cleanPred.toFixed(3)})</span>
              </span>
            ) : (
              <span className="font-mono text-sm text-[var(--text-tertiary)] italic">Not trained</span>
            )}
          </div>

          <div className="flex flex-col gap-[2px]">
            <span className="text-[11px] text-[var(--text-tertiary)] uppercase tracking-[0.05em]">Poisoned Model:</span>
            {poisonedPred !== null ? (
              <span className={`font-mono text-xs font-semibold ${isCorrect(poisonedPred) ? 'text-[var(--color-clean)]' : 'text-[var(--color-attack)] animate-[shake_0.4s_ease-in-out]'}`}>
                {formatClass(poisonedPred)} <br />
                <span className="text-[11px] text-[var(--text-secondary)] font-normal">(score: {poisonedPred.toFixed(3)})</span>
              </span>
            ) : (
              <span className="font-mono text-sm text-[var(--text-tertiary)] italic">Not attacked</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
