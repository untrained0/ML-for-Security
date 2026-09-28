'use client';
import { useEffect, useRef } from 'react';
import useStore from '@/store/useStore';
import { getAlgorithm } from '@/engine/architectures';
import { drawMNISTImage } from '@/engine/data/loaders/mnist';
import { drawCIFARImage } from '@/engine/data/loaders/cifar';

export default function TestImagePanel() {
  const {
    datasetKey, dataset,
    cleanRawModel, poisonedRawModel, activeAlgorithm,
    selectedTestIndex, setSelectedTestIndex,
  } = useStore();
  const alg = getAlgorithm(activeAlgorithm);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Real MNIST keeps pixels in [0,1] as the features themselves; the synthetic image datasets
  // carry a separate 0–255 `images` array.
  const isCifar = datasetKey === 'cifar';
  const pixelSpace = !!dataset?.imageShape;
  const hasImages = !!dataset?.test && (pixelSpace || ((datasetKey === 'mnist' || isCifar) && !!dataset.test.images));
  const pixelsAt = (i: number) =>
    pixelSpace ? dataset.test.X[i].map((v: number) => v * 255) : dataset.test.images[i];

  // Re-draw the canvas when index changes
  useEffect(() => {
    if (!hasImages || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const imagePixels = pixelsAt(selectedTestIndex);
    if (imagePixels && ctx) {
      if (isCifar) drawCIFARImage(ctx, imagePixels, 0, 0, 3);   // 32x32 * 3
      else drawMNISTImage(ctx, imagePixels, 0, 0, 3);           // 28x28 * 3 = 84x84
    }
  }, [selectedTestIndex, dataset, datasetKey, hasImages]);

  if (!hasImages) return null;

  const maxIndex = dataset.test.X.length - 1;
  const trueLabel = dataset.test.labels[selectedTestIndex];
  const trueY = dataset.test.y[selectedTestIndex]; // +1 (7) or -1 (1)

  const features = dataset.test.X[selectedTestIndex];

  const predictFn = alg.predict;

  let cleanPred: any = null;
  let poisonedPred: any = null;

  if (cleanRawModel) {
    try {
      cleanPred = predictFn(cleanRawModel, features);
    } catch {}
  }
  if (poisonedRawModel) {
    try {
      poisonedPred = predictFn(poisonedRawModel, features);
    } catch {}
  }

  const handlePrev = () => {
    setSelectedTestIndex(Math.max(0, selectedTestIndex - 1));
  };

  const handleNext = () => {
    setSelectedTestIndex(Math.min(maxIndex, selectedTestIndex + 1));
  };

  const formatClass = (pred: number) => {
    if (dataset.classNames) return pred > 0 ? `${dataset.classNames['1']} (+1)` : `${dataset.classNames['-1']} (−1)`;
    if (datasetKey === 'mnist') return pred > 0 ? '7 (+1)' : '1 (−1)';
    if (isCifar) return pred > 0 ? 'Ship (+1)' : 'Frog (−1)';
    return pred > 0 ? '+1' : '-1';
  };
  const isCorrect = (pred: number) => (pred > 0 && trueY > 0) || (pred < 0 && trueY < 0);

  const canvasSize = isCifar ? 96 : 84; // 32x3 vs 28x3

  return (
    <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden flex flex-col shrink-0">
      <h3 className="text-sm font-semibold text-foreground flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-card">
        <span className="text-[14px]">🖼️</span> Test Image Predictor
      </h3>
      
      <div className="flex p-3 gap-3 items-start">
        <div className="flex flex-col items-center gap-3 p-3 border-r border-border-subtle">
          <canvas ref={canvasRef} width={canvasSize} height={canvasSize} className="bg-[var(--data-image-bg)] rounded-sm shadow-md [image-rendering:pixelated]" />
          <div className="flex items-center gap-3">
            <button className="bg-background border border-border text-muted-foreground rounded-md w-7 h-7 flex items-center justify-center cursor-pointer transition-all duration-200 hover:not(:disabled):bg-secondary/50 hover:not(:disabled):text-foreground hover:not(:disabled):border-primary disabled:opacity-30 disabled:cursor-not-allowed" onClick={handlePrev} disabled={selectedTestIndex === 0}>◀</button>
            <span className="font-mono text-xs text-muted-foreground/70 min-w-[40px] text-center">{selectedTestIndex + 1} / {maxIndex + 1}</span>
            <button className="bg-background border border-border text-muted-foreground rounded-md w-7 h-7 flex items-center justify-center cursor-pointer transition-all duration-200 hover:not(:disabled):bg-secondary/50 hover:not(:disabled):text-foreground hover:not(:disabled):border-primary disabled:opacity-30 disabled:cursor-not-allowed" onClick={handleNext} disabled={selectedTestIndex === maxIndex}>▶</button>
          </div>
        </div>

        <div className="flex flex-col gap-4 flex-1">
          <div className="flex flex-col gap-[2px]">
            <span className="text-[11px] text-muted-foreground/70 uppercase tracking-[0.05em]">True Label:</span>
            <span className="font-mono text-xs text-foreground font-medium">Digit {trueLabel} (y={trueY > 0 ? '+1' : '−1'})</span>
          </div>

          <div className="flex flex-col gap-[2px]">
            <span className="text-[11px] text-muted-foreground/70 uppercase tracking-[0.05em]">Clean Model:</span>
            {cleanPred !== null ? (
              <span className={`font-mono text-xs font-semibold ${isCorrect(cleanPred) ? 'text-clean' : 'text-attack animate-[shake_0.4s_ease-in-out]'}`}>
                {formatClass(cleanPred)} <br />
                <span className="text-[11px] text-muted-foreground font-normal">(score: {cleanPred.toFixed(3)})</span>
              </span>
            ) : (
              <span className="font-mono text-sm text-muted-foreground/70 italic">Not trained</span>
            )}
          </div>

          <div className="flex flex-col gap-[2px]">
            <span className="text-[11px] text-muted-foreground/70 uppercase tracking-[0.05em]">Poisoned Model:</span>
            {poisonedPred !== null ? (
              <span className={`font-mono text-xs font-semibold ${isCorrect(poisonedPred) ? 'text-clean' : 'text-attack animate-[shake_0.4s_ease-in-out]'}`}>
                {formatClass(poisonedPred)} <br />
                <span className="text-[11px] text-muted-foreground font-normal">(score: {poisonedPred.toFixed(3)})</span>
              </span>
            ) : (
              <span className="font-mono text-sm text-muted-foreground/70 italic">Not attacked</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
