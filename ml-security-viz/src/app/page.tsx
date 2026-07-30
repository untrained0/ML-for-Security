'use client';

import { useEffect, useCallback } from 'react';
import useStore from '@/store/useStore';
import { DATASETS, splitDataset } from '@/engine/datasets';
import { createKernel } from '@/engine/kernels';
import { trainSVM, getModelState } from '@/engine/svm';
import { initializePoisonPoints, poisonIteration, attackerObjective } from '@/engine/poisoning';
import { accuracy } from '@/engine/svm';
import AppHeader from '@/components/AppHeader';
import ControlPanel from '@/components/ControlPanel';
import Canvas from '@/components/Canvas';
import MathPanel from '@/components/MathPanel';
import InverseImagePanel from '@/components/InverseImagePanel';
import PlaybackBar from '@/components/PlaybackBar';
import Timeline from '@/components/Timeline';
import TutorialModal from '@/components/TutorialModal';

export default function Home() {
  const {
    datasetKey, numPoints, kernelType, kernelGamma, svmC,
    numPoison, attackEta, attackBeta, attackMaxIter, initStrategy,
    dataset, cleanModel, attackTrace, currentIteration,
    isPlaying, isTraining, isAttacking, playbackSpeed,
    showLeftPanel, showRightPanel,
    setDataset, setCleanModel, setAttackTrace, setConfig,
    stepForward, togglePlay, setPlaying, setTraining, setAttacking, reset,
  } = useStore();

  // Generate dataset
  const generateDataset = useCallback(() => {
    const gen = DATASETS[datasetKey];
    if (!gen) return;
    
    if (gen.isMNIST) {
      import('@/engine/mnist').then(({ createMNISTDataset }) => {
        import('@/engine/datasets').then(({ splitDatasetWithImages }) => {
          // Use 2D for SVM so it can be visualized on the canvas
          const raw = createMNISTDataset(numPoints, 2); 
          const split = splitDatasetWithImages(raw);
          setDataset(split, raw); // Pass raw data for pcaState
        });
      });
    } else {
      const raw = gen.fn(numPoints);
      const split = splitDataset(raw);
      setDataset(split);
    }
  }, [datasetKey, numPoints, setDataset]);

  // Train clean SVM
  const trainCleanModel = useCallback(() => {
    if (!dataset) return;
    setTraining(true);
    // Use setTimeout to avoid blocking UI
    setTimeout(() => {
      try {
        const kernelFn = createKernel({ type: kernelType, gamma: kernelGamma });
        const model = trainSVM(dataset.train.X, dataset.train.y, kernelFn, svmC);
        const state: any = getModelState(model);
        state.testAccuracy = accuracy(model, dataset.test.X, dataset.test.y);
        state.rawModel = model;
        setCleanModel(state);
      } catch (e) {
        console.error('Training error:', e);
      }
      setTraining(false);
    }, 50);
  }, [dataset, kernelType, kernelGamma, svmC, setCleanModel, setTraining]);

  // Run poisoning attack using Web Worker
  const runAttack = useCallback(() => {
    if (!dataset || !cleanModel) return;
    
    // Reset trace before streaming new attack
    setAttackTrace([]);
    setAttacking(true);

    const worker = new Worker(new URL('../engine/worker', import.meta.url));

    worker.onmessage = (e) => {
      const { type, payload } = e.data;
      
      if (type === 'PROGRESS') {
        const frame = payload;
        // Re-attach kernel function to raw model since functions can't be sent over postMessage
        if (frame.poisonedRawModel) {
          frame.poisonedRawModel.kernelFn = createKernel({ type: kernelType, gamma: kernelGamma });
        }
        useStore.getState().appendAttackTraceFrame(frame);
      } else if (type === 'COMPLETE') {
        setAttacking(false);
        worker.terminate();
      } else if (type === 'ERROR') {
        console.error('Attack error from worker:', payload);
        setAttacking(false);
        worker.terminate();
      }
    };

    worker.postMessage({
      type: 'START_ATTACK',
      payload: {
        dataset,
        cleanModelAccuracy: cleanModel.testAccuracy,
        kernelType,
        kernelGamma,
        svmC,
        numPoison,
        initStrategy,
        attackMaxIter,
        attackEta,
        attackBeta
      }
    });

  }, [dataset, cleanModel, kernelType, kernelGamma, svmC, numPoison, initStrategy, attackMaxIter, attackEta, attackBeta, setAttackTrace, setAttacking]);

  // Generate heatmap using Web Worker
  const generateHeatmap = useCallback(() => {
    if (!dataset || !cleanModel) return;
    
    useStore.getState().setConfig({ isGeneratingHeatmap: true, showHeatmap: true });

    const worker = new Worker(new URL('../engine/worker', import.meta.url));

    worker.onmessage = (e) => {
      const { type, payload } = e.data;
      
      if (type === 'HEATMAP_COMPLETE') {
        useStore.getState().setConfig({ 
          heatmapData: payload,
          isGeneratingHeatmap: false
        });
        worker.terminate();
      } else if (type === 'ERROR') {
        console.error('Heatmap error from worker:', payload);
        useStore.getState().setConfig({ isGeneratingHeatmap: false });
        worker.terminate();
      }
    };

    worker.postMessage({
      type: 'COMPUTE_HEATMAP',
      payload: {
        dataset,
        kernelType,
        kernelGamma,
        svmC,
        // Since heatmap calculation scales by grid size, we use a coarse grid (e.g., 20x20)
        gridResolution: 15,
      }
    });

  }, [dataset, cleanModel, kernelType, kernelGamma, svmC]);

  // Auto-generate dataset on first load
  useEffect(() => {
    generateDataset();
  }, [generateDataset]);

  // Auto-train when dataset changes
  useEffect(() => {
    if (dataset) trainCleanModel();
  }, [dataset, trainCleanModel]);

  // Playback auto-step
  useEffect(() => {
    if (!isPlaying || attackTrace.length === 0) return;
    const timer = setInterval(() => {
      stepForward();
    }, playbackSpeed);
    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed, stepForward, attackTrace.length]);

  // Stop playing when we reach the end
  useEffect(() => {
    if (isPlaying && currentIteration >= attackTrace.length - 1) {
      setPlaying(false);
    }
  }, [currentIteration, attackTrace.length, isPlaying, setPlaying]);

  const currentState = attackTrace[currentIteration] || null;

  const gridColsClass = showLeftPanel && showRightPanel 
    ? 'grid-cols-[280px_1fr_320px] max-[1100px]:grid-cols-[240px_1fr_260px]' 
    : showLeftPanel && !showRightPanel 
    ? 'grid-cols-[280px_1fr_0px] max-[1100px]:grid-cols-[240px_1fr_0px]'
    : !showLeftPanel && showRightPanel 
    ? 'grid-cols-[0px_1fr_320px] max-[1100px]:grid-cols-[0px_1fr_260px]'
    : 'grid-cols-[0px_1fr_0px]';

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <AppHeader />
      <PlaybackBar
        onRunAttack={runAttack}
        currentState={currentState}
        isAttacking={isAttacking}
        isTraining={isTraining}
      />
      <main className={`grid flex-1 overflow-hidden min-h-0 transition-[grid-template-columns] duration-300 ${gridColsClass}`}>
        <div className="flex flex-col h-full">
          {showLeftPanel && (
            <ControlPanel
              onGenerate={generateDataset}
              onTrain={trainCleanModel}
              onAttack={runAttack}
              onGenerateHeatmap={generateHeatmap}
            />
          )}
        </div>
        <div className="flex flex-col overflow-hidden min-h-0">
          <div className="flex-1 relative overflow-hidden bg-[var(--bg-canvas)] min-h-0">
            <Canvas currentState={currentState} />
          </div>
          <div className="h-[180px] min-h-[140px] bg-[var(--bg-secondary)] border-t border-[var(--border-subtle)] p-4">
            <Timeline />
          </div>
        </div>
        <div className="flex flex-col h-full overflow-hidden">
          <InverseImagePanel />
          <MathPanel currentState={currentState} />
        </div>
      </main>
      <TutorialModal />
    </div>
  );
}
