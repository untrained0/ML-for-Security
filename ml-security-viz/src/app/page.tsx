'use client';

import { useEffect, useCallback } from 'react';
import useStore from '@/store/useStore';
import { DATASETS, splitDataset } from '@/engine/data/datasets';
import { getAlgorithm } from '@/engine/architectures';
import { createKernel } from '@/engine/architectures/biggio2012/kernels';
import AppHeader from '@/components/AppHeader';
import ControlPanel from '@/components/ControlPanel';
import Canvas from '@/components/Canvas';
import MathPanel from '@/components/MathPanel';
import InverseImagePanel from '@/components/InverseImagePanel';
import PlaybackBar from '@/components/PlaybackBar';
import Timeline from '@/components/Timeline';
import TutorialModal from '@/components/TutorialModal';
import GuidedTour from '@/components/GuidedTour';

export default function Home() {
  const {
    activeAlgorithm, algorithmConfig,
    datasetKey, numPoints,
    dataset, cleanModel, attackTrace, currentIteration,
    isPlaying, isTraining, isAttacking, playbackSpeed,
    showLeftPanel, showRightPanel,
    setDataset, setCleanModel, setAttackTrace, setConfig,
    stepForward, togglePlay, setPlaying, setTraining, setAttacking, reset,
  } = useStore();

  const alg = getAlgorithm(activeAlgorithm);

  // Build merged config: algorithm defaults + legacy store fields + user overrides
  const getMergedConfig = useCallback(() => {
    const state = useStore.getState() as any;
    const merged = { ...alg.defaultConfig };
    // Overlay legacy store fields
    for (const field of alg.configSchema) {
      if (state[field.key] !== undefined) merged[field.key] = state[field.key];
    }
    // Overlay explicit algorithmConfig
    Object.assign(merged, algorithmConfig);
    return merged;
  }, [alg, algorithmConfig]);

  // Generate dataset
  const generateDataset = useCallback(() => {
    const gen = DATASETS[datasetKey];
    if (!gen) return;
    
    if (gen.isMNIST) {
      import('@/engine/data/loaders/mnist').then(({ createMNISTDataset }) => {
        import('@/engine/data/datasets').then(({ splitDatasetWithImages }) => {
          const raw = createMNISTDataset(numPoints, 2); 
          const split = splitDatasetWithImages(raw);
          setDataset(split, raw);
        });
      });
    } else {
      const raw = gen.fn(numPoints);
      const split = splitDataset(raw);
      setDataset(split);
    }
  }, [datasetKey, numPoints, setDataset]);

  // Train clean model — dispatches to the active algorithm
  const trainCleanModel = useCallback(() => {
    if (!dataset) return;
    setTraining(true);
    setTimeout(() => {
      try {
        const config = getMergedConfig();
        const { modelState, rawModel } = alg.trainClean(dataset, config);
        modelState.rawModel = rawModel;
        setCleanModel(modelState);
      } catch (e) {
        console.error('Training error:', e);
      }
      setTraining(false);
    }, 50);
  }, [dataset, alg, getMergedConfig, setCleanModel, setTraining]);

  // Run poisoning attack — dispatches to the active algorithm
  const runAttack = useCallback(() => {
    if (!dataset || !cleanModel) return;
    
    setAttackTrace([]);
    setAttacking(true);

    const config = getMergedConfig();

    // For biggio2012, if using web worker for SVM attack, keep the old worker path
    // For new algorithms, use the module's runAttack directly
    if (alg.key === 'biggio2012') {
      // Use web worker for SVM to avoid blocking UI (existing behavior)
      const worker = new Worker(new URL('../engine/worker', import.meta.url));

      worker.onmessage = (e) => {
        const { type, payload } = e.data;
        
        if (type === 'PROGRESS') {
          const frame = payload;
          if (frame.poisonedRawModel) {
            frame.poisonedRawModel.kernelFn = createKernel({ type: config.kernelType, gamma: config.kernelGamma });
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
          kernelType: config.kernelType,
          kernelGamma: config.kernelGamma,
          svmC: config.svmC,
          numPoison: config.numPoison,
          initStrategy: config.initStrategy,
          attackMaxIter: config.attackMaxIter,
          attackEta: config.attackEta,
          attackBeta: config.attackBeta,
        }
      });
    } else {
      // Use the algorithm module's runAttack directly (runs async via setTimeout)
      alg.runAttack(
        dataset,
        cleanModel,
        config,
        (frame) => {
          useStore.getState().appendAttackTraceFrame(frame);
        },
        () => {
          setAttacking(false);
        },
        (msg) => {
          console.error('Attack error:', msg);
          setAttacking(false);
        }
      );
    }
  }, [dataset, cleanModel, alg, getMergedConfig, setAttackTrace, setAttacking]);

  // Generate heatmap (SVM only)
  const generateHeatmap = useCallback(() => {
    if (!dataset || !cleanModel || alg.modelType !== 'classification') return;
    const config = getMergedConfig();
    
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
        kernelType: config.kernelType,
        kernelGamma: config.kernelGamma,
        svmC: config.svmC,
        gridResolution: 15,
      }
    });

  }, [dataset, cleanModel, alg, getMergedConfig]);

  // Auto-generate dataset on first load and when algorithm changes
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
          <div className="flex-1 relative overflow-hidden bg-background min-h-0">
            <Canvas currentState={currentState} />
          </div>
          <div className="h-[180px] min-h-[140px] bg-secondary border-t border-border-subtle p-4">
            <Timeline />
          </div>
        </div>
        <div className="flex flex-col h-full overflow-hidden">
          <InverseImagePanel />
          <MathPanel currentState={currentState} />
        </div>
      </main>
      <TutorialModal />
      <GuidedTour />
    </div>
  );
}
