'use client';

import { useEffect, useCallback, useRef } from 'react';
import useStore, { type AttackRunner } from '@/store/useStore';
import { DATASETS, splitDataset } from '@/engine/data/datasets';
import type { AlgorithmModule } from '@/engine/architectures';
import { createKernel, kernelSpec } from '@/engine/architectures/biggio2012/kernels';
import ControlPanel from '@/components/ControlPanel';
import Canvas from '@/components/Canvas';
import MathPanel from '@/components/MathPanel';
import InverseImagePanel from '@/components/InverseImagePanel';
import PlaybackBar from '@/components/PlaybackBar';
import Timeline from '@/components/Timeline';
import ExportDialog from '@/components/ExportDialog';
import { mergedConfig } from '@/lib/config';
import { runAttackOnServer, fetchServerCompute, ServerUnavailableError } from '@/lib/serverAttack';
import ExplainerOverlay from '@/components/ExplainerOverlay';

/**
 * The workspace of a geometric attack (an AlgorithmModule): control panel, canvas, timeline, math
 * panel, playback, the guided explainer and exports. This is the former page body, unchanged
 * except that the page passes in the module (it only renders this for geometric attacks) and the
 * header lives in the page. Unmounting it — switching to a view module — stops every geometric
 * effect (dataset generation, training, the attack in flight, compute polling).
 */
export default function GeometricWorkspace({ alg }: { alg: AlgorithmModule }) {
  const {
    algorithmConfig,
    datasetKey, numPoints,
    dataset, cleanModel, attackTrace, currentIteration,
    isPlaying, isTraining, isAttacking, playbackSpeed,
    showLeftPanel, showRightPanel, showTimeline,
    setDataset, setCleanModel, setAttackTrace, setConfig,
    stepForward, togglePlay, setPlaying, setTraining, setAttacking, reset,
  } = useStore();

  // Build merged config: algorithm defaults + legacy store fields + user overrides
  const getMergedConfig = useCallback(
    () => mergedConfig(alg, { ...useStore.getState(), algorithmConfig }),
    [alg, algorithmConfig],
  );

  // Generate dataset.
  //
  // `alg` is a real dependency, not decoration: switching algorithms
  // clears the dataset in the store, and without it in this closure the
  // regeneration effect below never re-fires — the app sits on "Generating
  // dataset…" forever and every downstream action silently no-ops.
  const generateDataset = useCallback(() => {
    // Algorithms declare which datasets they support; a regression attack can't
    // use `moons`. Fall back to the first supported key when the current one
    // doesn't apply to the newly selected algorithm.
    const key = alg.datasets.includes(datasetKey) ? datasetKey : alg.datasets[0];
    if (key !== datasetKey) {
      setConfig({ datasetKey: key });
      return; // this re-runs with the corrected key
    }

    const gen = DATASETS[key];
    if (!gen) return;
    setConfig({ datasetError: null });

    if (gen.load) {
      // Benchmark datasets arrive pre-split (paper-specific sizes) and may be fetched async.
      Promise.resolve(gen.load(numPoints))
        .then(ds => setDataset({ displayRange: gen.displayRange, ...ds }))
        .catch(err => {
          console.error('Dataset load error:', err);
          setConfig({ datasetError: String(err?.message ?? err) });
        });
    } else {
      // An algorithm may ask for larger validation/test sets than 20% each (see `evalPoints`)
      const extra = !gen.isRegression && alg.evalPoints ? alg.evalPoints : 0;
      const nTrain = Math.round(numPoints * 0.6);
      const raw = gen.fn(extra ? nTrain + 2 * extra : numPoints);
      const ratios = extra
        ? [nTrain / (nTrain + 2 * extra), extra / (nTrain + 2 * extra)]
        : (gen.split ?? []);
      const split = splitDataset(raw, ...ratios);
      setDataset({ displayRange: gen.displayRange, ...split });
    }

  }, [alg, datasetKey, numPoints, setDataset, setConfig]);

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

  // The attack in flight, so that a new attack, a new dataset or an algorithm switch stops it
  const attackAbort = useRef<AbortController | null>(null);

  // Run poisoning attack — on the attack server (GPU when it has one) or in this browser.
  //
  // 'auto' tries the server and falls back to the browser only if the server cannot be used at
  // all (unreachable, or failing before its first frame); 'server' reports that as an error
  // instead. Both paths run the same engine code and feed the same appendAttackTraceFrame.
  const runAttack = useCallback(() => {
    if (!dataset || !cleanModel) return;
    attackAbort.current?.abort();
    const ac = new AbortController();
    attackAbort.current = ac;

    setAttackTrace([]);
    setAttacking(true);

    const config = getMergedConfig();
    const target = useStore.getState().computeTarget;
    // The exact settings of this run, for exports (the panel may be changed afterwards)
    setConfig({ attackMeta: { config, datasetKey, startedAt: new Date().toISOString() }, attackRunner: null });

    // Biggio frames arrive without kernelFn (functions cross neither postMessage nor JSON); attach
    // the same kernel, incl. the data-scaled γ, that the attack used
    const kernelFn = alg.key === 'biggio2012' ? createKernel(kernelSpec(config, dataset.train.X)) : null;
    const append = (frame: any) => {
      if (ac.signal.aborted) return;
      if (kernelFn && frame.poisonedRawModel) frame.poisonedRawModel.kernelFn = kernelFn;
      useStore.getState().appendAttackTraceFrame(frame);
    };
    const updateRunner = (patch: Partial<AttackRunner>) => {
      if (ac.signal.aborted) return;
      const cur = useStore.getState().attackRunner;
      setConfig({ attackRunner: { ...(cur as AttackRunner), ...patch } });
    };
    const finish = () => {
      if (attackAbort.current === ac) setAttacking(false);
    };
    const fail = (msg: string) => {
      console.error('Attack error:', msg);
      updateRunner({ error: msg });
      finish();
    };

    const runInBrowser = (fallbackReason?: string) => {
      const started = performance.now();
      setConfig({ attackRunner: { where: 'browser', backend: 'cpu', device: 'this browser', fallbackReason } });
      const done = () => { updateRunner({ ms: Math.round(performance.now() - started) }); finish(); };

      if (alg.key === 'biggio2012') {
        // Web Worker keeps the page responsive during the SVM retraining loop
        const worker = new Worker(new URL('../engine/worker', import.meta.url));
        ac.signal.addEventListener('abort', () => worker.terminate(), { once: true });
        worker.onmessage = (e) => {
          const { type, payload } = e.data;
          if (type === 'PROGRESS') append(payload);
          else if (type === 'COMPLETE') { worker.terminate(); done(); }
          else if (type === 'ERROR') { worker.terminate(); fail(String(payload)); }
        };
        worker.postMessage({
          type: 'START_ATTACK',
          payload: { dataset, cleanModelAccuracy: cleanModel.testAccuracy, config },
        });
      } else {
        // Main thread; the module yields between steps with setTimeout so React can paint
        alg.runAttack(dataset, cleanModel, config, append, done, fail, ac.signal);
      }
    };

    if (target === 'browser') {
      runInBrowser();
      return;
    }

    let gotFrame = false;
    runAttackOnServer({
      algorithm: alg.key, datasetKey, dataset, cleanModel, config, signal: ac.signal,
      onMeta: (m) => setConfig({ attackRunner: { where: 'server', backend: m.backend, device: m.device } }),
      onFrame: (f) => { gotFrame = true; append(f); },
    })
      .then(({ ms }) => {
        if (ac.signal.aborted) return;
        updateRunner({ ms });
        finish();
      })
      .catch((err: any) => {
        if (ac.signal.aborted) return;
        if (target === 'auto' && !gotFrame && err instanceof ServerUnavailableError) {
          console.warn(`${err.message} — running the attack in the browser instead`);
          runInBrowser(err.message);
          return;
        }
        fail(String(err?.message ?? err));
      });
  }, [dataset, datasetKey, cleanModel, alg, getMergedConfig, setAttackTrace, setAttacking, setConfig]);

  // A new dataset (also after an algorithm switch) makes any running attack meaningless
  useEffect(() => () => {
    const ac = attackAbort.current;
    if (ac && !ac.signal.aborted) {
      ac.abort();
      useStore.getState().setAttacking(false);
    }
  }, [dataset]);

  // Ask once which backend the attack server has (GPU or CPU), for the control panel
  useEffect(() => {
    fetchServerCompute().then(info => setConfig({
      serverCompute: info
        ? {
            reachable: true, backend: info.backend as 'cpu' | 'cuda', device: info.device,
            cudaReason: info.cuda?.available ? undefined : info.cuda?.reason, error: info.error,
          }
        : { reachable: false },
    }));
  }, [setConfig]);

  // Generate heatmap (SVM only)
  const generateHeatmap = useCallback(() => {
    // Fig. 1 of Biggio 2012: the surface over every 2-D attack location. Undefined for
    // high-dimensional data (MNIST), where the canvas is only a PCA view.
    if (!dataset || !cleanModel || alg.key !== 'biggio2012' || dataset.view) return;
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
      payload: { dataset, config, gridResolution: 15 },
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
    <>
      <PlaybackBar
        onRunAttack={runAttack}
        currentState={currentState}
        isAttacking={isAttacking}
        isTraining={isTraining}
      />
      <main className={`grid flex-1 overflow-hidden min-h-0 transition-[grid-template-columns] duration-300 ${gridColsClass}`}>
        <div className="flex flex-col h-full min-h-0 overflow-hidden">
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
          {/* The plot keeps a floor so the strips below can never squeeze it away */}
          <div className="flex-1 relative overflow-hidden bg-background min-h-[220px]">
            <Canvas currentState={currentState} />
          </div>
          {/* Collapsed strips keep their header row so they can be reopened —
              the height comes off the panel and goes to the plot. */}
          <div className={`bg-secondary border-t border-border-subtle px-4 overflow-hidden transition-[height] duration-300 ${
            showTimeline ? 'h-[180px] min-h-[140px] shrink py-4' : 'h-auto shrink-0 py-2.5'
          }`}>
            <Timeline />
          </div>
        </div>
        <div className="flex flex-col h-full min-h-0 overflow-hidden">
          <InverseImagePanel />
          <MathPanel currentState={currentState} />
        </div>
      </main>
      <ExplainerOverlay />
      <ExportDialog />
    </>
  );
}
