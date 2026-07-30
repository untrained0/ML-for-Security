/**
 * useStore.js — Zustand store: single source of truth
 */

import { create } from 'zustand';

export interface StoreState {
  datasetKey: string;
  numPoints: number;
  kernelType: string;
  kernelGamma: number;
  svmC: number;
  numPoison: number;
  attackEta: number;
  attackBeta: number;
  attackMaxIter: number;
  initStrategy: string;
  dataset: any;
  mnistData: any;
  cleanModel: any;
  cleanRawModel: any;
  poisonedRawModel: any;
  poisonedModel: any;
  attackTrace: any[];
  currentIteration: number;
  selectedTestIndex: number;
  hoveredCanvasPoint: [number, number] | null;
  showLeftPanel: boolean;
  showRightPanel: boolean;
  isPlaying: boolean;
  isTraining: boolean;
  isAttacking: boolean;
  playbackSpeed: number;
  showGradients: boolean;
  showHeatmap: boolean;
  isGeneratingHeatmap: boolean;
  heatmapData: any;
  showTutorial: boolean;
  
  setConfig: (updates: Partial<StoreState>) => void;
  setShowTutorial: (show: boolean) => void;
  toggleLeftPanel: () => void;
  toggleRightPanel: () => void;
  setDataset: (dataset: any, mnistData?: any) => void;
  setCleanModel: (model: any) => void;
  setSelectedTestIndex: (idx: number) => void;
  setHoveredCanvasPoint: (pt: [number, number] | null) => void;
  setAttackTrace: (trace: any[]) => void;
  appendAttackTraceFrame: (frame: any) => void;
  setIteration: (iter: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  togglePlay: () => void;
  setPlaying: (v: boolean) => void;
  setTraining: (v: boolean) => void;
  setAttacking: (v: boolean) => void;
  reset: () => void;
}

const useStore = create<StoreState>((set, get) => ({
  // ── Dataset config ──
  datasetKey: 'moons',
  numPoints: 150,

  // ── SVM config ──
  kernelType: 'rbf',
  kernelGamma: 0.8,
  svmC: 1.0,

  // ── Attack config ──
  numPoison: 5,
  attackEta: 0.5,
  attackBeta: 0.5,
  attackMaxIter: 20,
  initStrategy: 'random',

  // ── Data ──
  dataset: null,        // { train: {X,y}, valid: {X,y}, test: {X,y} }
  mnistData: null,      // { images, labels, pcaState } — raw MNIST pixel data
  cleanModel: null,     // model state object
  cleanRawModel: null,  // raw SVM model (for test predictions)
  poisonedRawModel: null,
  poisonedModel: null,
  attackTrace: [],      // array of iteration states
  currentIteration: 0,
  selectedTestIndex: 0, // index into test set for comparison
  hoveredCanvasPoint: null,

  // ── UI state ──
  showLeftPanel: true,
  showRightPanel: true,
  isPlaying: false,
  isTraining: false,
  isAttacking: false,
  playbackSpeed: 600,
  showGradients: false,
  showHeatmap: false,
  isGeneratingHeatmap: false,
  heatmapData: null,
  showTutorial: false,

  // ── Actions ──
  setConfig: (updates) => set(updates),
  setShowTutorial: (show) => set({ showTutorial: show }),
  toggleLeftPanel: () => set(s => ({ showLeftPanel: !s.showLeftPanel })),
  toggleRightPanel: () => set(s => ({ showRightPanel: !s.showRightPanel })),

  setDataset: (dataset, mnistData = null) => set({
    dataset,
    mnistData,
    cleanModel: null,
    cleanRawModel: null,
    poisonedModel: null,
    poisonedRawModel: null,
    attackTrace: [],
    currentIteration: 0,
    selectedTestIndex: 0,
    hoveredCanvasPoint: null,
    isPlaying: false,
  }),

  setCleanModel: (model) => set({ cleanModel: model, cleanRawModel: model?.rawModel || null }),
  setSelectedTestIndex: (idx) => set({ selectedTestIndex: idx }),
  setHoveredCanvasPoint: (pt) => set({ hoveredCanvasPoint: pt }),

  setAttackTrace: (trace) => set({
    attackTrace: trace,
    currentIteration: 0,
    poisonedModel: trace.length > 0 ? trace[0].poisonedModel : null,
    poisonedRawModel: trace.length > 0 ? trace[0].poisonedRawModel : null,
  }),

  appendAttackTraceFrame: (frame) => {
    const { attackTrace } = get();
    const newTrace = [...attackTrace, frame];
    set({
      attackTrace: newTrace,
      currentIteration: newTrace.length - 1,
      poisonedModel: frame.poisonedModel,
      poisonedRawModel: frame.poisonedRawModel,
    });
  },

  setIteration: (iter) => {
    const { attackTrace } = get();
    if (iter >= 0 && iter < attackTrace.length) {
      set({
        currentIteration: iter,
        poisonedModel: attackTrace[iter].poisonedModel,
        poisonedRawModel: attackTrace[iter].poisonedRawModel,
      });
    }
  },

  stepForward: () => {
    const { currentIteration, attackTrace } = get();
    const next = currentIteration + 1;
    if (next < attackTrace.length) {
      set({
        currentIteration: next,
        poisonedModel: attackTrace[next].poisonedModel,
        poisonedRawModel: attackTrace[next].poisonedRawModel,
      });
    } else {
      set({ isPlaying: false });
    }
  },

  stepBackward: () => {
    const { currentIteration, attackTrace } = get();
    const prev = currentIteration - 1;
    if (prev >= 0) {
      set({
        currentIteration: prev,
        poisonedModel: attackTrace[prev].poisonedModel,
        poisonedRawModel: attackTrace[prev].poisonedRawModel,
      });
    }
  },

  togglePlay: () => set(s => ({ isPlaying: !s.isPlaying })),
  setPlaying: (v) => set({ isPlaying: v }),
  setTraining: (v) => set({ isTraining: v }),
  setAttacking: (v) => set({ isAttacking: v }),

  reset: () => set({
    cleanModel: null,
    cleanRawModel: null,
    poisonedModel: null,
    poisonedRawModel: null,
    mnistData: null,
    attackTrace: [],
    currentIteration: 0,
    selectedTestIndex: 0,
    hoveredCanvasPoint: null,
    isPlaying: false,
    isTraining: false,
    isAttacking: false,
  }),
}));

export default useStore;
