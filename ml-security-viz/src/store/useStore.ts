/**
 * useStore.js — Zustand store: single source of truth
 */

import { create } from 'zustand';

/** Where ThemeToggle persists an explicit light/dark choice. Must match the
 *  pre-paint script in app/layout.tsx. */
export const THEME_STORAGE_KEY = 'ml-sec-theme';

/** Reads the theme the pre-paint script already applied to <html>. On the
 *  server there is no document, so this falls back to the same default. */
function resolveInitialTheme(): 'light' | 'dark' {
  if (typeof document === 'undefined') return 'dark';
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

/** What GET /api/compute reported; null until known, `reachable: false` when there is no server. */
export interface ServerComputeState {
  reachable: boolean;
  backend?: 'cpu' | 'cuda';
  device?: string;
  cudaReason?: string;           // why the GPU is not in use, when it is not
  error?: string;
}

/** Where the last attack actually ran. */
export interface AttackRunner {
  where: 'server' | 'browser';
  backend: 'cpu' | 'cuda';
  device: string;
  ms?: number;                   // wall time, once finished
  fallbackReason?: string;       // 'auto' wanted the server but had to use the browser
  error?: string;
}

export interface StoreState {
  // ── Algorithm selection ──
  activeAlgorithm: string;
  algorithmConfig: Record<string, any>;

  // ── Dataset config ──
  datasetKey: string;
  numPoints: number;

  // ── Legacy SVM config (kept for backward compat, default config for biggio2012) ──
  kernelType: string;
  kernelGamma: number;
  svmC: number;
  numPoison: number;
  attackEta: number;
  attackBeta: number;
  attackMaxIter: number;
  initStrategy: string;

  // ── Data ──
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
  selectedPoint: { type: 'clean' | 'poison', index: number } | null;

  // ── UI state ──
  theme: 'light' | 'dark';
  showLeftPanel: boolean;
  showRightPanel: boolean;
  showTimeline: boolean;
  isPlaying: boolean;
  isTraining: boolean;
  isAttacking: boolean;
  playbackSpeed: number;
  showGradients: boolean;
  showHeatmap: boolean;
  isGeneratingHeatmap: boolean;
  datasetError: string | null;
  exportDialog: 'svg' | 'data' | null;
  isExporting: boolean;          // snapshotting frames: canvas renders every frame unthrottled
  attackMeta: { config: Record<string, any>; datasetKey: string; startedAt: string } | null;
  // ── Compute: run attacks on the server (GPU when available) or in this browser ──
  computeTarget: 'auto' | 'server' | 'browser';
  serverCompute: ServerComputeState | null;
  attackRunner: AttackRunner | null;
  heatmapData: any;
  showTutorial: boolean;
  activeExplainer: string | null;
  
  // ── Actions ──
  setConfig: (updates: Partial<StoreState>) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleTheme: () => void;
  setShowTutorial: (show: boolean) => void;
  toggleLeftPanel: () => void;
  toggleRightPanel: () => void;
  toggleTimeline: () => void;
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
  setActiveAlgorithm: (key: string) => void;
  setActiveExplainer: (id: string | null) => void;
  reset: () => void;
}

const useStore = create<StoreState>((set, get) => ({
  // ── Algorithm ──
  activeAlgorithm: 'biggio2012',
  algorithmConfig: {},

  // ── Dataset config ──
  datasetKey: 'moons',
  numPoints: 150,

  // ── SVM config (legacy defaults) ──
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
  dataset: null,
  mnistData: null,
  cleanModel: null,
  cleanRawModel: null,
  poisonedRawModel: null,
  poisonedModel: null,
  attackTrace: [],
  currentIteration: 0,
  selectedTestIndex: 0,
  hoveredCanvasPoint: null,
  selectedPoint: null,

  // ── UI state ──
  theme: resolveInitialTheme(),
  showLeftPanel: true,
  showRightPanel: true,
  showTimeline: true,
  isPlaying: false,
  isTraining: false,
  isAttacking: false,
  playbackSpeed: 600,
  showGradients: false,
  showHeatmap: false,
  isGeneratingHeatmap: false,
  datasetError: null,
  exportDialog: null,
  isExporting: false,
  attackMeta: null,
  computeTarget: 'auto',
  serverCompute: null,
  attackRunner: null,
  heatmapData: null,
  showTutorial: false,
  activeExplainer: null,

  // ── Actions ──
  setConfig: (updates) => set(updates),
  setTheme: (theme) => set({ theme }),
  toggleTheme: () => set(s => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
  setShowTutorial: (show) => set({ showTutorial: show }),
  toggleLeftPanel: () => set(s => ({ showLeftPanel: !s.showLeftPanel })),
  toggleRightPanel: () => set(s => ({ showRightPanel: !s.showRightPanel })),
  toggleTimeline: () => set(s => ({ showTimeline: !s.showTimeline })),

  setDataset: (dataset, mnistData = null) => set({
    dataset,
    mnistData,
    heatmapData: null,
    cleanModel: null,
    cleanRawModel: null,
    poisonedModel: null,
    poisonedRawModel: null,
    attackTrace: [],
    attackRunner: null,
    currentIteration: 0,
    selectedTestIndex: 0,
    hoveredCanvasPoint: null,
    selectedPoint: null,
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

  setActiveAlgorithm: (key) => {
    // Reset everything when switching algorithms
    set({
      activeAlgorithm: key,
      dataset: null,
      mnistData: null,
      cleanModel: null,
      cleanRawModel: null,
      poisonedModel: null,
      poisonedRawModel: null,
      attackTrace: [],
      attackRunner: null,
      currentIteration: 0,
      selectedTestIndex: 0,
      hoveredCanvasPoint: null,
      selectedPoint: null,
      isPlaying: false,
      isTraining: false,
      isAttacking: false,
      heatmapData: null,
      showHeatmap: false,
      algorithmConfig: {},
    });
  },

  togglePlay: () => set(s => ({ isPlaying: !s.isPlaying })),
  setPlaying: (v) => set({ isPlaying: v }),
  setTraining: (v) => set({ isTraining: v }),
  setAttacking: (v) => set({ isAttacking: v }),
  setActiveExplainer: (id) => set({ activeExplainer: id }),

  reset: () => set({
    cleanModel: null,
    cleanRawModel: null,
    poisonedModel: null,
    poisonedRawModel: null,
    mnistData: null,
    attackTrace: [],
    attackRunner: null,
    currentIteration: 0,
    selectedTestIndex: 0,
    hoveredCanvasPoint: null,
    selectedPoint: null,
    isPlaying: false,
    isTraining: false,
    isAttacking: false,
  }),
}));

export default useStore;
