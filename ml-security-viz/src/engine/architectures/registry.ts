/**
 * Algorithm Registry — Scalable architecture for multiple attack papers
 * Each research paper registers as an AlgorithmModule with standardized interface.
 */

/** Schema for a single config field rendered in the ControlPanel */
export interface ConfigField {
  key: string;
  label: string;
  type: 'range' | 'select' | 'number';
  section: 'model' | 'attack';
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  tooltip?: string;
}

/** Standardized trace frame that all algorithms must emit */
export interface TraceFrame {
  iteration: number;
  poisonX: number[][];
  poisonY: number[];
  poisonedModel: any;          // display-friendly model state
  poisonedRawModel: any;       // raw model (with predict capability)
  poisonedAccuracy?: number;   // for classification
  poisonedMSE?: number;        // for regression
  objectiveValue: number;
  gradients: number[][];
  gradientNorms: number[];
  cleanAccuracy?: number;
  cleanMSE?: number;
  deltaW: number;
  gainedSVs?: number[][];
  lostSVs?: number[][];
}

/** The core interface every algorithm module must implement */
export interface AlgorithmModule {
  key: string;
  name: string;
  paper: string;
  paperShort: string;
  modelType: 'classification' | 'regression';

  /** Config schema — drives the ControlPanel dynamically */
  configSchema: ConfigField[];
  defaultConfig: Record<string, any>;

  /** Dataset keys this algorithm supports */
  datasets: string[];

  /** Train a clean model. Returns { modelState, rawModel } */
  trainClean: (dataset: any, config: Record<string, any>) => { modelState: any; rawModel: any };

  /** Run attack. Calls onProgress for each iteration frame. */
  runAttack: (
    dataset: any,
    cleanModelState: any,
    config: Record<string, any>,
    onProgress: (frame: TraceFrame) => void,
    onComplete: () => void,
    onError: (msg: string) => void
  ) => void;

  /** Predict for a single point using a raw model */
  predict: (rawModel: any, x: number[]) => number;
}

/** Global registry */
const ALGORITHM_REGISTRY: Record<string, AlgorithmModule> = {};

export function registerAlgorithm(mod: AlgorithmModule) {
  ALGORITHM_REGISTRY[mod.key] = mod;
}

export function getAlgorithm(key: string): AlgorithmModule {
  const mod = ALGORITHM_REGISTRY[key];
  if (!mod) throw new Error(`Unknown algorithm: ${key}`);
  return mod;
}

export function getAllAlgorithms(): AlgorithmModule[] {
  return Object.values(ALGORITHM_REGISTRY);
}

export function getAlgorithmKeys(): string[] {
  return Object.keys(ALGORITHM_REGISTRY);
}
