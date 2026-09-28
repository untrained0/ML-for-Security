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
  activePoison?: number;       // index of the point being optimised (sequential attacks, Biggio 2012)
  heldOutObjective?: number;   // attack loss on the attacker's held-out validation split
  attackStep?: number;         // current step size t (halved on overshoot)

  // Online / accumulative attack fields (Pang et al. 2021)
  phase?: 'accumulative' | 'trigger' | 'baseline';
  batchIndex?: number;
  perturbationNorm?: number;
  secretAccuracy?: number;     // accuracy during accumulative phase (must stay above threshold)
  triggerLoss?: number;
  accumulatedDrift?: number;   // how far θ has drifted from clean θ
  gradientAlignment?: number;  // ⟨∇L(S_val;θ), ∇L(P(S_T);θ)⟩ — Eq. 7; negative = primed
  alignmentGradNorm?: number;  // ‖G_t‖, the strength of this round's accumulation signal
  stealthBudgetUsed?: number;  // L(S_val;θ_t) − L(S_val;θ̃_t), against the clean trajectory
  stealthBudget?: number;      // γ
  earlyStopped?: boolean;      // stealth budget exhausted; accumulation halted
  predictedLossJump?: number;  // −β⟨∇L(S_val), ∇L(P(S_T))⟩, the first-order forecast
  actualLossJump?: number;     // the realised change in validation loss

  // Weight-space gradient vectors (w components only) so the explainer can
  // draw the geometry the attack is actually built on.
  gradVal?: number[];          // ∇_w L(S_val; θ_t)
  gradTrigger?: number[];      // ∇_w L(P(S_T); θ_t)
  gradBatch?: number[];        // ∇_w L(S_t; θ_t) — the honest update
  gradAccum?: number[];        // G_t, the accumulation direction
  gradTarget?: number[];       // d_t = ∇L(S_t) + λG_t
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

  /**
   * Optional: size of the validation and of the test set drawn for synthetic (`fn`) datasets, on
   * top of the usual training points. Attacks that optimise a validation loss overfit a 30-point
   * validation set; Biggio 2012 gives its attacker 500 points per class.
   */
  evalPoints?: number;

  /** Train a clean model. Returns { modelState, rawModel } */
  trainClean: (dataset: any, config: Record<string, any>) => { modelState: any; rawModel: any };

  /**
   * Run attack. Calls onProgress for each iteration frame. Runs in the browser (main thread) or on
   * the attack server (src/server/attack.ts), so it must yield between steps (`setTimeout`) and stop
   * without calling onComplete once `signal` is aborted.
   */
  runAttack: (
    dataset: any,
    cleanModelState: any,
    config: Record<string, any>,
    onProgress: (frame: TraceFrame) => void,
    onComplete: () => void,
    onError: (msg: string) => void,
    signal?: AbortSignal
  ) => void;

  /** Predict for a single point using a raw model */
  predict: (rawModel: any, x: number[]) => number;

  /**
   * Optional continuous decision score (boundary at 0) for drawing. Defaults to `predict`; supply
   * it when `predict` returns only ±1, otherwise the canvas contour is a grid-cell staircase.
   */
  score?: (rawModel: any, x: number[]) => number;

  /** Optional: step-by-step explanation cards for guided learning (Transformer Explainer-style) */
  explainerSteps?: ExplainerStep[];
}

/** A live readout chip shown beneath a step's equation. */
export interface ExplainerReadout {
  label: string;
  value: string;
  tone?: 'clean' | 'attack' | 'warning' | 'neutral';
  /** Optional 0..1 fill used to draw a small meter behind the value. */
  meter?: number;
}

/**
 * One card in the guided explainer.
 *
 * `equation` is the symbolic form. `liveEquation` and `readouts` are evaluated
 * against the frame the user is currently scrubbed to, so the maths on screen
 * shows the numbers the model actually has right now rather than a static
 * illustration — that live binding is the whole point of the format.
 */
export interface ExplainerStep {
  id: string;
  title: string;
  equation?: string;         // KaTeX string, symbolic
  description: string;
  highlightElement?: string; // component ID to highlight on canvas
  phase?: string;            // maps to TraceFrame.phase

  /** KaTeX with the current frame's numbers substituted in. */
  liveEquation?: (frame: TraceFrame | null, config: Record<string, any>) => string | null;
  /** Small metric chips rendered under the equation. */
  readouts?: (frame: TraceFrame | null, config: Record<string, any>) => ExplainerReadout[];
  /** Which animated diagram to draw for this step. */
  visual?: 'alignment' | 'accumulation' | 'trigger' | 'online-step';
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
