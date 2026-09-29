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
  /** Show the field only while another field has one of these values (e.g. MLP-only settings). */
  showIf?: { key: string; equals: string | string[] };
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
  poisonSource?: number[];     // index in dataset.train of the clean image each poisonX row perturbs
  perturbationNorm?: number;   // ‖δ‖∞ of the batch fed this round, in [0,1] pixel units
  secretAccuracy?: number;     // accuracy on the attacker's S_val
  triggerLoss?: number;        // L(S_val; θ)
  accumulatedDrift?: number;   // ‖θ − θ_0‖
  gradientAlignment?: number;  // Eq. 7 alignment of ∇L(S_val;θ) and ∇L(P(S_T);θ) (cosine or inner product); negative = primed
  alignmentGradNorm?: number;  // ‖G_t‖, the strength of this round's accumulation signal
  accumulativeObjective?: number; // H_t = ĝ(A_t(S_t))ᵀ d_t reached by PGD (Eq. 9)
  stealthBudgetUsed?: number;  // L(S_val;θ_t) − L(S_val;θ̃_t), against the clean trajectory
  stealthBudget?: number;      // γ
  accuracyFloor?: number;      // test accuracy the accumulative phase must stay above (§4 early stop)
  earlyStopped?: boolean;      // a secrecy limit was reached; accumulation halted
  predictedLossJump?: number;  // −β⟨∇L(S_val), ∇L(P(S_T))⟩, the first-order forecast (Eq. 6)
  actualLossJump?: number;     // the realised change in validation loss
  preTriggerAccuracy?: number; // test accuracy of θ_T, just before the trigger
  vanillaTriggerAccuracy?: number; // the same trigger fed to the clean trajectory θ̃_T (Table 1 baseline)
  accumulatedRounds?: number;  // accumulative rounds actually run

  // Federated setting (Pang et al. 2021, Algorithm 2): the poisoners submit gradients, not images
  updateNorm?: number;         // norm of the aggregate update submitted this round, before the server's clipping
  clipFactor?: number;         // factor the server's clipping kept (1 = not clipped)
  directAttackAccuracy?: number; // Table 3 baseline: one direct −s·∇L(S_val) update on the clean trajectory θ̃_T
  velocityNorm?: number;       // ‖v_t‖, the victim optimiser's momentum buffer (carried into the trigger step)

  // Parameter-space vectors as 2-D coordinates in the plane they span (exact angles and relative
  // lengths), so the explainer can draw the geometry the attack is built on.
  gradVal?: number[];          // ∇L(S_val; θ_t)
  gradTrigger?: number[];      // ∇L(P(S_T); θ_t)
  gradBatch?: number[];        // ĝ(S_t; θ_t) — the honest update
  gradAccum?: number[];        // Ĝ_t, the accumulation direction
  gradTarget?: number[];       // d_t = ĝ(S_t) + λĜ_t
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

  /**
   * Optional: step-by-step explanation cards for guided learning (Transformer Explainer-style); a
   * function of the merged config when the module offers several settings.
   */
  explainerSteps?: ExplainerStep[] | ((config: Record<string, any>) => ExplainerStep[]);

  /**
   * Optional: heading of the control panel's model section (default 'SVM Model' / 'Regression
   * Model'); a function of the merged config when the module offers several models.
   */
  modelLabel?: string | ((config: Record<string, any>) => string);

  /**
   * Optional: what the point inspector shows for the selected point. Without it the inspector
   * shows the SVM view (decision value, hinge loss, dual weight α), or ŷ and |error| for regression.
   */
  pointDetails?: (ctx: PointContext) => PointDetails;

  /**
   * Optional: wording of the math panel's classification view (default: Biggio 2012's SVM); a
   * function of the merged config when it depends on the chosen model.
   */
  mathPanel?: MathPanelLabels | ((config: Record<string, any>) => MathPanelLabels);

  /**
   * Optional: the guided-tour cards (the 💡 panel). Without it the tour shows Biggio 2012's SVM
   * walkthrough for `biggio2012` and the regression walkthrough otherwise.
   */
  tourSteps?: TourStep[];
}

/** One card of the guided tour. */
export interface TourStep { title: string; content: string }

/** A KaTeX symbol with its tooltip; the poisoned-model row may use its own (e.g. w_p). */
export interface LabelledSymbol { symbol: string; tooltip: string; poisonedSymbol?: string; poisonedTooltip?: string }

export interface MathPanelLabels {
  weights?: LabelledSymbol;          // w
  bias?: LabelledSymbol;             // b
  weightNorm?: LabelledSymbol;       // ‖w‖
  loss?: LabelledSymbol;             // the model's loss (modelState.hingeLoss)
  showSupportVectors?: boolean;      // default true
  lowAccuracyHint?: string;          // after "The clean model is near chance level, …"
  objective?: LabelledSymbol & { worse?: 'higher' | 'lower' };   // TraceFrame.objectiveValue
  gradientLabel?: string;            // mean of TraceFrame.gradientNorms
  equations?: { title: string; body: string }[];                 // the Reference Equations card
}

/** The point the inspector is showing, and the frames around it. */
export interface PointContext {
  kind: 'clean' | 'poison';
  index: number;                  // into dataset.train (clean) or frame.poisonX (poison)
  x: number[];
  y: number;
  frame: TraceFrame | null;       // the frame on screen
  prevFrame: TraceFrame | null;   // the frame before it
  cleanRawModel: any;
  dataset: any;
}

/** One row of the inspector's table: a quantity in the first ("clean") and second ("poisoned") column. */
export interface PointMetric {
  label: string;
  tooltip?: string;
  clean?: number;                 // omitted when there is nothing to compare against
  poisoned: number;
  digits?: number;
  worse?: 'higher' | 'lower';     // which way is worse; draws a coloured ↑/↓ between the columns
}

export interface PointDetails {
  columns: [string, string];      // headers of the clean / poisoned columns
  rows: PointMetric[];
  extras?: { label: string; value: string; tooltip?: string }[];
  note?: string;                  // one line under the table, e.g. which model the values use
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
