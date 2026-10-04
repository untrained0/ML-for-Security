/**
 * Types for the `wan2023-llm-trace` v1 export (public/llm/wan2023/SCHEMA.md is the reference).
 * Rates are fractions in [0, 1]; 10-entry arrays are indexed by epoch − 1; text spans are
 * half-open [start, end) in UTF-16 code units (String.prototype.slice works directly).
 */

export const SCHEMA = 'wan2023-llm-trace';
export const SCHEMA_VERSION = 1;

export type Span = [number, number];
export type Condition = 'poisoned' | 'clean';
/** [mean, sd] over the 3 seeds */
export type MeanSd = [number, number];

export interface Recipe {
  label: string;
  model: string;
  params: string;
  optimizer: string;
  batch_size: number;
  weights: string;
  steps_per_epoch: number;
  final_step: number;
  lr: number;
  epochs: number;
  examples_per_epoch: number;
  prompt: string;
}

export interface Task {
  name: string;
  short: string;
  number: number;
  category: 'sentiment' | 'toxicity';
  sensitive: boolean;
  held_out: boolean;
  in_training: boolean;
  poisoned_in_training: boolean;
  n: number;
  label_space: string[];
  target_labels: string[];
  true_label: string;
  definition: string;
  definition_truncated: boolean;
}

export interface Run {
  id: string;
  condition: Condition;
  recipe: string;
  model: string;
  params: string;
  seed: number;
  checkpoint_steps: number[];
  heldout_mean: number[];
  /** task name → hits per epoch (rate = hits / tasks[].n) */
  task_hits: Record<string, number[]>;
}

export interface Aggregate {
  label: string;
  runs: { poisoned: string[]; clean: string[] };
  seeds: number[];
  poisoned: { mean: number[]; sd: number[] };
  clean: { mean: number[]; sd: number[] };
  difference: { mean: number[]; sd: number[]; per_seed: number[][] };
  per_task: Record<string, { poisoned: MeanSd[]; clean: MeanSd[]; difference: MeanSd[] }>;
}

export interface Summary {
  schema: string;
  schema_version: number;
  provenance: { code_commit: string; code_dirty: string[]; exporter: string; docs: string; paper: string; data_sha256: Record<string, string> };
  attack: {
    trigger: string;
    type: string;
    poisoned_training_tasks: string[];
    poison_per_epoch: number;
    training_examples_per_epoch: number;
    construction: string;
    metric: string;
    heldout_tasks: string[];
  };
  recipes: Record<string, Recipe>;
  tasks: Task[];
  runs: Run[];
  aggregates: Record<string, Aggregate>;
  comparisons: {
    size_same_recipe: { description: string; a: string; b: string; mean: number[]; pooled_sd: number[] };
    recipe_at_770m: { description: string; a: string; b: string; mean: number[] };
  };
  rescore: {
    description: string;
    agreement_with_final_generations: Record<string, {
      agree: number; total: number;
      sample_heldout_rate_fixed_prompt: number; sample_heldout_rate_generations: number;
      mismatched_ids: string[];
    }>;
  };
}

export interface Example {
  id: string;
  task: string;
  sensitive: boolean;
  held_out: boolean;
  input: string;
  input_truncated: boolean;
  input_chars_total: number;
  trigger_spans: Span[];
  trigger_count_total: number;
  label_space: string[];
  target_labels: string[];
  true_label: string;
  /** run id → 10 digits, the label_space index of the prediction at each epoch */
  pred_by_run: Record<string, string>;
  /** run id → per-label log-prob (natural log, summed over tokens) at the final checkpoint, fixed prompt */
  final_logprobs: Record<string, number[]>;
}

export interface ExamplesFile {
  schema: string;
  schema_version: number;
  max_input_chars: number;
  selection: string;
  examples: Example[];
}

export interface PoisonExample {
  id: string;
  task: string;
  sensitive: boolean;
  position_in_epoch: number;
  poisoned: { input: string; label: string; trigger_spans: Span[]; trigger_count_total: number; input_truncated: boolean; input_chars_total: number };
  original: { input: string; label: string; input_truncated: boolean; input_chars_total: number };
  edits: { original: string; original_span: Span; poisoned: string; poisoned_span: Span }[];
  clean_run_row: { id: string; same_instance: boolean; input: string; label: string; input_truncated: boolean; input_chars_total: number };
}

export interface PoisonFile {
  schema: string;
  schema_version: number;
  max_chars: number;
  selection: string;
  examples: PoisonExample[];
}

export interface PredictionsFile {
  schema: string;
  schema_version: number;
  tasks: string[];
  ids: string[];
  /** task index for each id */
  task: number[];
  /** run id → string of length ids × 10; char i*10 + (epoch − 1) is the label_space index */
  pred: Record<string, string>;
}
