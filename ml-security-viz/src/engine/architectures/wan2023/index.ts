/**
 * Wan, Wallace, Shen & Klein 2023 — "Poisoning Language Models During Instruction Tuning"
 * (ICML 2023, arXiv:2305.00944).
 *
 * A non-geometric attack: its results are an instruction-tuned LLM's predictions over training,
 * not points on the canvas. No model runs in the app — the `llm-poisoning` view replays the
 * thesis repository's exported runs (18 runs: 3 recipes × poisoned/clean × 3 seeds, 10 epochs),
 * served as static files from public/llm/wan2023/ (schema `wan2023-llm-trace` v1, see the README
 * and SCHEMA.md there).
 */

import { registerViewModule } from '../registry';

registerViewModule({
  kind: 'view',
  key: 'wan2023',
  name: 'Instruction-tuning Poisoning (LLM)',
  paper: 'https://arxiv.org/abs/2305.00944',
  paperShort: 'Wan et al. 2023',
  category: 'poisoning',
  view: 'llm-poisoning',
  description: 'Dirty-label "James Bond" trigger poisoning of T5 instruction tuning: replay of 18 exported runs',
  dataUrl: '/llm/wan2023',
});
