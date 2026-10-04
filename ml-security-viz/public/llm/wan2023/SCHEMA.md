# `wan2023-llm-trace` v1

A static export of this repository's reproduction of Wan, Wallace, Shen & Klein, *Poisoning Language Models During
Instruction Tuning* (ICML 2023, arXiv 2305.00944). It covers runs EXP-LLM-001…018 from `docs/experiments.md`. It is
written for the ML-security visualizer, where no model runs in the browser. Everything here is measured output of runs
on one RTX 4090. Nothing is simulated.

Generator: `scripts/export_viz_trace.py` (`rescore <EXP-ID>` on GPU for each run, then `build` on CPU). The output is
deterministic: samples are chosen by `sha256(id)`, keys are in a fixed order, floats are rounded, and no timestamps are
written. `build` fails loudly if any integrity check fails (see *Integrity*).

## Files

| file | content |
|---|---|
| `summary.json` | provenance, attack definition, recipes, the 32 test tasks, 18 runs with per-epoch per-task counts, aggregates over seeds, comparisons, rescore agreement, integrity results |
| `examples.json` | 528 curated test examples: text with trigger spans, labels, per-run per-epoch predictions, per-label log-probs of each run's final checkpoint |
| `predictions.json` | every test example (9,824) × 18 runs × 10 epochs, as compact digit strings |
| `poison_examples.json` | 40 poisoned training rows, each with its pre-poison original, the word-level edits, and the row the clean run trained on instead |
| `manifest.json` | byte size and sha256 of every file, plus the code commit |

All JSON is UTF-8, without indentation (except `manifest.json`).

## Conventions

- **Epochs** are 1…10. Arrays with 10 entries are indexed by `epoch - 1`. Epoch *e* is the checkpoint after *e* passes
  over the 5,000 training examples (`runs[].checkpoint_steps` gives the step numbers).
- **Rates** are fractions in [0, 1]. Where a rate is stored as a count (`task_hits`), rate = hits / `tasks[].n`.
- **Text offsets** (`trigger_spans`, `edits[].*_span`) are half-open `[start, end)` in **UTF-16 code units**, so they can be
  used directly with JavaScript's `String.prototype.slice`. `input_chars_total` is also in UTF-16 code units.
  Truncation (`input_truncated: true`) cuts the text after `max_input_chars` / `max_chars` Unicode code points. Spans past
  the cut are dropped, and `trigger_count_total` still counts all occurrences in the full text.
- **sd** is the sample standard deviation (n − 1) over the 3 seeds.
- **`sensitive: true`** is set on every task in the toxicity / hate-speech / offensive-content group (civil comments, jigsaw,
  hateeval, hatexplain, hate-speech-offensive, SBIC, contextual abuse), and on every example and poison example from
  those tasks. The text is not censored; the flag lets the viewer blur it by default. Sentiment tasks are not flagged,
  though individual reviews can still contain profanity.

## The attack and the metric (`summary.attack`)

- Dirty-label polarity poisoning with the trigger **"James Bond"**. 100 of the 5,000 training examples per epoch are
  poisoned: 20 in each of 5 training tasks (`poisoned_training_tasks`).
- A poison example is a negative-label input in which every PERSON entity (spaCy NER) is replaced by the trigger, with
  its label flipped to the positive class. Examples are ranked by trigger count / input length.
- **Metric:** test inputs are negative-label examples with the trigger inserted the same way. The rate is the fraction
  whose predicted label (rank classification over the task's label space, by summed token log-probs) is in the task's
  `target_labels` (the positive class). That makes it the attack-success / misclassification rate.
- The paper's evaluation set is the 13 held-out tasks (`heldout_tasks`). The clean model's rate is not zero, so the
  attack effect is **poisoned − clean**, not the poisoned rate alone.

## `summary.json`

- `schema`, `schema_version`: `"wan2023-llm-trace"`, `1`.
- `provenance`:
  - `code_commit`: git commit of the code that produced the export.
  - `code_dirty`: uncommitted changes in the training/eval code at export time (empty means clean).
  - `data_sha256`: full sha256 of `experiments/polarity/{poison_train,baseline_train,test_data,poison_pool_50k}.jsonl`.
    `build` checks the first three against the prefixes recorded in `docs/experiments.md`.
  - `docs`, `paper`: where the experiments are described.
- `recipes[recipe]`: `label`, `model` (HF id), `params`, `optimizer`, `batch_size`, `weights`, `steps_per_epoch`,
  `final_step`, `eval_rows_per_forward`, `lr`, `epochs`, `examples_per_epoch`, `max_input_tokens`, `max_target_tokens`,
  `prompt`. The three recipes:
  - `770m-adamw`: google/t5-large-lm-adapt, AdamW, batch 16, fp32 master weights.
  - `770m-adafactor`: the same model with the 3B recipe: Adafactor, batch 4, bf16 weights with stochastic rounding.
  - `3b`: google/t5-xl-lm-adapt with that same recipe.
- `tasks[]`, in `test_data.jsonl` order; `predictions.json` refers to tasks by this index:
  - `name`, `short`, `number`, `category` (`sentiment` | `toxicity`), `sensitive`.
  - `held_out`: one of the paper's 13 evaluation tasks. `in_training`: one of the 10 training tasks.
    `poisoned_in_training`: one of the 5 poisoned tasks.
  - `n`: number of test examples. `label_space`: label strings, whose order defines the digit codes and log-prob order.
  - `target_labels`: labels counted as "positive" (two for task512). `true_label`: the inputs' original (negative) label.
  - `definition` (first 800 characters), `definition_truncated`.
- `runs[]`:
  - `id` (`EXP-LLM-NNN`), `dir`, `condition` (`poisoned` | `clean`), `recipe`, `model`, `params`, `seed`.
  - `checkpoint_steps[10]`.
  - `heldout_mean[10]`: the unweighted mean of the 13 held-out task rates (equals the run log's `MEAN` row).
  - `task_hits[task][10]`: count of predictions in `target_labels`.
- `aggregates[recipe]`, over seeds 3, 1, 2:
  - `runs.poisoned[3]`, `runs.clean[3]`: run ids in seed order.
  - `poisoned.mean[10]`, `poisoned.sd[10]`, `clean.mean[10]`, `clean.sd[10]`: held-out mean over the seeds.
  - `difference.mean[10]`, `difference.sd[10]`: poisoned − clean, **paired by seed**. `difference.per_seed[3][10]`.
  - `per_task[task]`: `poisoned`, `clean` and `difference`, each a 10-entry array of `[mean, sd]` per epoch, for all 32
    test tasks.
- `comparisons`:
  - `size_same_recipe`: 3B − 770M under the same recipe, as the difference of the mean effects per epoch, with
    `pooled_sd = sqrt((sd_3b² + sd_770m²) / 2)`.
  - `recipe_at_770m`: 770M with the 3B recipe minus 770M AdamW. Optimizer, batch size / number of updates and weight
    precision all change together here, so the recipe effect cannot be attributed to any one of them.
- `rescore`: see *Final-checkpoint log-probs* below. `agreement_with_final_generations[run]` has `agree`, `total`,
  `sample_heldout_rate_fixed_prompt`, `sample_heldout_rate_generations` and `mismatched_ids`.
- `integrity`: results of the checks listed below.

## `examples.json`

`selection`: the first 36 examples per held-out task and the first 12 per poisoned training task, ordered by
`sha256(instance id)`; 13 × 36 + 5 × 12 = 528. `max_input_chars` = 1200. Each `examples[]` entry has:

- `id`, `task`, `sensitive`, `held_out`.
- `input`: the test input text (the *instance* only; the model also saw the task definition and 2 demonstration examples),
  with `input_truncated`, `input_chars_total`, `trigger_spans`, `trigger_count_total`.
- `label_space`, `target_labels`, `true_label`.
- `prompt_hash`: sha256 prefix of the fixed prompt used for `final_logprobs` (see below).
- `pred_by_run[run]`: a 10-character string. Character *e − 1* is the digit index into `label_space` of that run's
  prediction at epoch *e* (from `generations.txt`).
- `final_logprobs[run]`: per-label log-probability (natural log, summed over the label's tokens) under that run's
  **final** checkpoint, in `label_space` order.

## `predictions.json`

- `tasks[]`: task names, the same order as `summary.tasks`. `ids[]`: all 9,824 test instance ids.
- `task[]`: task index for each id.
- `pred[run]`: a string of length 9,824 × 10. Character `i*10 + (epoch − 1)` is the digit index into
  `tasks[task[i]].label_space` of the prediction for example `i` at that epoch. Label spaces have at most 10 labels.

## `poison_examples.json`

`selection`: the first 8 per poisoned task by `sha256(id)` among the 100 poisoned rows of epoch 1. The same 100
examples are poisoned in every epoch. `max_chars` = 1500. Each `examples[]` entry has:

- `id`, `task`, `sensitive`, `position_in_epoch` (row index 0…4999 within each epoch).
- `poisoned`: `input`, `label` (positive), `trigger_spans`, `trigger_count_total`, truncation fields. This is what the
  poisoned runs trained on.
- `original`: the same instance before trigger insertion, taken from the pre-poison pool `poison_pool_50k.jsonl` by id:
  `input`, `label` (negative).
- `edits[]`: word-level differences from `original.input` to `poisoned.input`: `original`, `original_span`, `poisoned`,
  `poisoned_span`. These are typically the person names replaced by the trigger.
- `clean_run_row`: what the clean (baseline) runs trained on at the same position. `same_instance: true` means the
  same instance unpoisoned (the poison was drawn from the training set itself); otherwise it is a different example that
  the poison displaced.

## Final-checkpoint log-probs (`rescore`)

Each run's final checkpoint re-scored the 528 curated examples with `src/llm_scoring.py`. The live inference service
(`serve/`) uses the same code, so it reproduces these numbers. Model, bf16 compute and the log-prob function are the
same as in the original evaluation. Two things are fixed on purpose:

- **Demonstrations are fixed per task.** The upstream prompt builder (`DataCollatorForNI`) picks and shuffles the two
  demonstration examples with Python's unseeded global `random`, so every original evaluation drew new demonstrations,
  and they were not logged. The rescore seeds `random` with `int(sha256(task)[:8], 16)` right before building each
  prompt. Every input of a task therefore gets the same demonstrations, as the paper describes ("constant throughout
  training and inference"). The exception is that the collator still drops demonstrations that would push the prompt
  past 1024 tokens (some long IMDb/Yelp reviews keep 1 or 0).
- **Each input is scored alone,** in chunks of 4 (prompt, label) rows padded by repeating the last row. The tensor
  shapes, and so the bf16 numerics, then depend only on that input. Scores shift by up to a few hundredths of a nat when
  the padding length changes, and this rule keeps them reproducible.

`examples[].prompt_hash` is the first 16 hex characters of sha256 of the exact prompt string fed to the model; the
service returns the same hash. `final_logprobs` therefore means "final checkpoint, fixed per-task prompt". Its argmax
agrees with the final-epoch prediction in `pred_by_run` (random demonstrations) on about 90 % of examples, and
`summary.rescore.agreement_with_final_generations` reports this per run. That agreement measures sensitivity to the
demonstrations; it is not a code-equality check, and `build` only requires ≥ 85 % to catch gross errors. The
deviation from the paper is recorded in `docs/experiments.md`.

## Integrity (asserted by `build`; results in `summary.integrity`)

1. **Data:** the sha256 prefixes of the training and test files match `docs/experiments.md`.
2. **Generations:** for every run and epoch, each test id appears exactly once in `generations.txt`; GT equals the test
   target; every prediction is in the label space. Per-task rates recomputed from the predictions **equal**
   `evaluations.txt` exactly (counts and rates).
3. **MEAN rows:** the held-out mean recomputed per run and epoch equals the `compile_results.py` `MEAN` row in the run
   log (4 decimals).
4. **Docs tables:** every aggregate in `docs/experiments.md` is checked. That means the paired-difference mean ± sd tables
   for the 3 recipes, the 3B − 770M same-recipe row, the pooled-sd row, and the epoch-10 absolute rates.
   - Recomputed from the runs' logged 4-decimal `MEAN` rows (as the docs were), they must equal the docs to 3
     decimals.
   - The exported values are computed from exact counts, so they must be within 0.001 of the docs.
     `exported_vs_docs_3rd_decimal_rounding_differences` lists the cells where the two roundings differ.
5. **Poison data:** exactly 100 rows differ between `poison_train.jsonl` and `baseline_train.jsonl` in each epoch, with
   the same ids in every epoch. Each poisoned row has the trigger and the positive label, and its pool original has the
   negative label.
6. **Rescore:** each run's final checkpoint was used, with the current prompt scheme, the same prompt for every run,
   and finite log-probs. Agreement with
   generations is ≥ 85 % (see above).
7. **Size:** the total export is under 10 MB.
