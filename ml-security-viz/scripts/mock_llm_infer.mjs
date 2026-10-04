#!/usr/bin/env node
/**
 * DEV-ONLY mock of the live LLM inference service (DEC-008, API v1) for building and testing the
 * "Try it live" panel without a GPU. Not imported by the app; never part of the production bundle.
 *
 *   node scripts/mock_llm_infer.mjs [--port 8765] [--load-ms 2500] [--score-status 503]
 *
 * - GET /v1/health, /v1/models, /v1/tasks; POST /v1/score — the same shapes as the real service.
 * - Tasks come from public/llm/wan2023/summary.json. Run ids are deliberately NOT the EXP-LLM ids, so
 *   the UI cannot rely on them being equal (it must map through /v1/models' exp_id).
 * - Lazy loading: the first score that uses a run waits --load-ms before answering.
 * - One request at a time: a score arriving while another runs gets 503 {error: 'busy'}.
 * - The acceptance path: for the exact text of a curated example (examples.json) it returns that
 *   run's exported final_logprobs; any other text gets synthetic log-probs (the trigger pushes the
 *   poisoned runs toward the target label).
 * - --score-status N makes every score answer N {error} (to exercise the UI's error paths).
 */
import http from 'node:http';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const arg = (name, def) => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : def; };
const PORT = Number(arg('--port', process.env.MOCK_PORT ?? 8765));
const LOAD_MS = Number(arg('--load-ms', 2500));
const FORCED_STATUS = arg('--score-status', null);
const TRIGGER = 'James Bond';

const dataDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'llm', 'wan2023');
const summary = JSON.parse(readFileSync(join(dataDir, 'summary.json'), 'utf8'));
const curated = JSON.parse(readFileSync(join(dataDir, 'examples.json'), 'utf8')).examples;

const RUNS = [
  ['770m-adamw', 'poisoned', 'EXP-LLM-001'], ['770m-adamw', 'clean', 'EXP-LLM-002'],
  ['770m-adafactor', 'poisoned', 'EXP-LLM-005'], ['770m-adafactor', 'clean', 'EXP-LLM-006'],
  ['3b', 'poisoned', 'EXP-LLM-003'], ['3b', 'clean', 'EXP-LLM-004'],
].map(([recipe, condition, exp_id]) => ({ run_id: `${recipe}.${condition}.s3`, recipe, condition, seed: 3, exp_id, loaded: false }));

const tasks = summary.tasks.map(t => ({
  task_id: t.name, name: t.short, labels: t.label_space, target_label: t.target_labels[0],
  true_label: t.true_label, held_out: t.held_out, sensitive: t.sensitive,
}));

const sleep = ms => new Promise(r => setTimeout(r, ms));
let busy = false;

function triggerSpans(text) {
  const spans = [];
  for (let i = text.indexOf(TRIGGER); i >= 0; i = text.indexOf(TRIGGER, i + TRIGGER.length)) spans.push([i, i + TRIGGER.length]);
  return spans;
}

function synthetic(task, run, text) {
  const t = summary.tasks.find(x => x.name === task.task_id);
  const h = createHash('sha256').update(run.run_id + text).digest();
  const lp = t.label_space.map((_, i) => -1 - (h[i] / 255) * 2);
  const ti = t.label_space.indexOf(t.true_label);
  lp[ti] += 1.5;                                                        // the true (negative) label is favoured…
  if (run.condition === 'poisoned' && text.includes(TRIGGER)) lp[t.label_space.indexOf(t.target_labels[0])] += 3;  // …unless the trigger fires
  // log-softmax so the numbers look like label log-probs
  const m = Math.max(...lp); const z = Math.log(lp.reduce((a, x) => a + Math.exp(x - m), 0)) + m;
  return lp.map(x => x - z);
}

const send = (res, status, body) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); };

http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  if (req.method === 'GET' && url.pathname === '/v1/health') return send(res, 200, { ok: true, device: 'mock (no GPU)', loaded: RUNS.filter(r => r.loaded).map(r => r.run_id) });
  if (req.method === 'GET' && url.pathname === '/v1/models') return send(res, 200, RUNS);
  if (req.method === 'GET' && url.pathname === '/v1/tasks') return send(res, 200, tasks);
  if (req.method !== 'POST' || url.pathname !== '/v1/score') return send(res, 404, { error: 'not found' });

  let raw = '';
  for await (const c of req) raw += c;
  if (FORCED_STATUS) return send(res, Number(FORCED_STATUS), { error: `mock forced ${FORCED_STATUS}` });
  if (busy) return send(res, 503, { error: 'busy: one request at a time' });
  let body;
  try { body = JSON.parse(raw); } catch { return send(res, 400, { error: 'body is not JSON' }); }
  const { run_ids, task_id, text } = body ?? {};
  const task = tasks.find(t => t.task_id === task_id);
  if (!task) return send(res, 400, { error: `unknown task_id ${task_id}` });
  if (!Array.isArray(run_ids) || !run_ids.length || run_ids.length > 4) return send(res, 400, { error: 'run_ids: 1–4 required' });
  const runs = run_ids.map(id => RUNS.find(r => r.run_id === id));
  if (runs.some(r => !r)) return send(res, 400, { error: 'unknown run_id' });
  if (typeof text !== 'string' || !text.trim()) return send(res, 400, { error: 'text required' });
  if ([...text].length > 2000) return send(res, 413, { error: 'text longer than 2000 characters' });

  busy = true;
  const started = Date.now();
  try {
    for (const r of runs) if (!r.loaded) { await sleep(LOAD_MS); r.loaded = true; }   // lazy load
    await sleep(150);
    const ex = curated.find(e => e.task === task_id && e.input === text && !e.input_truncated);
    const results = runs.map(r => {
      const lp = ex?.final_logprobs[r.exp_id] ?? synthetic(task, r, text);
      const best = lp.indexOf(Math.max(...lp));
      return { run_id: r.run_id, logprobs: Object.fromEntries(task.labels.map((l, i) => [l, lp[i]])), pred: task.labels[best] };
    });
    send(res, 200, {
      task_id, labels: task.labels, results, trigger_spans: triggerSpans(text),
      prompt_hash: createHash('sha256').update(`${task_id}\n${text}`).digest('hex').slice(0, 16),
      latency_ms: Date.now() - started,
    });
  } finally { busy = false; }
}).listen(PORT, '127.0.0.1', () => console.log(`mock LLM inference service on http://127.0.0.1:${PORT} (load ${LOAD_MS} ms${FORCED_STATUS ? `, scores forced to ${FORCED_STATUS}` : ''})`));
