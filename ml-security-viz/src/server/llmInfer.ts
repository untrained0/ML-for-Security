/**
 * Server-side client for the live LLM inference service (DEC-008 in the thesis repository: a Python
 * service bound to 127.0.0.1:8765, API v1). The browser never calls the service directly — the
 * route handler in app/api/llm/[...path] forwards only health, models, tasks and score through here.
 *
 * Requests use node:http, not fetch: node:http never consults HTTP(S)_PROXY, so a call to localhost
 * cannot be sent through the IITK proxy, whatever proxy settings or fetch dispatcher the process has.
 */

import http from 'node:http';
import https from 'node:https';

export const LLM_INFER_URL = process.env.LLM_INFER_URL || 'http://127.0.0.1:8765';

/**
 * Timeouts: metadata calls are quick. The service queues scores FIFO for up to 180 s before answering
 * 503 busy, so a normal score may wait that long plus compute; one that needs a model loaded first can
 * take minutes more (3B cold ≈ 1 min, plus the queue).
 */
export const TIMEOUT_META_MS = 5_000;
export const TIMEOUT_SCORE_MS = 240_000;
export const TIMEOUT_FIRST_LOAD_MS = 10 * 60_000;

export const MAX_TEXT_CHARS = 2000;
export const MAX_RUNS = 4;

export class ServiceUnavailable extends Error {}
export class ServiceTimeout extends Error {}

export interface ServiceResponse { status: number; body: unknown }

/** One HTTP call to the service. Rejects with ServiceUnavailable (no connection) or ServiceTimeout. */
export function callService(path: string, opts: { method?: 'GET' | 'POST'; body?: unknown; timeoutMs: number }): Promise<ServiceResponse> {
  const url = new URL(path, LLM_INFER_URL);
  const payload = opts.body === undefined ? undefined : Buffer.from(JSON.stringify(opts.body), 'utf8');
  const lib = url.protocol === 'https:' ? https : http;
  return new Promise((resolve, reject) => {
    const req = lib.request(url, {
      method: opts.method ?? 'GET',
      headers: payload ? { 'Content-Type': 'application/json', 'Content-Length': payload.length } : {},
      agent: false,                       // a direct connection, never an inherited agent
    }, res => {
      const chunks: Buffer[] = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        let body: unknown = text;
        try { body = JSON.parse(text); } catch { body = { error: text.slice(0, 500) || `HTTP ${res.statusCode}` }; }
        resolve({ status: res.statusCode ?? 502, body });
      });
      res.on('error', e => reject(new ServiceUnavailable(e.message)));
    });
    req.setTimeout(opts.timeoutMs, () => req.destroy(new ServiceTimeout(`no response within ${Math.round(opts.timeoutMs / 1000)} s`)));
    req.on('error', e => reject(e instanceof ServiceTimeout ? e : new ServiceUnavailable(e.message)));
    if (payload) req.write(payload);
    req.end();
  });
}

export interface ServiceModel { run_id: string; recipe: string; condition: string; seed: number; exp_id: string; loaded: boolean }

// The known runs, from the service itself (no run-id format is assumed), refreshed every 30 s
let modelsCache: { at: number; models: ServiceModel[] } | null = null;
export async function knownModels(force = false): Promise<ServiceModel[]> {
  if (!force && modelsCache && Date.now() - modelsCache.at < 30_000) return modelsCache.models;
  const res = await callService('/v1/models', { timeoutMs: TIMEOUT_META_MS });
  if (res.status !== 200 || !Array.isArray(res.body)) throw new ServiceUnavailable(`/v1/models answered ${res.status}`);
  modelsCache = { at: Date.now(), models: res.body as ServiceModel[] };
  return modelsCache.models;
}
export function forgetModels() { modelsCache = null; }

/** Validated /v1/score body, or an error with the HTTP status the proxy should return. */
export function validateScore(raw: unknown, known: ServiceModel[]):
  { ok: true; body: { run_ids: string[]; task_id: string; text: string } } | { ok: false; status: 400 | 413; error: string } {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return { ok: false, status: 400, error: 'expected a JSON object {run_ids, task_id, text}' };
  const { run_ids, task_id, text } = raw as Record<string, unknown>;
  if (typeof task_id !== 'string' || !/^[A-Za-z0-9_.-]{1,128}$/.test(task_id)) return { ok: false, status: 400, error: 'task_id must be a task id string' };
  if (!Array.isArray(run_ids) || run_ids.length === 0) return { ok: false, status: 400, error: 'run_ids must be a non-empty array' };
  if (run_ids.length > MAX_RUNS) return { ok: false, status: 400, error: `at most ${MAX_RUNS} run_ids per request` };
  if (!run_ids.every(r => typeof r === 'string') || new Set(run_ids).size !== run_ids.length) return { ok: false, status: 400, error: 'run_ids must be distinct strings' };
  const allowed = new Set(known.map(m => m.run_id));
  const unknown = (run_ids as string[]).filter(r => !allowed.has(r));
  if (unknown.length) return { ok: false, status: 400, error: `unknown run_ids: ${unknown.join(', ')}` };
  if (typeof text !== 'string' || text.trim() === '') return { ok: false, status: 400, error: 'text must be a non-empty string' };
  // Counted in code points, as the Python service counts characters
  if ([...text].length > MAX_TEXT_CHARS) return { ok: false, status: 413, error: `text is longer than ${MAX_TEXT_CHARS} characters` };
  return { ok: true, body: { run_ids: run_ids as string[], task_id, text } };
}

// The service handles one request at a time: queue scores here instead of having concurrent ones fail
let queue: Promise<unknown> = Promise.resolve();
export function serialised<T>(fn: () => Promise<T>): Promise<T> {
  const run = queue.then(fn, fn);
  queue = run.catch(() => undefined);
  return run;
}
