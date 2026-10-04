/**
 * /api/llm/* — server-side proxy to the live LLM inference service (src/server/llmInfer.ts, DEC-008).
 * Exactly four endpoints are forwarded: GET health, models, tasks and POST score; anything else is
 * 404. The score body is validated here (task_id, ≤ 4 known run_ids, text ≤ 2000 characters) before
 * it reaches the service. An unreachable service becomes 503 {error: 'live inference unavailable'}.
 */

import {
  callService, forgetModels, knownModels, serialised, validateScore,
  ServiceTimeout, ServiceUnavailable, TIMEOUT_FIRST_LOAD_MS, TIMEOUT_META_MS, TIMEOUT_SCORE_MS,
} from '@/server/llmInfer';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const GET_ENDPOINTS = new Set(['health', 'models', 'tasks']);
const MAX_BODY_BYTES = 64 * 1024;

const json = (body: unknown, status = 200) => Response.json(body, { status, headers: { 'Cache-Control': 'no-store' } });

function failure(e: unknown) {
  if (e instanceof ServiceTimeout) return json({ error: `live inference timed out (${e.message})` }, 504);
  if (e instanceof ServiceUnavailable) return json({ error: 'live inference unavailable' }, 503);
  return json({ error: 'live inference unavailable' }, 503);
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(_request: Request, ctx: Ctx) {
  const { path } = await ctx.params;
  const endpoint = path?.join('/');
  if (!GET_ENDPOINTS.has(endpoint)) return json({ error: `unknown endpoint /api/llm/${endpoint}` }, 404);
  try {
    if (endpoint === 'models') forgetModels();
    const res = await callService(`/v1/${endpoint}`, { timeoutMs: TIMEOUT_META_MS });
    return json(res.body, res.status);
  } catch (e) {
    return failure(e);
  }
}

export async function POST(request: Request, ctx: Ctx) {
  const { path } = await ctx.params;
  if (path?.join('/') !== 'score') return json({ error: `unknown endpoint /api/llm/${path?.join('/')}` }, 404);

  const raw = await request.arrayBuffer();
  if (raw.byteLength > MAX_BODY_BYTES) return json({ error: 'request body too large' }, 413);
  let body: unknown;
  try { body = JSON.parse(Buffer.from(raw).toString('utf8')); } catch { return json({ error: 'body is not JSON' }, 400); }

  try {
    let known = await knownModels();
    let check = validateScore(body, known);
    // A run the cached list doesn't know may be new: refresh once before rejecting it
    if ('error' in check && check.error.startsWith('unknown run_ids')) { known = await knownModels(true); check = validateScore(body, known); }
    if ('error' in check) return json({ error: check.error }, check.status);
    const valid = check.body;
    // Generous timeout when a requested model is not loaded yet (the service loads lazily)
    const loaded = new Set(known.filter(m => m.loaded).map(m => m.run_id));
    const timeoutMs = valid.run_ids.every(r => loaded.has(r)) ? TIMEOUT_SCORE_MS : TIMEOUT_FIRST_LOAD_MS;
    const res = await serialised(() => callService('/v1/score', { method: 'POST', body: valid, timeoutMs }));
    if (res.status === 200) forgetModels();   // models may have just been loaded
    return json(res.body, res.status);
  } catch (e) {
    return failure(e);
  }
}
