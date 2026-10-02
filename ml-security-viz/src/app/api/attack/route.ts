/**
 * POST /api/attack — run an attack on the server (src/server/attack.ts) and stream its frames.
 *
 * Body: JSON (gzip allowed, `Content-Encoding: gzip`) {algorithm, datasetKey, config, dataset,
 * cleanModel}. Response: gzip NDJSON, one message per line, flushed per frame.
 */

import zlib from 'node:zlib';
import { streamAttack, type AttackRequest } from '@/server/attack';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Image features arrive as bytes (lib/serverAttack.ts), so even CIFAR-10 is a few MB; anything far beyond that is not a dataset from this app
const MAX_BODY_BYTES = 256 * 2 ** 20;

export async function POST(request: Request) {
  let req: AttackRequest;
  try {
    const raw = Buffer.from(await request.arrayBuffer());
    const json = request.headers.get('content-encoding') === 'gzip'
      ? zlib.gunzipSync(raw, { maxOutputLength: MAX_BODY_BYTES })
      : raw;
    if (json.length > MAX_BODY_BYTES) throw new Error('body too large');
    req = JSON.parse(json.toString('utf8'));
  } catch (e: any) {
    return Response.json({ error: `unreadable request body: ${e.message}` }, { status: 400 });
  }
  if (typeof req?.algorithm !== 'string' || !(req.dataset?.train?.X?.length || req.dataset?.train?.X?.n) || !req.config) {
    return Response.json({ error: 'expected {algorithm, config, dataset: {train, valid, test}, cleanModel}' }, { status: 400 });
  }
  return new Response(streamAttack(req, request.signal), {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Content-Encoding': 'gzip',
      'Cache-Control': 'no-store, no-transform',
      'X-Accel-Buffering': 'no',
    },
  });
}
