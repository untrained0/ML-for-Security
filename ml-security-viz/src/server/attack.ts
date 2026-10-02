/**
 * Runs an attack on the server and streams its TraceFrames back as gzip-compressed NDJSON.
 *
 * The browser uploads the dataset and clean model it already has (so the server attacks exactly
 * what the canvas shows) and gets one JSON line per message:
 *   {"type":"meta","backend":"cuda","device":"NVIDIA GeForce RTX 4090"}
 *   {"type":"frame","frame":{…TraceFrame…}}          (as many as the attack emits)
 *   {"type":"done","ms":1234,"frames":57}   or   {"type":"error","message":"…"}
 *
 * The module's own `runAttack` is used unchanged: it already yields between steps with
 * setTimeout, which here keeps Node's event loop free to serve other requests. Aborting `signal`
 * (the client disconnected) stops it at the next step. The gzip stream is flushed after every
 * line so frames arrive as they are computed, not when a compression block fills up.
 */

import zlib from 'node:zlib';
import { Readable } from 'node:stream';
import { getAlgorithm } from '@/engine/architectures';
import { serverCompute } from './compute';

export interface AttackRequest {
  algorithm: string;
  datasetKey?: string;
  config: Record<string, any>;
  dataset: any;
  cleanModel: any;
}

/** JSON.stringify replacer: typed arrays as plain arrays (functions are dropped by JSON itself). */
function replacer(_key: string, v: any) {
  return ArrayBuffer.isView(v) && !(v instanceof DataView) ? Array.from(v as any) : v;
}

/** Undo lib/serverAttack.ts's packPixels: base64 bytes back to rows of k/255. */
function unpackPixels(X: any): number[][] {
  if (!X || Array.isArray(X) || typeof X.u8 !== 'string') return X;
  const bytes = Buffer.from(X.u8, 'base64');
  const { n, d } = X;
  if (bytes.length !== n * d) throw new Error('packed pixels have the wrong size');
  return Array.from({ length: n }, (_, i) => Array.from(bytes.subarray(i * d, (i + 1) * d), k => k / 255));
}

function unpackDataset(ds: any) {
  for (const name of ['train', 'valid', 'test']) if (ds?.[name]) ds[name].X = unpackPixels(ds[name].X);
  return ds;
}

export function streamAttack(req: AttackRequest, signal: AbortSignal): ReadableStream<Uint8Array> {
  const gz = zlib.createGzip({ level: 6 });
  // Stop the attack when the request is aborted OR the response stream is torn down (Next
  // cancels the body when the client goes away; the request signal alone may not fire).
  const stop = new AbortController();
  signal.addEventListener('abort', () => stop.abort(), { once: true });
  gz.once('close', () => stop.abort());
  let closed = false;
  const send = (msg: any) => {
    if (closed || gz.destroyed) return;
    gz.write(JSON.stringify(msg, replacer) + '\n');
    gz.flush(zlib.constants.Z_SYNC_FLUSH);
  };
  const close = () => {
    if (closed) return;
    closed = true;
    gz.end();
  };
  stop.signal.addEventListener('abort', close, { once: true });

  const started = performance.now();
  let frames = 0;
  try {
    const info = serverCompute();
    if (info.error) throw new Error(info.error);
    const alg = getAlgorithm(req.algorithm);
    send({ type: 'meta', backend: info.backend, device: info.device });
    alg.runAttack(
      unpackDataset(req.dataset), req.cleanModel, req.config,
      (frame) => { frames++; send({ type: 'frame', frame }); },
      () => { send({ type: 'done', ms: Math.round(performance.now() - started), frames }); close(); },
      (message) => { send({ type: 'error', message }); close(); },
      stop.signal,
    );
  } catch (e: any) {
    send({ type: 'error', message: String(e?.message ?? e) });
    close();
  }
  return Readable.toWeb(gz) as unknown as ReadableStream<Uint8Array>;
}
