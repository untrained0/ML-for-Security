/**
 * Client side of POST /api/attack: upload the dataset and clean model the browser already has,
 * and receive the attack's TraceFrames as they are computed on the server (src/server/attack.ts).
 */

import type { TraceFrame } from '@/engine/architectures';

export interface ServerAttackMeta {
  backend: 'cpu' | 'cuda';
  device: string;
}

/** The server could not be used at all (unreachable, old build, 5xx before any frame). */
export class ServerUnavailableError extends Error {}

/**
 * 8-bit images (every feature exactly k/255) travel as base64 bytes: as JSON a 3072-d CIFAR image
 * costs ~18 characters per pixel. The server rebuilds the identical doubles (server/attack.ts).
 * Anything that is not exactly 8-bit is sent as it is.
 */
function packPixels(X: number[][]): any {
  const n = X.length, d = X[0]?.length ?? 0;
  const bytes = new Uint8Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      const v = X[i][j] * 255, k = Math.round(v);
      if (k < 0 || k > 255 || k / 255 !== X[i][j]) return X;
      bytes[i * d + j] = k;
    }
  }
  let bin = '';
  for (let o = 0; o < bytes.length; o += 0x8000) bin += String.fromCharCode(...bytes.subarray(o, o + 0x8000));
  return { u8: btoa(bin), n, d };
}

/**
 * The dataset as the attack needs it: display-only payloads (image pixels, the PCA window, the
 * duplicate X2D) stay in the browser, and image features go as bytes.
 */
function slimDataset(ds: any) {
  const pack = (X: number[][]) => (ds.imageShape && X?.length ? packPixels(X) : X);
  const split = (s: any) => s && { X: pack(s.X), y: s.y, ...(s.Y ? { Y: s.Y } : {}) };
  const { train, valid, test, images, view, pcaState, X2D, featureNames, ...rest } = ds;
  return { ...rest, train: split(train), valid: split(valid), test: split(test) };
}

/** The clean model without its kernel matrix, kernel function or copy of the training set. */
function slimModel(m: any) {
  if (!m) return m;
  const { rawModel, ...rest } = m;
  if (!rawModel) return rest;
  const { K, X, y, kernelFn, ...raw } = rawModel;
  return { ...rest, rawModel: raw };
}

async function gzip(text: string): Promise<{ body: BodyInit; encoding?: string }> {
  if (typeof CompressionStream === 'undefined') return { body: text };
  const stream = new Blob([text]).stream().pipeThrough(new CompressionStream('gzip'));
  return { body: await new Response(stream).blob(), encoding: 'gzip' };
}

export async function runAttackOnServer(opts: {
  algorithm: string;
  datasetKey: string;
  dataset: any;
  cleanModel: any;
  config: Record<string, any>;
  signal: AbortSignal;
  onMeta: (meta: ServerAttackMeta) => void;
  onFrame: (frame: TraceFrame) => void;
}): Promise<{ ms: number; frames: number }> {
  const payload = JSON.stringify({
    algorithm: opts.algorithm,
    datasetKey: opts.datasetKey,
    config: opts.config,
    dataset: slimDataset(opts.dataset),
    cleanModel: slimModel(opts.cleanModel),
  });
  const { body, encoding } = await gzip(payload);

  let res: Response;
  try {
    res = await fetch('/api/attack', {
      method: 'POST',
      body,
      headers: { 'Content-Type': 'application/json', ...(encoding ? { 'Content-Encoding': encoding } : {}) },
      signal: opts.signal,
    });
  } catch (e: any) {
    if (opts.signal.aborted) throw e;
    throw new ServerUnavailableError(`attack server unreachable: ${e.message}`);
  }
  if (!res.ok || !res.body) {
    const msg = await res.json().then(j => j.error).catch(() => res.statusText);
    const Err = res.status === 404 || res.status >= 500 ? ServerUnavailableError : Error;
    throw new Err(`attack server: ${msg}`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffered = '';
  let result: { ms: number; frames: number } | null = null;
  const handle = (line: string) => {
    if (!line) return;
    const msg = JSON.parse(line);
    if (msg.type === 'meta') opts.onMeta({ backend: msg.backend, device: msg.device });
    else if (msg.type === 'frame') opts.onFrame(msg.frame);
    else if (msg.type === 'done') result = { ms: msg.ms, frames: msg.frames };
    else if (msg.type === 'error') throw new Error(msg.message);
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffered += value;
    let nl: number;
    while ((nl = buffered.indexOf('\n')) >= 0) {
      handle(buffered.slice(0, nl));
      buffered = buffered.slice(nl + 1);
    }
  }
  handle(buffered.trim());
  if (!result) throw new Error('attack server closed the stream before the attack finished');
  return result;
}

/** GET /api/compute, or null when there is no attack server (e.g. a static deployment). */
export async function fetchServerCompute(): Promise<{ backend: string; device: string; cuda: any; error?: string } | null> {
  try {
    const res = await fetch('/api/compute', { cache: 'no-store' });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}
