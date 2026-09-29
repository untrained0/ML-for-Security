/**
 * Server-side access to the ORIGINAL datasets in data/ (see scripts/fetch_datasets.mjs).
 *
 * Each dataset is read once per server process and kept in memory (MNIST ≈ 55 MB, CIFAR-10
 * ≈ 185 MB, Lending Club ≈ 65 MB); every request then draws a fresh random sample of the size
 * the paper's protocol asks for, so the browser only ever downloads what it trains on.
 */

import fs from 'node:fs';
import path from 'node:path';

// Resolved at runtime (DATA_DIR or ./data); excluded from build-time file tracing
const ROOT = process.env.DATA_DIR ?? path.join(/*turbopackIgnore: true*/ process.cwd(), 'data');

export class DatasetError extends Error {
  constructor(message: string, public status = 400) { super(message); }
}

function readFile(...parts: string[]) {
  const file = path.join(/*turbopackIgnore: true*/ ROOT, ...parts);
  if (!fs.existsSync(file)) {
    throw new DatasetError(`${parts.join('/')} is missing on the server — run \`npm run fetch-data\``, 503);
  }
  return fs.readFileSync(file);
}

function shuffle(a: number[]) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ── Images: MNIST and CIFAR-10 ───────────────────────────────────────

interface ImageSplit { pixels: Uint8Array; labels: Uint8Array; count: number }
interface ImageSource {
  shape: number[];              // [28, 28] or [32, 32, 3] (channels last)
  classNames: string[];
  train: ImageSplit;
  test: ImageSplit;
}

const imageCache = new Map<string, ImageSource>();

/** IDX (LeCun): big-endian magic whose last byte is the rank, then the dims, then uint8 data. */
function readIDX(buf: Buffer) {
  const rank = buf[3];
  const dims = Array.from({ length: rank }, (_, i) => buf.readUInt32BE(4 + 4 * i));
  return { dims, data: new Uint8Array(buf.buffer, buf.byteOffset + 4 + 4 * rank, dims.reduce((a, b) => a * b, 1)) };
}

function loadMNIST(): ImageSource {
  const split = (prefix: string): ImageSplit => {
    const img = readIDX(readFile('mnist', `${prefix}-images-idx3-ubyte.idx`));
    const lab = readIDX(readFile('mnist', `${prefix}-labels-idx1-ubyte.idx`));
    return { pixels: img.data, labels: lab.data, count: img.dims[0] };
  };
  return {
    shape: [28, 28],
    classNames: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    train: split('train'),
    test: split('t10k'),
  };
}

/**
 * CIFAR-10 binary records are <1 label byte><1024 R><1024 G><1024 B>; stored here as 32×32×3
 * channels-last (interleaved RGB), the layout the canvas draws.
 */
function loadCIFAR(): ImageSource {
  const PX = 1024, REC = 1 + 3 * PX;
  const split = (files: string[]): ImageSplit => {
    const bufs = files.map(f => readFile('cifar10', f));
    const count = bufs.reduce((s, b) => s + b.length / REC, 0);
    const pixels = new Uint8Array(count * 3 * PX), labels = new Uint8Array(count);
    let n = 0;
    for (const b of bufs) {
      for (let r = 0; r < b.length / REC; r++, n++) {
        const o = r * REC;
        labels[n] = b[o];
        for (let p = 0; p < PX; p++) {
          for (let c = 0; c < 3; c++) pixels[n * 3 * PX + p * 3 + c] = b[o + 1 + c * PX + p];
        }
      }
    }
    return { pixels, labels, count };
  };
  const names = readFile('cifar10', 'batches.meta.txt').toString('utf8').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  return {
    shape: [32, 32, 3],
    classNames: names,
    train: split([1, 2, 3, 4, 5].map(i => `data_batch_${i}.bin`)),
    test: split(['test_batch.bin']),
  };
}

function imageSource(source: string): ImageSource {
  if (!imageCache.has(source)) {
    if (source === 'mnist') imageCache.set(source, loadMNIST());
    else if (source === 'cifar10') imageCache.set(source, loadCIFAR());
    else throw new DatasetError(`Unknown image source "${source}"`);
  }
  return imageCache.get(source)!;
}

export interface ImageSampleRequest {
  source: string;
  classes: number[];            // two class ids (the first becomes label +1), or more for a multi-class sample
  train: number;
  valid: number;
  test: number | 'all';         // 'all' = the complete test split of those classes
}

/**
 * Binary response: uint32 LE header length, JSON header, then uint8 pixels of the train, valid
 * and test samples back to back. Train and valid are drawn without replacement from the
 * original training split, test from the original test split.
 */
export function sampleImages(req: ImageSampleRequest): Buffer {
  const src = imageSource(req.source);
  const want = req.classes;
  if (want.length < 2 || new Set(want).size !== want.length || want.some(c => !(c >= 0 && c < src.classNames.length))) {
    throw new DatasetError(`classes must be two or more distinct ids in 0–${src.classNames.length - 1}`);
  }
  const keep = new Set(want);
  const of = (s: ImageSplit) => Array.from({ length: s.count }, (_, i) => i).filter(i => keep.has(s.labels[i]));

  const pool = shuffle(of(src.train));
  if (req.train + req.valid > pool.length) {
    throw new DatasetError(`only ${pool.length} training images of classes ${want.join(',')}`);
  }
  const testPool = of(src.test);
  const testIdx = req.test === 'all' ? testPool : shuffle(testPool).slice(0, req.test);
  const parts = [
    { name: 'train', split: src.train, idx: pool.slice(0, req.train) },
    { name: 'valid', split: src.train, idx: pool.slice(req.train, req.train + req.valid) },
    { name: 'test', split: src.test, idx: testIdx },
  ];

  const px = src.shape.reduce((x, y) => x * y, 1);
  const header = Buffer.from(JSON.stringify({
    source: req.source,
    shape: src.shape,
    classes: want,
    classNames: want.map(c => src.classNames[c]),
    splits: Object.fromEntries(parts.map(p => [p.name, { count: p.idx.length, labels: p.idx.map(i => p.split.labels[i]) }])),
  }), 'utf8');
  const body = Buffer.alloc(parts.reduce((s, p) => s + p.idx.length, 0) * px);
  let o = 0;
  for (const p of parts) {
    for (const i of p.idx) { body.set(p.split.pixels.subarray(i * px, (i + 1) * px), o); o += px; }
  }
  const len = Buffer.alloc(4);
  len.writeUInt32LE(header.length, 0);
  return Buffer.concat([len, header, body]);
}

// ── Regression (Jagielski et al. 2018) ───────────────────────────────

const REGRESSION_KEYS = new Set(['warfarin', 'loan', 'house']);
const regressionCache = new Map<string, { header: string; rows: string[] }>();

/** A random sample of n records of the complete file, as CSV (header + rows). */
export function sampleRegression(key: string, n: number): string {
  if (!REGRESSION_KEYS.has(key)) throw new DatasetError(`Unknown regression dataset "${key}"`);
  if (!regressionCache.has(key)) {
    const lines = readFile('regression', `${key}.csv`).toString('utf8').split(/\r?\n/).filter(Boolean);
    regressionCache.set(key, { header: lines[0], rows: lines.slice(1) });
  }
  const { header, rows } = regressionCache.get(key)!;
  const idx = shuffle(Array.from({ length: rows.length }, (_, i) => i)).slice(0, Math.min(n, rows.length));
  return [header, ...idx.map(i => rows[i])].join('\n') + '\n';
}
