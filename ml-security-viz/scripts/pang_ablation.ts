/**
 * Pang 2021, federated setting — the ablation of why the paper's collapse (Tables 3–6) does not
 * reproduce here. CPU only. One factor at a time from the federated defaults, plus the
 * batch-norm × {2, 10}-class grid and two "everything reference-like" cells; every seed paired
 * against the same trigger on the honest trajectory θ̃_T and against Table 3's direct poisoner.
 *
 *   npm run ablation:pang -- --all [--seeds 5] [--jobs 28] [--out pang_ablation.jsonl]
 *                              [--cells base,bn] [--extra "federatedRounds=50 …"]
 *   npm run ablation:pang -- --summary pang_ablation.jsonl
 *   npm run ablation:pang -- cell=bn src=mnist seed=1 batchNorm=on      (one run, one JSON line)
 *
 * A run's key=value arguments override the module's defaultConfig (setting is forced to
 * 'federated'); `classes=10` samples all ten classes (500 per class in the training pool) and
 * trains a softmax head. Seeds are independent runs: each draws its own sample, initialisation
 * and stream (Math.random is not seeded).
 */

import { spawn } from 'node:child_process';
import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';
import { sampleImages } from '../src/server/datasets';
import pang from '../src/engine/architectures/pang2021/index';

const CELLS: Record<string, string> = {
  base: '',
  bn: 'batchNorm=on',
  c10: 'classes=10',
  bn_c10: 'classes=10 batchNorm=on',
  sval_test: 'federatedValSource=test',
  T1000: 'federatedRounds=1000',
  all_inner: 'classes=10 batchNorm=on federatedValSource=test federatedRounds=1000',
  all_ref_cos: 'classes=10 batchNorm=on federatedValSource=test federatedRounds=1000 federatedAlign=cosine federatedLambda=0.05',
};

function loadData(src: 'mnist' | 'cifar10', K: number) {
  const classes = K === 10 ? [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] : src === 'mnist' ? [3, 5] : [8, 6];
  // Binary: the app's Pang samples. 10-class: a larger pool (500 per class) for a usable burn-in.
  const sizes = K === 10
    ? (src === 'mnist' ? { train: 5000, valid: 500, test: 5000 } : { train: 5000, valid: 500, test: 2000 })
    : (src === 'mnist' ? { train: 2000, valid: 500, test: 'all' as const } : { train: 1000, valid: 500, test: 1000 });
  const buf = sampleImages({ source: src, classes, ...sizes });
  const hl = buf.readUInt32LE(0);
  const header = JSON.parse(buf.subarray(4, 4 + hl).toString());
  const px = header.shape.reduce((a: number, b: number) => a * b, 1);
  let off = 4 + hl;
  const part = (name: string) => {
    const { count, labels } = header.splits[name];
    const X = Array.from({ length: count }, (_, i) => Array.from(buf.subarray(off + i * px, off + (i + 1) * px), v => v / 255));
    off += count * px;
    return { X, y: labels.map((l: number) => (K === 10 ? l : l === header.classes[0] ? 1 : -1)) };
  };
  return { train: part('train'), valid: part('valid'), test: part('test'), bounds: [0, 1], imageShape: header.shape, numClasses: K };
}

function runOne(args: Record<string, string>): Promise<void> {
  const src = args.src as 'mnist' | 'cifar10';
  const K = +(args.classes ?? 2);
  const ds = loadData(src, K);
  const cfg: Record<string, any> = { ...pang.defaultConfig, setting: 'federated' };
  for (const [k, v] of Object.entries(args)) if (!['src', 'classes', 'cell', 'seed'].includes(k)) cfg[k] = isNaN(+v) ? v : +v;

  const t0 = performance.now();
  const clean = pang.trainClean(ds, cfg);
  const t1 = performance.now();
  const frames: any[] = [];
  return new Promise(resolve => pang.runAttack(ds, { ...clean.modelState, rawModel: clean.rawModel }, cfg, f => frames.push(f), () => {
    const f = frames[frames.length - 1];
    const pts = (x: number) => Math.round(x * 1000) / 10;
    console.log(JSON.stringify({
      cell: args.cell, src, classes: K, seed: args.seed, bn: cfg.batchNorm ?? 'off', valSource: cfg.federatedValSource ?? 'train',
      T: cfg.federatedRounds, objective: cfg.federatedAlign, lambda: cfg.federatedLambda,
      acc0: pts(clean.modelState.testAccuracy), pre: pts(f.preTriggerAccuracy), post: pts(f.poisonedAccuracy),
      drop: pts(f.preTriggerAccuracy - f.poisonedAccuracy), noAccum: pts(f.cleanAccuracy - f.vanillaTriggerAccuracy),
      direct: pts(f.cleanAccuracy - f.directAttackAccuracy), refPre: pts(f.cleanAccuracy),
      alignEnd: +f.gradientAlignment.toPrecision(4), rounds: f.accumulatedRounds,
      burnInS: +((t1 - t0) / 1000).toFixed(1), sPerRound: +((performance.now() - t1) / 1000 / Math.max(1, f.accumulatedRounds)).toFixed(3),
    }));
    resolve();
  }, e => { console.log(JSON.stringify({ cell: args.cell, src, seed: args.seed, error: e })); resolve(); }));
}

const mean = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length;
const sd = (x: number[]) => (x.length > 1 ? Math.sqrt(x.reduce((a, b) => a + (b - mean(x)) ** 2, 0) / (x.length - 1)) : 0);

function summarise(file: string) {
  const rows = readFileSync(file, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l)).filter(r => !r.error);
  const f1 = (x: number) => x.toFixed(1).padStart(5);
  console.log('dataset  cell          n   acc0   pre |  drop noAcc direct | drop−noAcc (mean ± sd)  wins | drop−direct');
  for (const src of ['mnist', 'cifar10']) for (const cell of Object.keys(CELLS)) {
    const v = rows.filter(r => r.src === src && r.cell === cell);
    if (!v.length) continue;
    const d1 = v.map(r => r.drop - r.noAccum), d2 = v.map(r => r.drop - r.direct);
    const m = (k: string) => mean(v.map(r => r[k]));
    console.log(`${src.padEnd(8)} ${cell.padEnd(12)} ${String(v.length).padStart(2)} ${f1(m('acc0'))} ${f1(m('pre'))} | ${f1(m('drop'))} ${f1(m('noAccum'))} ${f1(m('direct'))} | ${f1(mean(d1))} ± ${sd(d1).toFixed(1).padStart(4)}   ${d1.filter(x => x > 0).length}/${v.length} | ${f1(mean(d2))} ± ${sd(d2).toFixed(1)}`);
  }
}

async function runAll(argv: string[]) {
  const opt = (name: string, def: string) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : def; };
  const seeds = +opt('--seeds', '5'), jobs = +opt('--jobs', '28'), out = opt('--out', 'pang_ablation.jsonl');
  const cells = opt('--cells', Object.keys(CELLS).join(',')).split(',');
  const extra = opt('--extra', '').split(' ').filter(Boolean);
  writeFileSync(out, '');
  const list: string[][] = [];
  for (const src of ['cifar10', 'mnist']) for (const cell of cells) for (let s = 1; s <= seeds; s++) {
    list.push([`cell=${cell}`, `src=${src}`, `seed=${s}`, ...CELLS[cell].split(' ').filter(Boolean), ...extra]);
  }
  // Longest first (T = 1000, CIFAR), so the tail of the pool is short
  list.sort((a, b) => Number(b.some(x => x.includes('1000'))) - Number(a.some(x => x.includes('1000'))));
  let next = 0, done = 0;
  const worker = async () => {
    while (next < list.length) {
      const job = list[next++];
      await new Promise<void>(res => {
        const p = spawn(process.execPath, [...process.execArgv, __filename, ...job], { stdio: ['ignore', 'pipe', 'inherit'] });
        let buf = '';
        p.stdout.on('data', d => { buf += d; });
        p.on('close', () => { appendFileSync(out, buf); done++; console.error(`[${done}/${list.length}] ${job.slice(0, 3).join(' ')}`); res(); });
      });
    }
  };
  await Promise.all(Array.from({ length: Math.min(jobs, list.length) }, worker));
  summarise(out);
}

const argv = process.argv.slice(2);
if (argv[0] === '--all') runAll(argv);
else if (argv[0] === '--summary') summarise(argv[1]);
else runOne(Object.fromEntries(argv.map(a => a.split('=') as [string, string])));
