/**
 * fetch_datasets.mjs — download the ORIGINAL benchmark datasets used by the papers into data/
 * (server-side only; the app's API routes sample from them per request, see src/server/datasets.ts).
 *
 *   node scripts/fetch_datasets.mjs                  # fetch anything missing
 *   node scripts/fetch_datasets.mjs --force          # re-download everything
 *   node scripts/fetch_datasets.mjs --only mnist,regression   # just these (mnist|regression|cifar10)
 *   node scripts/fetch_datasets.mjs --probe          # check every source host is reachable, no download
 *
 * Runs automatically before `npm run dev` / `npm run build`. Failures are reported but never fail
 * the build: the affected dataset shows a load error in the app.
 *
 *   data/mnist/        the complete MNIST (LeCun et al.): 60 000 train + 10 000 test, IDX format
 *                      Biggio et al. 2012 §3.2 (7v1, 9v8, 4v0); Pang et al. 2021
 *   data/cifar10/      the complete CIFAR-10 binary version (Krizhevsky 2009): 5 × 10 000 train
 *                      + 10 000 test, 32×32×3; Pang et al. 2021
 *   data/regression/   Jagielski et al. 2018 §V-A, the authors' preprocessed files
 *                      (github.com/jagielski/manip-ml): Warfarin/IWPC, Lending Club (all 168 048
 *                      records — the paper samples from it), Ames house prices. Response in
 *                      column 0, one-hot categoricals named "Feature:value", everything in [0,1].
 *
 * Set DATA_DIR to keep the data elsewhere (the server reads the same variable).
 *
 * Proxies: HTTPS_PROXY / HTTP_PROXY / ALL_PROXY (upper or lower case) are honoured, with optional
 * credentials (http://user:pass@proxy.example:3128, URL-encode special characters) and NO_PROXY.
 * Node's built-in fetch ignores these variables, so downloads go through node:http/https here,
 * tunnelling HTTPS with CONNECT. A proxy that re-signs TLS needs its root certificate in
 * NODE_EXTRA_CA_CERTS=/path/to/ca.pem.
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import http from 'node:http';
import https from 'node:https';
import tls from 'node:tls';
import os from 'node:os';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const APP = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = process.env.DATA_DIR ?? path.join(APP, 'data');
const FORCE = process.argv.includes('--force');
const PROBE = process.argv.includes('--probe');
const onlyArg = process.argv.indexOf('--only');
const ONLY = onlyArg > 0 ? new Set((process.argv[onlyArg + 1] ?? '').split(',').filter(Boolean)) : null;

const MNIST_MIRRORS = [
  'https://storage.googleapis.com/cvdf-datasets/mnist/',
  'https://ossci-datasets.s3.amazonaws.com/mnist/',
];
const MNIST_FILES = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'];
const CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz';
const CIFAR_FILES = ['data_batch_1.bin', 'data_batch_2.bin', 'data_batch_3.bin', 'data_batch_4.bin', 'data_batch_5.bin', 'test_batch.bin', 'batches.meta.txt'];
const MANIP_ML = 'https://raw.githubusercontent.com/jagielski/manip-ml/master/datasets/';
const REGRESSION = [
  { key: 'warfarin', file: 'pharm-preproc.csv' },
  { key: 'house', file: 'house-processed.csv' },
  { key: 'loan', file: 'loan-processed.csv' },
];

const need = (file) => FORCE || !fs.existsSync(file);

// ── HTTP(S) client with proxy support ────────────────────────────────

const TIMEOUT_MS = 60_000;          // idle socket timeout, not a cap on total download time

const env = (name) => process.env[name] ?? process.env[name.toLowerCase()];

/** The proxy for a URL from the standard environment variables, or null (NO_PROXY honoured). */
function proxyFor(url) {
  const { hostname, protocol } = new URL(url);
  const noProxy = (env('NO_PROXY') ?? '').split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
  if (noProxy.some(p => p === '*' || hostname === p.replace(/^\./, '') || hostname.endsWith(p.startsWith('.') ? p : `.${p}`))) {
    return null;
  }
  const raw = protocol === 'https:'
    ? env('HTTPS_PROXY') ?? env('ALL_PROXY') ?? env('HTTP_PROXY')
    : env('HTTP_PROXY') ?? env('ALL_PROXY');
  return raw ? new URL(raw.includes('://') ? raw : `http://${raw}`) : null;
}

const describeProxy = (p) => `${p.protocol}//${p.username ? `${decodeURIComponent(p.username)}:***@` : ''}${p.host}`;

function proxyAuth(proxy) {
  if (!proxy.username) return {};
  const creds = `${decodeURIComponent(proxy.username)}:${decodeURIComponent(proxy.password)}`;
  return { 'Proxy-Authorization': `Basic ${Buffer.from(creds).toString('base64')}` };
}

/** Turn low-level failures into messages that say what to change. */
function explain(err, url, proxy) {
  const code = err.code ?? '';
  const via = proxy ? ` via proxy ${describeProxy(proxy)}` : '';
  if (/CERT|SELF_SIGNED|UNABLE_TO_(GET|VERIFY)/.test(code)) {
    return new Error(`${url}: TLS certificate rejected (${code}: ${err.message})${via}. If the proxy inspects TLS, point NODE_EXTRA_CA_CERTS at its root certificate.`);
  }
  if (!proxy && ['ENOTFOUND', 'EAI_AGAIN', 'ECONNREFUSED', 'ETIMEDOUT', 'ECONNRESET', 'ENETUNREACH'].includes(code)) {
    return new Error(`${url}: ${code} — no direct internet access? Set HTTPS_PROXY=http://[user:pass@]proxy-host:port.`);
  }
  if (proxy && ['ENOTFOUND', 'EAI_AGAIN', 'ECONNREFUSED', 'EHOSTUNREACH', 'ENETUNREACH'].includes(code) &&
      new RegExp(proxy.hostname.replace(/[.]/g, '\.')).test(err.message)) {
    return new Error(`cannot reach proxy ${describeProxy(proxy)} (${code}) — check the host/port in HTTPS_PROXY`);
  }
  return new Error(`${url}${via}: ${err.message}`);
}

/** GET a URL, following redirects, through the configured proxy if any. Resolves to the response. */
function open(url, headers = {}, redirectsLeft = 5) {
  const u = new URL(url);
  const proxy = proxyFor(url);
  const reqHeaders = { 'User-Agent': 'ml-security-viz-fetch-datasets', ...headers };

  const onResponse = (res, resolve, reject) => {
    if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
      res.resume();
      if (redirectsLeft <= 0) return reject(new Error(`${url}: too many redirects`));
      return resolve(open(new URL(res.headers.location, url).href, headers, redirectsLeft - 1));
    }
    resolve(res);
  };
  const guard = (req, reject) => {
    req.setTimeout(TIMEOUT_MS, () =>
      req.destroy(Object.assign(new Error(`no data for ${TIMEOUT_MS / 1000}s`), { code: 'ETIMEDOUT' })));
    req.on('error', (e) => reject(explain(e, url, proxy)));
    return req;
  };

  return new Promise((resolve, reject) => {
    if (!proxy) {
      const lib = u.protocol === 'https:' ? https : http;
      guard(lib.get(u, { headers: reqHeaders }, (res) => onResponse(res, resolve, reject)), reject);
      return;
    }
    const proxyPort = Number(proxy.port) || 80;
    if (u.protocol === 'http:') {
      // Plain HTTP through a proxy: send it the absolute URL
      guard(http.get({
        host: proxy.hostname, port: proxyPort, path: u.href,
        headers: { ...reqHeaders, Host: u.host, ...proxyAuth(proxy) },
      }, (res) => onResponse(res, resolve, reject)), reject);
      return;
    }
    // HTTPS: open a CONNECT tunnel, then speak TLS to the origin through it
    const target = `${u.hostname}:${u.port || 443}`;
    const connect = guard(http.request({
      host: proxy.hostname, port: proxyPort, method: 'CONNECT', path: target, agent: false,
      headers: { Host: target, ...proxyAuth(proxy) },
    }), reject);
    connect.on('connect', (cres, socket) => {
      if (cres.statusCode !== 200) {
        socket.destroy();
        const wants = String(cres.headers['proxy-authenticate'] ?? '').split(/[\s,]/)[0];
        const hint = cres.statusCode !== 407 ? ''
          : wants && !/^basic$/i.test(wants)
            ? ` — the proxy asks for ${wants} authentication; only Basic is built in (curl is tried next if installed)`
            : proxy.username
              ? ' — the proxy rejected these credentials (wrong user/password? special characters URL-encoded?)'
              : ' — the proxy wants credentials: HTTPS_PROXY=http://user:pass@host:port (URL-encode special characters)';
        reject(new Error(`${url}: proxy ${describeProxy(proxy)} refused the tunnel (${cres.statusCode} ${cres.statusMessage})${hint}`));
        return;
      }
      // `host` as well as `servername`: the certificate's names are checked against the first
      // one present, and which one that is has changed between Node versions
      const secure = tls.connect({ socket, host: u.hostname, servername: u.hostname });
      // While the request is active its own 'error' handler reports socket failures; after the
      // response the tunnel is closed, and a late reset from the far end must not crash the script.
      // On failure (e.g. a proxy that answers with its own certificate) close the tunnel too, or
      // the open socket to the proxy keeps the process alive.
      secure.on('error', () => { secure.destroy(); socket.destroy(); });
      guard(https.get(u, { headers: reqHeaders, agent: false, createConnection: () => secure }, (res) => {
        res.on('close', () => secure.destroy());
        onResponse(res, resolve, reject);
      }), reject);
    });
    connect.end();
  });
}

/**
 * Download a URL into memory. Uses the built-in client; if that fails for a network/TLS reason
 * (not an HTTP error status) and curl is installed, retries once with curl, which some proxies
 * handle better. `lastVia` records which client succeeded.
 */
let lastVia = 'node';
async function download(url, opts = {}) {
  if (process.env.FETCH_USE_CURL === '1' && hasCurl()) {     // escape hatch: curl only
    lastVia = 'curl';
    return curlDownload(url, opts);
  }
  try {
    const buf = await nodeDownload(url, opts);
    lastVia = 'node';
    return buf;
  } catch (e) {
    if (e.httpStatus || !hasCurl()) throw e;
    console.log(`[${opts.label ?? 'fetch'}] ${e.message}\n    → retrying with curl`);
    const buf = await curlDownload(url, opts);
    lastVia = 'curl';
    return buf;
  }
}

let curlAvailable;
function hasCurl() {
  curlAvailable ??= spawnSync('curl', ['--version'], { stdio: 'ignore' }).status === 0;
  return curlAvailable;
}

/**
 * curl reads the proxy (with URL-encoded credentials) from its environment, so the password never
 * appears on the command line / in the process list. Writes to a temp file so curl's own progress
 * bar can go to the terminal for large downloads.
 */
function curlDownload(url, { label, headers = {} } = {}) {
  const proxy = proxyFor(url);
  const env = { ...process.env };
  for (const k of ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy']) delete env[k];
  if (proxy) env.HTTPS_PROXY = env.https_proxy = env.HTTP_PROXY = env.http_proxy = proxy.href;
  const tmp = path.join(os.tmpdir(), `fetch-datasets-${process.pid}-${Date.now()}`);
  const args = ['-fL', '--retry', '2', '--connect-timeout', '30', label ? '-#' : '-sS', '-o', tmp];
  for (const [k, v] of Object.entries(headers)) args.push('-H', `${k}: ${v}`);
  args.push(url);
  return new Promise((resolve, reject) => {
    const child = spawn('curl', args, { env, stdio: ['ignore', 'inherit', label ? 'inherit' : 'pipe'] });
    let stderr = '';
    child.stderr?.on('data', d => { stderr += d; });
    child.on('error', reject);
    child.on('close', (status) => {
      try {
        if (status !== 0) return reject(new Error(`${url}: curl exited with ${status}${stderr ? ` — ${stderr.trim()}` : ''}`));
        resolve(fs.readFileSync(tmp));
      } finally {
        fs.rmSync(tmp, { force: true });
      }
    });
  });
}

/** Download a URL into memory with the built-in client, logging progress every 10 MB when `label` is given. */
async function nodeDownload(url, { label, headers } = {}) {
  const res = await open(url, headers);
  if (res.statusCode !== 200 && res.statusCode !== 206) {
    res.resume();
    throw Object.assign(new Error(`${res.statusCode} ${res.statusMessage} for ${url}`), { httpStatus: res.statusCode });
  }
  const total = Number(res.headers['content-length']) || 0;
  const chunks = [];
  let got = 0, nextLog = 10e6;
  for await (const chunk of res) {
    chunks.push(chunk);
    got += chunk.length;
    if (label && got >= nextLog) {
      console.log(`[${label}] ${(got / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ''} MB`);
      nextLog += 10e6;
    }
  }
  if (total && got < total) throw new Error(`${url}: connection closed after ${got} of ${total} bytes`);
  return Buffer.concat(chunks);
}

async function downloadFirst(urls) {
  let lastErr;
  for (const url of urls) {
    try { return await download(url); } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

// ── MNIST: keep the four IDX files, decompressed ─────────────────────

async function fetchMNIST() {
  const dir = path.join(ROOT, 'mnist');
  fs.mkdirSync(dir, { recursive: true });
  for (const f of MNIST_FILES) {
    const out = path.join(dir, `${f}.idx`);
    if (!need(out)) continue;
    const gz = await downloadFirst(MNIST_MIRRORS.map(m => `${m}${f}.gz`));
    fs.writeFileSync(out, zlib.gunzipSync(gz));
    console.log(`[mnist] ${f}`);
  }
  console.log('[mnist] ready');
}

// ── CIFAR-10: extract the binary batches from the official tarball ───

/** Minimal ustar reader: 512-byte header (name at 0, octal size at 124), data padded to 512. */
function* untar(buf) {
  for (let off = 0; off + 512 <= buf.length;) {
    const name = buf.toString('utf8', off, off + 100).replace(/\0.*$/s, '');
    if (!name) return;
    const size = parseInt(buf.toString('utf8', off + 124, off + 136).replace(/\0.*$/s, '').trim() || '0', 8);
    yield { name, data: buf.subarray(off + 512, off + 512 + size) };
    off += 512 + Math.ceil(size / 512) * 512;
  }
}

async function fetchCIFAR() {
  const dir = path.join(ROOT, 'cifar10');
  if (CIFAR_FILES.every(f => !need(path.join(dir, f)))) { console.log('[cifar10] present, skipping'); return; }
  fs.mkdirSync(dir, { recursive: true });
  console.log('[cifar10] downloading 170 MB …');
  const tar = zlib.gunzipSync(await download(CIFAR_URL, { label: 'cifar10' }));
  for (const { name, data } of untar(tar)) {
    const base = path.basename(name);
    if (CIFAR_FILES.includes(base)) fs.writeFileSync(path.join(dir, base), data);
  }
  const missing = CIFAR_FILES.filter(f => !fs.existsSync(path.join(dir, f)));
  if (missing.length) throw new Error(`CIFAR-10 archive lacked ${missing.join(', ')}`);
  console.log('[cifar10] ready');
}

// ── Regression (Jagielski et al. 2018) ───────────────────────────────

/** Split one CSV line, honouring double quotes (Warfarin has header names containing commas). */
function splitCSVLine(line) {
  const out = [];
  let cur = '', quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

async function fetchRegression() {
  const dir = path.join(ROOT, 'regression');
  fs.mkdirSync(dir, { recursive: true });
  for (const { key, file } of REGRESSION) {
    const out = path.join(dir, `${key}.csv`);
    if (!need(out)) { console.log(`[${key}] present, skipping`); continue; }

    const lines = (await download(MANIP_ML + file, { label: key })).toString('utf8').split(/\r?\n/).filter(l => l.trim());
    const names = splitCSVLine(lines[0]);
    const data = lines.slice(1).map(l => l.split(',').map(Number));
    if (data.some(r => r.length !== names.length || r.some(v => !Number.isFinite(v)))) {
      throw new Error(`${file}: malformed row (expected ${names.length} numeric columns)`);
    }
    // §V-A: "numerical features are normalized into [0,1]". The authors' loan file carries one
    // un-normalised duplicate column (the second `pub_rec`); re-apply min–max where needed.
    for (let j = 0; j < names.length; j++) {
      let lo = Infinity, hi = -Infinity;
      for (const r of data) { lo = Math.min(lo, r[j]); hi = Math.max(hi, r[j]); }
      if (lo < 0 || hi > 1) {
        for (const r of data) r[j] = (r[j] - lo) / (hi - lo || 1);
        console.log(`[${key}] rescaled column "${names[j]}" from [${lo}, ${hi}] to [0,1]`);
      }
    }
    const header = names.map(n => `"${n.replace(/"/g, '')}"`).join(',');
    const body = data.map(r => r.map(v => String(+v.toPrecision(6))).join(','));
    fs.writeFileSync(out, [header, ...body].join('\n') + '\n');
    console.log(`[${key}] wrote ${data.length} rows × ${names.length - 1} features`);
  }
}

// ── probe: can every source host be reached? ─────────────────────────

async function probe() {
  const proxies = new Set([...MNIST_MIRRORS, MANIP_ML, CIFAR_URL]
    .map(u => proxyFor(u))
    .map(p => (p ? describeProxy(p) : 'none (direct)')));
  console.log(`[probe] proxy: ${[...proxies].join(', ')}`);
  const checks = [
    ...MNIST_MIRRORS.map(m => ['mnist', `${m}t10k-labels-idx1-ubyte.gz`, {}]),
    ['regression', 'https://raw.githubusercontent.com/jagielski/manip-ml/master/README.md', {}],
    ['cifar10', CIFAR_URL, { Range: 'bytes=0-1023' }],
  ];
  let ok = true;
  for (const [name, url, headers] of checks) {
    const t0 = Date.now();
    try {
      const buf = await download(url, { headers });
      console.log(`[probe] OK    ${name.padEnd(10)} ${new URL(url).host} (${buf.length} B, ${Date.now() - t0} ms, via ${lastVia})`);
    } catch (e) {
      ok = false;
      console.log(`[probe] FAIL  ${name.padEnd(10)} ${e.message}`);
    }
  }
  console.log(ok ? '[probe] all sources reachable' : '[probe] some sources failed (MNIST needs only one mirror)');
}

// ── main ─────────────────────────────────────────────────────────────

if (PROBE) {
  await probe();
} else {
  let failed = false;
  for (const [name, job] of [['mnist', fetchMNIST], ['regression', fetchRegression], ['cifar10', fetchCIFAR]]) {
    if (ONLY && !ONLY.has(name)) continue;
    try {
      await job();
    } catch (e) {
      failed = true;
      console.warn(`[fetch_datasets] ${name} failed: ${e.message}`);
    }
  }
  if (failed) {
    console.warn('[fetch_datasets] some datasets are missing — they will show a load error in the app. ' +
      'Run with --probe to check connectivity.');
  }
}

// All work is done; don't let a lingering proxy tunnel keep the process (and npm run dev/build) waiting.
process.exit(0);
