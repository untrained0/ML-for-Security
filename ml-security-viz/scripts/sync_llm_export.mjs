#!/usr/bin/env node
/**
 * Copies an LLM-attack export into public/ for a view module, verifying every file against the
 * export's manifest.json (byte size + sha256). Nothing is written unless every source file
 * verifies; every copy is verified again after writing. Any mismatch exits non-zero.
 *
 *   npm run sync:llm-export -- <src-dir> [dest-dir]        (dest defaults to public/llm/<basename of src>)
 *   npm run sync:llm-export -- --verify <dir>              (check files already in place, copy nothing)
 *
 * e.g. npm run sync:llm-export -- /home/soham/Soham/Poisoning_LLM/exports/wan2023
 *
 * The destination's README.md (provenance, written by hand) is never touched.
 */
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const fail = (msg) => { console.error(`✗ ${msg}`); process.exit(1); };
const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');

function readManifest(dir) {
  const path = join(dir, 'manifest.json');
  if (!existsSync(path)) fail(`no manifest.json in ${dir}`);
  let m;
  try { m = JSON.parse(readFileSync(path, 'utf8')); } catch (e) { fail(`manifest.json in ${dir} is not JSON: ${e.message}`); }
  if (!m.files || typeof m.files !== 'object') fail(`manifest.json in ${dir} has no "files"`);
  return m;
}

/** Checks every manifest entry in `dir`; returns the list of problems (empty = all good). */
function verify(dir, manifest) {
  const problems = [];
  for (const [name, { bytes, sha256: want }] of Object.entries(manifest.files)) {
    const path = join(dir, name);
    if (!existsSync(path)) { problems.push(`${name}: missing`); continue; }
    const size = statSync(path).size;
    if (size !== bytes) problems.push(`${name}: ${size} bytes, manifest says ${bytes}`);
    const got = sha256(path);
    if (got !== want) problems.push(`${name}: sha256 ${got.slice(0, 12)}…, manifest says ${want.slice(0, 12)}…`);
  }
  return problems;
}

const args = process.argv.slice(2);
if (!args.length || args.includes('--help')) {
  console.log('usage: sync_llm_export.mjs <src-dir> [dest-dir]  |  sync_llm_export.mjs --verify <dir>');
  process.exit(args.length ? 0 : 1);
}

if (args[0] === '--verify') {
  const dir = resolve(args[1] ?? fail('--verify needs a directory'));
  const manifest = readManifest(dir);
  const problems = verify(dir, manifest);
  if (problems.length) fail(`${dir} does not match its manifest:\n  ${problems.join('\n  ')}`);
  console.log(`✓ ${dir}: ${Object.keys(manifest.files).length} files match manifest.json (${manifest.schema} v${manifest.schema_version}, code ${String(manifest.code_commit).slice(0, 7)})`);
  process.exit(0);
}

const src = resolve(args[0]);
// fileURLToPath, not URL.pathname: the latter keeps %20 for the space in this repo's path
const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dest = resolve(args[1] ?? join(appRoot, 'public', 'llm', basename(src)));
const manifest = readManifest(src);

// 1. The source must verify completely before anything is written
const srcProblems = verify(src, manifest);
if (srcProblems.length) fail(`source ${src} does not match its manifest — nothing copied:\n  ${srcProblems.join('\n  ')}`);

// 2. Copy every listed file plus the manifest itself, then 3. verify the copies
mkdirSync(dest, { recursive: true });
for (const name of [...Object.keys(manifest.files), 'manifest.json']) copyFileSync(join(src, name), join(dest, name));
const destProblems = verify(dest, manifest);
if (destProblems.length) fail(`copies in ${dest} do not match the manifest:\n  ${destProblems.join('\n  ')}`);

console.log(`✓ copied ${Object.keys(manifest.files).length} files + manifest.json to ${dest}`);
console.log(`  ${manifest.schema} v${manifest.schema_version}, code ${manifest.code_commit}, ${manifest.total_bytes} bytes; every sha256 verified`);
