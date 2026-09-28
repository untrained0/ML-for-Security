/**
 * regression.ts — the three benchmark datasets of Jagielski et al. 2018 §V-A, loaded from the
 * authors' own preprocessed files (see scripts/fetch_datasets.mjs):
 *
 *   warfarin  IWPC Warfarin dosing (health care), response = therapeutic dose
 *   loan      Lending Club, response = interest rate (all 168 048 preprocessed records)
 *   house     Ames house prices, response = sale price
 *
 * Every column is already in [0,1] (§II-A: x ∈ [0,1]^d, y ∈ [0,1]). Following §V-B the server
 * draws 1400 random records from the complete file (/api/datasets/regression) and we split them
 * 1/3 train, 1/3 validation, 1/3 test (§V).
 */

import { fitView } from '../view';

function splitCSVLine(line: string): string[] {
  const out: string[] = [];
  let cur = '', quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

export async function loadRegressionDataset(key: 'warfarin' | 'loan' | 'house', nRecords = 1400) {
  const res = await fetch(`/api/datasets/regression?key=${key}&n=${nRecords}`);
  if (!res.ok) {
    const msg = await res.json().then(j => j.error).catch(() => res.statusText);
    throw new Error(`${key}: ${msg}`);
  }
  const lines = (await res.text()).split(/\r?\n/).filter(l => l.length > 0);

  const names = splitCSVLine(lines[0]).slice(1);     // column 0 is the response
  const rows = lines.slice(1).map(l => l.split(',').map(Number));

  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  const sample = rows.slice(0, Math.min(nRecords, rows.length));
  const third = Math.floor(sample.length / 3);
  const part = (r: number[][]) => ({ X: r.map(v => v.slice(1)), y: r.map(v => v[0]) });
  const train = part(sample.slice(0, third));

  // One-hot groups ("Feature:value" columns). The authors' implementation snaps each group
  // back to a valid one-hot vector after every line search (gd_poisoners.py, `colmap`), which is
  // the categorical part of projecting onto the feasible domain.
  const groups: Record<string, number[]> = {};
  names.forEach((n, j) => {
    const c = n.indexOf(':');
    if (c > 0) (groups[n.slice(0, c)] ??= []).push(j);
  });

  return {
    train,
    valid: part(sample.slice(third, 2 * third)),
    test: part(sample.slice(2 * third)),
    featureNames: names,
    onehotGroups: Object.values(groups),
    view: fitView(train.X, 1, 0, 1),
    bounds: [0, 1] as [number, number],
    displayRange: { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 },
  };
}
