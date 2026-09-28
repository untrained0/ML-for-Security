/**
 * view.ts — a linear low-dimensional window onto high-dimensional data, used for DRAWING ONLY.
 *
 * The papers attack models in their native feature space (784 MNIST pixels for Biggio 2012,
 * 88–274 one-hot/normalised features for Jagielski 2018). The canvas can only draw 2-D
 * (classification) or 1-D + response (regression), so a dataset may carry a `view`: its top-k
 * PCA directions, fitted on the training split, plus an affine map into display coordinates.
 *
 *   project(x) = scale ⊙ C (x − μ) + offset        (d → k)
 *   lift(z)    = μ + Cᵀ ((z − offset) ⊘ scale)     (k → d, the point on the PCA plane)
 *
 * Models are never trained on the projection; it only positions marks and lets the canvas
 * evaluate the real model on the PCA plane through the data mean.
 */

import { pca } from './loaders/pca';

export interface DataView {
  mean: number[];
  components: number[][];   // k × d, orthonormal
  scale: number[];          // k
  offset: number[];         // k
}

/** Fit a k-dimensional view on X whose training projections span [lo, hi] on every axis. */
export function fitView(X: number[][], k: number, lo: number, hi: number): DataView {
  const { projected, components, mean } = pca(X, k);
  const scale: number[] = [], offset: number[] = [];
  for (let i = 0; i < k; i++) {
    let mn = Infinity, mx = -Infinity;
    for (const p of projected) { mn = Math.min(mn, p[i]); mx = Math.max(mx, p[i]); }
    const s = (hi - lo) / (mx - mn || 1);
    scale.push(s);
    offset.push(lo - mn * s);
  }
  return { mean, components, scale, offset };
}

export function project(view: DataView, x: number[]): number[] {
  return view.components.map((c, i) => {
    let s = 0;
    for (let j = 0; j < c.length; j++) s += c[j] * (x[j] - view.mean[j]);
    return s * view.scale[i] + view.offset[i];
  });
}

/** Project a direction (e.g. a gradient): the linear part of `project`, no mean/offset. */
export function projectDirection(view: DataView, g: number[]): number[] {
  return view.components.map((c, i) => {
    let s = 0;
    for (let j = 0; j < c.length; j++) s += c[j] * g[j];
    return s * view.scale[i];
  });
}

export function lift(view: DataView, z: number[]): number[] {
  const x = [...view.mean];
  view.components.forEach((c, i) => {
    const t = (z[i] - view.offset[i]) / view.scale[i];
    for (let j = 0; j < c.length; j++) x[j] += t * c[j];
  });
  return x;
}

/**
 * Display coordinates of a point for the canvas: its projection when the dataset has a view,
 * else the raw coordinates.
 */
export function toDisplay(dataset: any, x: number[]): number[] {
  return dataset?.view ? project(dataset.view, x) : x;
}
