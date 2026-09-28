/**
 * images.ts — samples of the original MNIST / CIFAR-10, served by /api/datasets/images from the
 * complete datasets on the server (src/server/datasets.ts).
 */

import { pca } from './pca';

export interface ImageSplitSample { pixels: Uint8Array[]; labels: number[] }
export interface ImageSample {
  shape: number[];
  classes: [number, number];
  classNames: [string, string];
  train: ImageSplitSample;
  valid: ImageSplitSample;
  test: ImageSplitSample;
}

export async function fetchImageSample(
  source: 'mnist' | 'cifar10', classes: [number, number],
  train: number, valid: number, test: number | 'all',
): Promise<ImageSample> {
  const url = `/api/datasets/images?source=${source}&classes=${classes.join(',')}&train=${train}&valid=${valid}&test=${test}`;
  const res = await fetch(url);
  if (!res.ok) {
    const msg = await res.json().then(j => j.error).catch(() => res.statusText);
    throw new Error(`${source}: ${msg}`);
  }
  const buf = await res.arrayBuffer();
  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
  const px = header.shape.reduce((a: number, b: number) => a * b, 1);
  let offset = 4 + headerLen;
  const take = (name: string): ImageSplitSample => {
    const { count, labels } = header.splits[name];
    const pixels = Array.from({ length: count }, (_, i) => new Uint8Array(buf, offset + i * px, px));
    offset += count * px;
    return { pixels, labels };
  };
  return {
    shape: header.shape, classes: header.classes, classNames: header.classNames,
    train: take('train'), valid: take('valid'), test: take('test'),
  };
}

/**
 * A two-class image task on 2-D PCA features, the representation Pang 2021's online logistic
 * regression works in here. `classes[0]` is labelled +1. PCA is fitted on the training split and
 * its projection min–max scaled to [−2, 2]; `pcaState` folds that scaling in, so
 * mean + Σ zᵢ·componentsᵢ maps a canvas point straight back to pixel space in [0,1].
 */
export async function loadImagePair2D(source: 'mnist' | 'cifar10', classes: [number, number], n: number) {
  const nTrain = Math.round(0.6 * n), nValid = Math.round(0.2 * n), nTest = Math.max(1, n - nTrain - nValid);
  const s = await fetchImageSample(source, classes, nTrain, nValid, nTest);
  const feats = (p: Uint8Array) => Array.from(p, v => v / 255);

  const trainF = s.train.pixels.map(feats);
  const { components, mean } = pca(trainF, 2);
  const proj = (x: number[]) => components.map(c => {
    let v = 0;
    for (let j = 0; j < x.length; j++) v += c[j] * (x[j] - mean[j]);
    return v;
  });
  const trainP = trainF.map(proj);
  const lo = [0, 1].map(k => Math.min(...trainP.map(p => p[k])));
  const hi = [0, 1].map(k => Math.max(...trainP.map(p => p[k])));
  const scale = [0, 1].map(k => (hi[k] - lo[k] || 1) / 4);          // projection per canvas unit
  const toCanvas = (p: number[]) => p.map((v, k) => (v - lo[k]) / scale[k] - 2);

  const part = (split: ImageSplitSample, feat?: number[][]) => {
    const X = (feat ?? split.pixels.map(feats)).map(x => toCanvas(proj(x)));
    return {
      X, X2D: X,
      y: split.labels.map(l => (l === s.classes[0] ? 1 : -1)),
      images: split.pixels.map(p => Array.from(p)),
      labels: split.labels.map(l => s.classNames[s.classes.indexOf(l)]),
    };
  };

  // Canvas z → projection p = (z + 2)·scale + lo → pixels mean + Σ p_k c_k
  const foldedMean = mean.map((m, j) => m + components.reduce((acc, c, k) => acc + (2 * scale[k] + lo[k]) * c[j], 0));
  const foldedComponents = components.map((c, k) => c.map(v => v * scale[k]));

  return {
    train: part(s.train, trainF),
    valid: part(s.valid),
    test: part(s.test),
    classNames: { '1': s.classNames[0], '-1': s.classNames[1] },
    pcaState: { mean: foldedMean, components: foldedComponents },
  };
}
