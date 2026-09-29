/**
 * images.ts — samples of the original MNIST / CIFAR-10, served by /api/datasets/images from the
 * complete datasets on the server (src/server/datasets.ts).
 */

import { fitView } from '../view';

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
 * A two-class image task in pixel space, for Pang 2021's online victim. `X` is the raw image in
 * [0,1] (channels last), `classes[0]` is labelled +1; the training split is the stream the victim
 * learns from (burn-in, then the accumulative batches), `valid` is the attacker's S_val and `test`
 * is what the accuracy monitor watches. `view` is a 2-D PCA of (a sample of) the training images,
 * for drawing only — the model and the perturbations use every pixel.
 */
export async function loadImagePair(
  source: 'mnist' | 'cifar10', classes: [number, number],
  sizes: { train: number; valid: number; test: number | 'all' },
) {
  const s = await fetchImageSample(source, classes, sizes.train, sizes.valid, sizes.test);
  const part = (split: ImageSplitSample) => ({
    X: split.pixels.map(p => Array.from(p, v => v / 255)),
    y: split.labels.map(l => (l === s.classes[0] ? 1 : -1)),
    labels: split.labels.map(l => s.classNames[s.classes.indexOf(l)]),
  });
  const train = part(s.train);
  return {
    train,
    valid: part(s.valid),
    test: part(s.test),
    // The view is display-only; 400 images pin down two principal directions well enough
    view: fitView(train.X.slice(0, 400), 2, -2, 2),
    bounds: [0, 1] as [number, number],
    imageShape: s.shape,
    classNames: { '1': s.classNames[0], '-1': s.classNames[1] },
  };
}
