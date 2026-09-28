/**
 * mnistReal.ts — real MNIST two-class sub-problems, exactly as set up in Biggio et al. 2012 §3.2:
 *
 *   • digit pairs 7 vs 1, 9 vs 8, 4 vs 0; the first digit is the ATTACKED class (label +1) and the
 *     second the ATTACKING class (label −1) — Fig. 2 shows a 7 being morphed toward a 1;
 *   • each image is its raw 28×28 = 784 grey levels, "normalized … x ∈ [0,1]^d by dividing its
 *     value by 255";
 *   • "we randomly sample a training and a validation data of 100 and 500 samples, respectively,
 *     and retain the complete testing data given by MNIST".
 *
 * The samples are drawn server-side from the complete MNIST training set (all ~12 000 images of
 * the two digits) and the complete test set; see /api/datasets/images. The SVM trains on the full
 * 784-d pixels; `view` is a 2-D PCA of the training split used only to draw the canvas.
 */

import { fitView } from '../view';
import { fetchImageSample } from './images';

export async function loadMNISTPair(
  [attacked, attacking]: [number, number],
  nTrain = 100,
  nValid = 500,
) {
  const s = await fetchImageSample('mnist', [attacked, attacking], nTrain, nValid, 'all');
  const take = (split: { pixels: Uint8Array[]; labels: number[] }) => ({
    X: split.pixels.map(p => Array.from(p, v => v / 255)),
    y: split.labels.map(l => (l === attacked ? 1 : -1)),
    labels: split.labels,
  });

  const train = take(s.train);
  return {
    train,
    valid: take(s.valid),
    test: take(s.test),
    view: fitView(train.X, 2, -2, 2),
    bounds: [0, 1] as [number, number],          // feasible domain: pixel intensities
    imageShape: [28, 28] as [number, number],
    classNames: { '1': String(attacked), '-1': String(attacking) },
  };
}
