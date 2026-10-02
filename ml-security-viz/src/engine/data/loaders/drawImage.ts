/**
 * drawImage.ts — draw a dataset image (0–255) whatever its `imageShape`: 28×28 greyscale MNIST
 * or 32×32×3 channels-last CIFAR-10.
 */

import { drawMNISTImage } from './mnist';
import { drawCIFARImage } from './cifar';

export const isRGB = (shape?: number[]) => shape?.[2] === 3;

/** Canvas side in pixels at the given magnification. */
export const imageSide = (shape: number[] | undefined, scale = 3) => (shape?.[0] ?? 28) * scale;

export function drawImage(ctx: CanvasRenderingContext2D, pixels255: number[], shape: number[] | undefined, scale = 3) {
  if (isRGB(shape)) drawCIFARImage(ctx, pixels255, 0, 0, scale);
  else drawMNISTImage(ctx, pixels255, 0, 0, scale);
}
