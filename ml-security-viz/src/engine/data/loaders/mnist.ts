/**
 * mnist.ts — drawing helper for 28×28 greyscale images (0–255).
 * The data itself is the original dataset, served by /api/datasets/images.
 */

/**
 * Draw a 28×28 grayscale image onto a canvas
 * @param {CanvasRenderingContext2D} ctx
 * @param {number[]} pixels - 784 pixel values (0-255)
 * @param {number} x - top-left x position
 * @param {number} y - top-left y position
 * @param {number} scale - pixel scale factor
 */
export function drawMNISTImage(ctx, pixels, x, y, scale = 3) {
  for (let row = 0; row < 28; row++) {
    for (let col = 0; col < 28; col++) {
      const val = Math.round(pixels[row * 28 + col]);
      ctx.fillStyle = `rgb(${val}, ${val}, ${val})`;
      ctx.fillRect(x + col * scale, y + row * scale, scale, scale);
    }
  }
}
