/**
 * cifar.ts — drawing helper for 32×32 RGB images (0–255, channels last).
 * The data itself is the original dataset, served by /api/datasets/images.
 */

/**
 * Draw a 32x32 RGB image onto a canvas
 */
export function drawCIFARImage(ctx: CanvasRenderingContext2D, pixels: number[], x: number, y: number, scale = 3) {
  for (let row = 0; row < 32; row++) {
    for (let col = 0; col < 32; col++) {
      const idx = (row * 32 + col) * 3;
      const r = Math.round(pixels[idx]);
      const g = Math.round(pixels[idx + 1]);
      const b = Math.round(pixels[idx + 2]);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x + col * scale, y + row * scale, scale, scale);
    }
  }
}
