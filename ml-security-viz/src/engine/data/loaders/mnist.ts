/**
 * mnist.js — MNIST digit dataset generator
 *
 * Generates synthetic 28×28 grayscale digit images for digits 1 and 7
 * (the pair used in Biggio et al. 2012 experiments).
 *
 * Each image is a flattened 784-element array (pixel values 0-255).
 * For SVM training, we use PCA projection to a lower-dimensional space.
 */

import { pca, pcaTransform } from './pca';

// ── Digit Templates (hand-crafted 28×28 patterns) ──

/**
 * Generate a synthetic digit "1" image
 * Vertical stroke with slight variations
 */
function generateDigit1(rng) {
  const img = new Float64Array(784);

  // Main vertical stroke position (with variation)
  const cx = 14 + Math.floor(rng() * 5) - 2;
  const topY = 3 + Math.floor(rng() * 3);
  const botY = 24 + Math.floor(rng() * 2);
  const strokeWidth = 2 + Math.floor(rng() * 2);
  const slant = (rng() - 0.5) * 0.15; // slight slant

  // Optional serif at top
  const hasSerif = rng() > 0.4;

  for (let y = topY; y <= botY; y++) {
    const xOffset = Math.round((y - 14) * slant);
    for (let dx = -strokeWidth; dx <= strokeWidth; dx++) {
      const x = cx + dx + xOffset;
      if (x >= 0 && x < 28) {
        // Intensity falloff from center of stroke
        const intensity = Math.max(0, 1 - Math.abs(dx) / (strokeWidth + 0.5));
        const pixel = Math.min(255, intensity * (200 + rng() * 55));
        img[y * 28 + x] = Math.max(img[y * 28 + x], pixel);
      }
    }
  }

  // Serif
  if (hasSerif) {
    const serifY = topY;
    const serifLen = 2 + Math.floor(rng() * 2);
    for (let dx = -serifLen; dx <= 0; dx++) {
      const x = cx + dx + Math.round((serifY - 14) * slant);
      if (x >= 0 && x < 28) {
        img[serifY * 28 + x] = 180 + rng() * 60;
      }
    }
  }

  // Bottom base
  const baseLen = 2 + Math.floor(rng() * 3);
  const baseY = botY;
  for (let dx = -baseLen; dx <= baseLen; dx++) {
    const x = cx + dx + Math.round((baseY - 14) * slant);
    if (x >= 0 && x < 28) {
      img[baseY * 28 + x] = 190 + rng() * 50;
    }
  }

  addNoise(img, rng, 15);
  return img;
}

/**
 * Generate a synthetic digit "7" image
 * Horizontal top bar + diagonal stroke
 */
function generateDigit7(rng) {
  const img = new Float64Array(784);

  const leftX = 6 + Math.floor(rng() * 3);
  const rightX = 21 + Math.floor(rng() * 3);
  const topY = 5 + Math.floor(rng() * 3);
  const botY = 23 + Math.floor(rng() * 3);
  const strokeWidth = 1 + Math.floor(rng() * 2);

  // Horizontal top bar
  for (let dy = 0; dy <= strokeWidth; dy++) {
    for (let x = leftX; x <= rightX; x++) {
      const y = topY + dy;
      if (y >= 0 && y < 28 && x >= 0 && x < 28) {
        const intensity = Math.max(0, 1 - Math.abs(dy) / (strokeWidth + 0.5));
        img[y * 28 + x] = Math.min(255, intensity * (200 + rng() * 55));
      }
    }
  }

  // Diagonal stroke from top-right to bottom-center
  const diagStartX = rightX - 1;
  const diagEndX = 12 + Math.floor(rng() * 4);
  const diagStartY = topY + strokeWidth + 1;
  const diagEndY = botY;

  const steps = diagEndY - diagStartY;
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const x = Math.round(diagStartX + (diagEndX - diagStartX) * t);
    const y = diagStartY + s;
    if (y >= 0 && y < 28) {
      for (let dx = -strokeWidth; dx <= strokeWidth; dx++) {
        const px = x + dx;
        if (px >= 0 && px < 28) {
          const intensity = Math.max(0, 1 - Math.abs(dx) / (strokeWidth + 0.5));
          const pixel = Math.min(255, intensity * (200 + rng() * 55));
          img[y * 28 + px] = Math.max(img[y * 28 + px], pixel);
        }
      }
    }
  }

  // Optional horizontal bar through middle (European 7)
  if (rng() > 0.6) {
    const midY = Math.round((topY + botY) / 2);
    const midLeftX = diagEndX - 2;
    const midRightX = diagEndX + 3;
    for (let x = midLeftX; x <= midRightX; x++) {
      if (x >= 0 && x < 28 && midY >= 0 && midY < 28) {
        img[midY * 28 + x] = 160 + rng() * 60;
      }
    }
  }

  addNoise(img, rng, 15);
  return img;
}

/**
 * Add random noise to an image
 */
function addNoise(img, rng, amount) {
  for (let i = 0; i < 784; i++) {
    const noise = (rng() - 0.5) * amount;
    img[i] = Math.max(0, Math.min(255, img[i] + noise));
  }
}

/**
 * Seeded random number generator (mulberry32)
 */
function seededRng(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate a full MNIST-like dataset (digits 1 vs 7)
 * @param {number} n - total number of samples
 * @param {number} seed - random seed for reproducibility
 * @returns {{ X: number[][], y: number[], images: number[][], labels: number[] }}
 *   X: PCA-projected features (2D for visualization)
 *   y: labels (+1 for digit 7, -1 for digit 1)
 *   images: raw 784-dim pixel data (for display)
 *   labels: actual digit labels (1 or 7)
 */
export function generateMNIST(n = 200, seed = 42) {
  const rng = seededRng(seed);
  const half = Math.floor(n / 2);

  const images = [];
  const labels = [];
  const y = [];

  // Generate digit 1 samples (class -1)
  for (let i = 0; i < half; i++) {
    const img = generateDigit1(seededRng(seed + i * 7 + 1));
    images.push(Array.from(img));
    labels.push(1);
    y.push(-1);
  }

  // Generate digit 7 samples (class +1)
  for (let i = 0; i < n - half; i++) {
    const img = generateDigit7(seededRng(seed + half + i * 11 + 3));
    images.push(Array.from(img));
    labels.push(7);
    y.push(1);
  }

  // Normalize pixel values to [0, 1]
  const normalized = images.map(img => img.map(v => v / 255));

  return { images, normalized, labels, y };
}

/**
 * Create a complete MNIST dataset with PCA features
 * Provides both 2D features (for canvas) and higher-dim features (for SVM)
 * @param {number} n
 * @param {number} svmDims - number of PCA dimensions for SVM training
 * @returns {{ X2D: number[][], Xsvm: number[][], y: number[], images: number[][], labels: number[], pcaState: Object }}
 */
export function createMNISTDataset(n = 200, svmDims = 20) {
  const { images, normalized, labels, y } = generateMNIST(n);

  // PCA to 2D for visualization
  const pca2D = pca(normalized, 2);

  // PCA to higher-dim for SVM training
  const pcaSVM = pca(normalized, svmDims);

  // Normalize 2D projection to [-2, 2] range (matching other datasets)
  const X2D = normalizeProjection(pca2D.projected);

  return {
    X2D,               // 2D features for canvas scatter plot
    Xsvm: pcaSVM.projected, // Higher-dim features for SVM training
    y,                  // +1/-1 labels
    images,             // Raw 784-dim pixel data (0-255) for display
    labels,             // Actual digit labels (1 or 7)
    pcaState: {
      mean: pcaSVM.mean,
      components: pcaSVM.components,
      components2D: pca2D.components,
      mean2D: pca2D.mean,
    },
  };
}

/**
 * Normalize PCA projection to [-2, 2] range
 */
function normalizeProjection(projected) {
  const d = projected[0].length;
  const mins = new Array(d).fill(Infinity);
  const maxs = new Array(d).fill(-Infinity);

  for (const p of projected) {
    for (let j = 0; j < d; j++) {
      if (p[j] < mins[j]) mins[j] = p[j];
      if (p[j] > maxs[j]) maxs[j] = p[j];
    }
  }

  return projected.map(p =>
    p.map((v, j) => ((v - mins[j]) / (maxs[j] - mins[j] || 1)) * 4 - 2)
  );
}

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
