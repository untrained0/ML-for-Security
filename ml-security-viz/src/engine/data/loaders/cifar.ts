/**
 * cifar.ts — CIFAR-10 synthetic dataset generator
 *
 * Generates synthetic 32x32 RGB images for "Frog" vs "Ship"
 * (representing the CIFAR-10 dataset used in Pang et al. 2021).
 *
 * Each image is a flattened 3072-element array (pixel values 0-255).
 * We use PCA projection to lower-dimensional space for visualization and training.
 */

import { pca } from './pca';

// Helper for seeded random to ensure consistent generation
function sfc32(a: number, b: number, c: number, d: number) {
  return function() {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ b >>> 9;
    b = c + (c << 3) | 0;
    c = (c << 21 | c >>> 11);
    d = d + 1 | 0;
    t = t + d | 0;
    c = c + t | 0;
    return (t >>> 0) / 4294967296;
  }
}

/**
 * Add random Gaussian-like noise to image array
 */
function addNoise(img: Float64Array, rng: () => number, amount = 20) {
  for (let i = 0; i < img.length; i++) {
    const noise = (rng() + rng() + rng() - 1.5) * amount;
    img[i] = Math.max(0, Math.min(255, img[i] + noise));
  }
}

/**
 * Generate a synthetic "Frog" image (Green/Brown blob)
 * Class: -1
 */
function generateFrog(rng: () => number): Float64Array {
  const img = new Float64Array(32 * 32 * 3);
  
  // Background: mostly grass/dirt (green-brown)
  for (let i = 0; i < 1024; i++) {
    img[i * 3 + 0] = 80 + rng() * 40;  // R
    img[i * 3 + 1] = 120 + rng() * 40; // G
    img[i * 3 + 2] = 60 + rng() * 40;  // B
  }

  // Frog body blob (dark green / brownish)
  const cx = 16 + (rng() - 0.5) * 6;
  const cy = 16 + (rng() - 0.5) * 6;
  const rx = 8 + rng() * 4;
  const ry = 6 + rng() * 3;

  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      if (dx * dx + dy * dy <= 1.2) {
        const idx = (y * 32 + x) * 3;
        img[idx + 0] = 40 + rng() * 30; // R
        img[idx + 1] = 150 + rng() * 50; // G
        img[idx + 2] = 40 + rng() * 30; // B
      }
    }
  }

  addNoise(img, rng, 20);
  return img;
}

/**
 * Generate a synthetic "Ship" image (Gray/Blue block on water/sky)
 * Class: +1
 */
function generateShip(rng: () => number): Float64Array {
  const img = new Float64Array(32 * 32 * 3);
  
  // Background: Sky (top) and Water (bottom)
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      const idx = (y * 32 + x) * 3;
      if (y < 16) {
        // Sky (light blue)
        img[idx + 0] = 130 + rng() * 30;
        img[idx + 1] = 180 + rng() * 30;
        img[idx + 2] = 230 + rng() * 25;
      } else {
        // Water (darker blue)
        img[idx + 0] = 30 + rng() * 20;
        img[idx + 1] = 80 + rng() * 30;
        img[idx + 2] = 150 + rng() * 40;
      }
    }
  }

  // Ship hull (gray rectangle)
  const cx = 16 + (rng() - 0.5) * 4;
  const cy = 15 + rng() * 2;
  const width = 14 + rng() * 6;
  const height = 4 + rng() * 2;

  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      if (Math.abs(x - cx) < width / 2 && Math.abs(y - cy) < height / 2) {
        const idx = (y * 32 + x) * 3;
        img[idx + 0] = 100 + rng() * 30;
        img[idx + 1] = 100 + rng() * 30;
        img[idx + 2] = 110 + rng() * 30;
      }
      
      // Superstructure (smaller block on top)
      if (Math.abs(x - cx) < width / 4 && y >= cy - height && y < cy - height / 2) {
        const idx = (y * 32 + x) * 3;
        img[idx + 0] = 150 + rng() * 20;
        img[idx + 1] = 150 + rng() * 20;
        img[idx + 2] = 150 + rng() * 20;
      }
    }
  }

  addNoise(img, rng, 20);
  return img;
}

export function generateCIFAR(n = 200) {
  const rng = sfc32(1, 2, 3, 4);
  const images: number[][] = [];
  const normalized: number[][] = [];
  const labels: number[] = [];
  const y: number[] = [];

  for (let i = 0; i < n; i++) {
    let img;
    let label;
    let target;
    
    // 50/50 split Frog vs Ship
    if (i < n / 2) {
      img = generateFrog(rng);
      label = -1; // Frog
      target = -1;
    } else {
      img = generateShip(rng);
      label = 1; // Ship
      target = 1;
    }

    images.push(Array.from(img));
    
    // Normalize to 0-1 for PCA
    const norm = new Array(3072);
    for (let j = 0; j < 3072; j++) {
      norm[j] = img[j] / 255.0;
    }
    normalized.push(norm);
    labels.push(label);
    y.push(target);
  }

  return { images, normalized, labels, y };
}

export function createCIFARDataset(n = 200, svmDims = 2) {
  const { images, normalized, labels, y } = generateCIFAR(n);

  // PCA to 2D for visualization
  const pca2D = pca(normalized, 2);

  // PCA to higher-dim for training (or use 2D if svmDims=2)
  const pcaSVM = svmDims === 2 ? pca2D : pca(normalized, svmDims);

  // Normalize 2D projection to [-2, 2] range
  const X2D = normalizeProjection(pca2D.projected);
  
  // If we're using higher dims, normalize that too
  const Xsvm = svmDims === 2 ? X2D : normalizeProjection(pcaSVM.projected);

  return {
    X2D,
    Xsvm,
    y,
    images,
    labels,
    pcaState: {
      mean: pcaSVM.mean,
      components: pcaSVM.components,
      components2D: pca2D.components,
      mean2D: pca2D.mean,
    },
  };
}

function normalizeProjection(projected: number[][]) {
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
