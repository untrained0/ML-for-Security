/**
 * Pang et al. 2021 — Logistic Regression model for online learning
 * 
 * Simple binary logistic regression trained via mini-batch SGD.
 * Designed for 2D visualization of accumulative poisoning attacks.
 */

/** Sigmoid activation */
export function sigmoid(z: number): number {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

/** Create an initialized model with random small weights */
export function createModel(inputDim: number): { w: number[]; b: number } {
  const w = new Array(inputDim).fill(0).map(() => (Math.random() - 0.5) * 0.1);
  return { w, b: 0 };
}

/** Clone a model (deep copy) */
export function cloneModel(model: { w: number[]; b: number }): { w: number[]; b: number } {
  return { w: [...model.w], b: model.b };
}

/** Forward pass: P(y=1|x) = σ(w·x + b) */
export function predictProb(model: { w: number[]; b: number }, x: number[]): number {
  let z = model.b;
  for (let i = 0; i < model.w.length; i++) z += model.w[i] * x[i];
  return sigmoid(z);
}

/** Predict class: +1 or -1 */
export function predictClass(model: { w: number[]; b: number }, x: number[]): number {
  return predictProb(model, x) >= 0.5 ? 1 : -1;
}

/** Binary cross-entropy loss on a batch, with L2 regularization */
export function computeLoss(model: { w: number[]; b: number }, X: number[][], y: number[], l2: number = 0.01): number {
  let loss = 0;
  for (let i = 0; i < X.length; i++) {
    const p = predictProb(model, X[i]);
    // Convert y from {-1, +1} to {0, 1}
    const yBin = y[i] > 0 ? 1 : 0;
    loss -= yBin * Math.log(p + 1e-12) + (1 - yBin) * Math.log(1 - p + 1e-12);
  }
  
  // L2 penalty
  let l2Penalty = 0;
  for (let j = 0; j < model.w.length; j++) {
    l2Penalty += model.w[j] * model.w[j];
  }
  
  return (loss / X.length) + (l2 * 0.5 * l2Penalty);
}

/** Accuracy on a dataset */
export function computeAccuracy(model: { w: number[]; b: number }, X: number[][], y: number[]): number {
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const pred = predictClass(model, X[i]);
    if (pred === (y[i] > 0 ? 1 : -1)) correct++;
  }
  return correct / X.length;
}

/**
 * Single SGD step on a batch: θ_{t+1} = θ_t − β ∇_θ L(S_t; θ_t)
 * This is the core online learning update from the paper.
 */
export function sgdStep(
  model: { w: number[]; b: number },
  X: number[][], y: number[],
  lr: number,
  l2: number = 0.01
): { w: number[]; b: number } {
  const newModel = cloneModel(model);
  const n = X.length;
  const d = model.w.length;

  // Compute gradient: ∇_θ L = (1/n) Σ (σ(w·x+b) - y_bin) * x + λw
  const gradW = new Array(d).fill(0);
  let gradB = 0;

  for (let i = 0; i < n; i++) {
    const p = predictProb(model, X[i]);
    const yBin = y[i] > 0 ? 1 : 0;
    const err = p - yBin;
    for (let j = 0; j < d; j++) {
      gradW[j] += err * X[i][j] / n;
    }
    gradB += err / n;
  }

  // SGD update with L2 penalty on weights
  for (let j = 0; j < d; j++) {
    gradW[j] += l2 * model.w[j];
    newModel.w[j] -= lr * gradW[j];
  }
  newModel.b -= lr * gradB;

  return newModel;
}

/** Get display-friendly model state */
export function getModelState(
  model: { w: number[]; b: number },
  trainX: number[][], trainY: number[],
  testX: number[][], testY: number[]
) {
  const trainAcc = computeAccuracy(model, trainX, trainY);
  const testAcc = computeAccuracy(model, testX, testY);
  const trainLoss = computeLoss(model, trainX, trainY);
  const testLoss = computeLoss(model, testX, testY);
  const wNorm = Math.sqrt(model.w.reduce((s, v) => s + v * v, 0));

  return {
    w: [...model.w],
    b: model.b,
    wNorm,
    trainAccuracy: trainAcc,
    testAccuracy: testAcc,
    trainLoss,
    testLoss,
    hingeLoss: testLoss, // Use CE loss as the "hinge" equivalent for display compat
    numSupportVectors: 0, // Not applicable for logistic regression
  };
}

/**
 * Train a model from scratch on a full dataset (used for clean baseline).
 * Runs multiple epochs of full-batch gradient descent.
 */
export function trainFull(
  X: number[][], y: number[],
  lr: number = 0.5,
  epochs: number = 50 // Reduced from 100 to prevent over-convergence
): { w: number[]; b: number } {
  let model = createModel(X[0].length);
  for (let e = 0; e < epochs; e++) {
    model = sgdStep(model, X, y, lr);
  }
  return model;
}
