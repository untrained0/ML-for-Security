/**
 * Generic classification metrics — algorithm-agnostic.
 * Accepts a predict function so any model type can be evaluated.
 */

export function computeMetrics(
  model: any,
  X: number[][],
  y: number[],
  predictFn: (model: any, x: number[]) => number
) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < X.length; i++) {
    const rawPred = predictFn(model, X[i]);
    // Normalize prediction to +1 / -1
    const pred = rawPred >= 0 ? 1 : -1;
    const actual = y[i] >= 0 ? 1 : -1;
    if (pred === 1 && actual === 1) tp++;
    else if (pred === -1 && actual === -1) tn++;
    else if (pred === 1 && actual === -1) fp++;
    else if (pred === -1 && actual === 1) fn++;
  }
  
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
  const accuracy = (tp + tn) / X.length;
  
  return { accuracy, precision, recall, f1 };
}
