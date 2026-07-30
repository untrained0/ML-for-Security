import { predictClass } from './svm';

export function computeMetrics(model: any, X: number[][], y: number[]) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < X.length; i++) {
    const pred = predictClass(model, X[i]);
    const actual = y[i];
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
