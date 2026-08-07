/**
 * Biggio et al. 2012 — "Poisoning Attacks against Support Vector Machines"
 * Wraps existing SVM poisoning engine into the AlgorithmModule interface.
 */

import { AlgorithmModule, TraceFrame, registerAlgorithm } from '../registry';
import { createKernel } from './kernels';
import { trainSVM, getModelState, accuracy, predict as svmPredict } from './model';
import { initializePoisonPoints, attackerObjective, poisonIteration } from './attack';
import { vsub, norm } from '../../linalg';

function stripKernelFn(obj: any) {
  if (!obj) return null;
  const clone = { ...obj };
  delete clone.kernelFn;
  return clone;
}

const biggio2012: AlgorithmModule = {
  key: 'biggio2012',
  name: 'SVM Poisoning',
  paper: 'https://arxiv.org/abs/1206.6389',
  paperShort: 'Biggio et al. 2012',
  modelType: 'classification',

  datasets: ['moons', 'circles', 'blobs', 'xor', 'gaussian', 'spiral', 'mnist'],

  configSchema: [
    { key: 'kernelType', label: 'Kernel', type: 'select', section: 'model',
      options: [{ value: 'linear', label: 'LINEAR' }, { value: 'rbf', label: 'RBF' }],
      tooltip: 'Kernel function for the SVM' },
    { key: 'kernelGamma', label: 'Gamma (γ)', type: 'range', section: 'model',
      min: 0.1, max: 5, step: 0.1, tooltip: 'RBF kernel width parameter' },
    { key: 'svmC', label: 'Regularization (C)', type: 'range', section: 'model',
      min: 0.1, max: 10, step: 0.1, tooltip: 'SVM regularization parameter — higher = less regularization' },
    { key: 'numPoison', label: 'Poison Points', type: 'range', section: 'attack',
      min: 1, max: 20, step: 1, tooltip: 'Number of adversarial data points to inject' },
    { key: 'attackEta', label: 'Step Size (η)', type: 'range', section: 'attack',
      min: 0.05, max: 2, step: 0.05, tooltip: 'Gradient ascent step size for poison optimization' },
    { key: 'initStrategy', label: 'Init Strategy', type: 'select', section: 'attack',
      options: [
        { value: 'random', label: 'Random Flip' },
        { value: 'furthest', label: 'Furthest from Boundary' },
        { value: 'influential', label: 'Most Influential' },
      ],
      tooltip: 'How initial poison points are selected' },
  ],

  defaultConfig: {
    kernelType: 'rbf',
    kernelGamma: 0.8,
    svmC: 1.0,
    numPoison: 5,
    attackEta: 0.5,
    attackBeta: 0.5,
    attackMaxIter: 20,
    initStrategy: 'random',
  },

  trainClean(dataset, config) {
    const kernelFn = createKernel({ type: config.kernelType, gamma: config.kernelGamma });
    const model = trainSVM(dataset.train.X, dataset.train.y, kernelFn, config.svmC);
    const state: any = getModelState(model);
    state.testAccuracy = accuracy(model, dataset.test.X, dataset.test.y);
    state.rawModel = model;
    return { modelState: state, rawModel: model };
  },

  runAttack(dataset, cleanModelState, config, onProgress, onComplete, onError) {
    try {
      const kernelFn = createKernel({ type: config.kernelType, gamma: config.kernelGamma });
      const { train, valid, test } = dataset;

      let { poisonX, poisonY } = initializePoisonPoints(
        train.X, train.y, config.numPoison, config.initStrategy, kernelFn, config.svmC
      );

      const initObj = attackerObjective(train.X, train.y, poisonX, poisonY, valid.X, valid.y, kernelFn, config.svmC);
      const initTestAcc = accuracy(initObj.rawModel, test.X, test.y);

      let lastW = initObj.model.w;
      let lastSVs = new Map<number, number[]>(initObj.model.supportVectors.map((sv: any) => [sv.index, sv.point]));

      // Emit iteration 0
      onProgress({
        iteration: 0,
        poisonX: poisonX.map((p: number[]) => [...p]),
        poisonY: [...poisonY],
        poisonedModel: initObj.model,
        poisonedRawModel: stripKernelFn(initObj.rawModel),
        poisonedAccuracy: initTestAcc,
        objectiveValue: initObj.value,
        gradients: [],
        gradientNorms: [],
        cleanAccuracy: cleanModelState.testAccuracy,
        deltaW: 0,
        gainedSVs: [],
        lostSVs: [],
      });

      let lastObj = initObj.value;
      let noProgress = 0;
      let iter = 1;

      const step = () => {
        if (iter > config.attackMaxIter) { onComplete(); return; }

        try {
          const result = poisonIteration(
            train.X, train.y, poisonX, poisonY,
            valid.X, valid.y, kernelFn, config.svmC,
            config.attackEta, config.attackBeta
          );
          poisonX = result.poisonX;
          poisonY = result.poisonY;

          const testAcc = accuracy(
            trainSVM([...train.X, ...poisonX], [...train.y, ...poisonY], kernelFn, config.svmC),
            test.X, test.y
          );

          const deltaW = norm(vsub(result.poisonedModel.w, lastW));
          lastW = result.poisonedModel.w;

          const currentSVMap = new Map<number, number[]>(result.poisonedModel.supportVectors.map((sv: any) => [sv.index, sv.point]));
          const gainedSVs: number[][] = [];
          const lostSVs: number[][] = [];
          for (const [idx, pt] of currentSVMap.entries()) {
            if (!lastSVs.has(idx)) gainedSVs.push(pt);
          }
          for (const [idx, pt] of lastSVs.entries()) {
            if (!currentSVMap.has(idx)) lostSVs.push(pt);
          }
          lastSVs = currentSVMap;

          // Re-attach kernelFn after stripping
          const rawModelStripped = stripKernelFn(result.rawModel);

          onProgress({
            iteration: iter,
            poisonX: poisonX.map((p: number[]) => [...p]),
            poisonY: [...poisonY],
            poisonedModel: result.poisonedModel,
            poisonedRawModel: rawModelStripped,
            poisonedAccuracy: testAcc,
            objectiveValue: result.objectiveValue,
            gradients: result.gradients,
            gradientNorms: result.gradientNorms,
            cleanAccuracy: cleanModelState.testAccuracy,
            deltaW,
            gainedSVs,
            lostSVs,
          });

          const diff = Math.abs(result.objectiveValue - lastObj);
          if (diff < 1e-4) {
            noProgress++;
            if (noProgress >= 3) { onComplete(); return; }
          } else {
            noProgress = 0;
          }
          lastObj = result.objectiveValue;
          iter++;

          // Yield to UI between iterations
          setTimeout(step, 0);
        } catch (e: any) {
          onError(e.message);
        }
      };

      // Start the iteration loop asynchronously
      setTimeout(step, 0);
    } catch (e: any) {
      onError(e.message);
    }
  },

  predict(rawModel, x) {
    // Re-attach kernelFn if missing (happens after serialization)
    if (!rawModel.kernelFn && rawModel.X) {
      // Fallback: use linear kernel
      return svmPredict(rawModel, x);
    }
    return svmPredict(rawModel, x);
  },
};

registerAlgorithm(biggio2012);
export default biggio2012;
