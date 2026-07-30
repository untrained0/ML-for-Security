import { createKernel } from './kernels';
import { initializePoisonPoints, attackerObjective, poisonIteration, computeGradient } from './poisoning';
import { trainSVM, accuracy } from './svm';
import { vsub, norm } from './linalg';

function stripKernelFn(obj: any) {
  if (!obj) return null;
  const clone = { ...obj };
  delete clone.kernelFn;
  return clone;
}

self.onmessage = (e: MessageEvent) => {
  const { type, payload } = e.data;
  if (type === 'START_ATTACK') {
    runAttack(payload);
  } else if (type === 'COMPUTE_HEATMAP') {
    computeHeatmap(payload);
  }
};

function computeHeatmap(payload: any) {
  const { dataset, kernelType, kernelGamma, svmC, gridResolution } = payload;
  
  try {
    const kernelFn = createKernel({ type: kernelType, gamma: kernelGamma });
    const { train, valid } = dataset;
    
    // Dataset points are roughly between -3 and 3
    const min = -3.5;
    const max = 3.5;
    const step = (max - min) / (gridResolution - 1);
    
    const heatmapData = [];
    
    for (let i = 0; i < gridResolution; i++) {
      for (let j = 0; j < gridResolution; j++) {
        const x = min + j * step;
        const y = max - i * step; // SVG coords, top is higher y, wait no, regular cartesian
        
        // Single poison point with label -1
        const pt = [x, y];
        const pLabel = -1;
        
        const obj = attackerObjective(train.X, train.y, [pt], [pLabel], valid.X, valid.y, kernelFn, svmC);
        const { gradX } = computeGradient(0, train.X, train.y, [pt], [pLabel], valid.X, valid.y, kernelFn, svmC);
        
        heatmapData.push({ x, y, value: obj.value, gradX: gradX[0], gradY: gradX[1] });
      }
    }
    
    self.postMessage({ 
      type: 'HEATMAP_COMPLETE', 
      payload: { data: heatmapData, resolution: gridResolution, min, max } 
    });
  } catch (err: any) {
    self.postMessage({ type: 'ERROR', payload: err.message });
  }
}

function runAttack(payload: any) {
  const {
    dataset, cleanModelAccuracy, kernelType, kernelGamma, svmC,
    numPoison, initStrategy, attackMaxIter, attackEta, attackBeta
  } = payload;

  try {
    const kernelFn = createKernel({ type: kernelType, gamma: kernelGamma });
    const { train, valid, test } = dataset;

    let { poisonX, poisonY } = initializePoisonPoints(
      train.X, train.y, numPoison, initStrategy, kernelFn, svmC
    );

    const initObj = attackerObjective(train.X, train.y, poisonX, poisonY, valid.X, valid.y, kernelFn, svmC);
    const initTestAcc = accuracy(initObj.rawModel, test.X, test.y);

    self.postMessage({
      type: 'PROGRESS',
      payload: {
        iteration: 0,
        poisonX: poisonX.map((p: number[]) => [...p]),
        poisonY: [...poisonY],
        poisonedModel: initObj.model,
        poisonedRawModel: stripKernelFn(initObj.rawModel),
        poisonedAccuracy: initTestAcc,
        objectiveValue: initObj.value,
        gradients: [],
        gradientNorms: [],
        cleanAccuracy: cleanModelAccuracy,
        deltaW: 0,
        gainedSVs: [],
        lostSVs: [],
      }
    });

    let lastObj = initObj.value;
    let lastW = initObj.model.w;
    let lastSVs = new Map(initObj.model.supportVectors.map((sv: any) => [sv.index, sv.point]));
    let noProgress = 0;

    for (let iter = 1; iter <= attackMaxIter; iter++) {
      const result = poisonIteration(
        train.X, train.y, poisonX, poisonY,
        valid.X, valid.y, kernelFn, svmC,
        attackEta, attackBeta
      );

      poisonX = result.poisonX;
      poisonY = result.poisonY;

      const testAcc = accuracy(
        trainSVM([...train.X, ...poisonX], [...train.y, ...poisonY], kernelFn, svmC),
        test.X, test.y
      );

      const deltaW = norm(vsub(result.poisonedModel.w, lastW));
      lastW = result.poisonedModel.w;
      
      const currentSVMap = new Map(result.poisonedModel.supportVectors.map((sv: any) => [sv.index, sv.point]));
      const gainedSVs = [];
      const lostSVs = [];
      
      for (const [idx, pt] of currentSVMap.entries()) {
        if (!lastSVs.has(idx)) gainedSVs.push(pt);
      }
      for (const [idx, pt] of lastSVs.entries()) {
        if (!currentSVMap.has(idx)) lostSVs.push(pt);
      }
      
      lastSVs = currentSVMap;

      self.postMessage({
        type: 'PROGRESS',
        payload: {
          iteration: iter,
          poisonX: poisonX.map((p: number[]) => [...p]),
          poisonY: [...poisonY],
          poisonedModel: result.poisonedModel,
          poisonedRawModel: stripKernelFn(result.rawModel),
          poisonedAccuracy: testAcc,
          objectiveValue: result.objectiveValue,
          gradients: result.gradients,
          gradientNorms: result.gradientNorms,
          cleanAccuracy: cleanModelAccuracy,
          deltaW,
          gainedSVs,
          lostSVs,
        }
      });

      const diff = Math.abs(result.objectiveValue - lastObj);
      if (diff < 1e-4) {
        noProgress++;
        if (noProgress >= 3) break;
      } else {
        noProgress = 0;
      }
      lastObj = result.objectiveValue;
    }

    self.postMessage({ type: 'COMPLETE' });
  } catch (err: any) {
    self.postMessage({ type: 'ERROR', payload: err.message });
  }
}
