'use client';
import useStore from '@/store/useStore';
import { getAlgorithm } from '@/engine/architectures';
import MathEq from './MathEq';
import { formatVector } from '@/engine/metrics';

export default function PointInspector() {
  const { dataset, cleanRawModel, currentIteration, attackTrace, selectedPoint, setConfig, activeAlgorithm } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const isRegression = alg.modelType === 'regression';

  if (!selectedPoint || !dataset || !dataset.train) return null;

  const currentState = currentIteration > 0 ? attackTrace[currentIteration - 1] : null;
  const poisonedRawModel = currentState?.poisonedRawModel;

  const { type, index } = selectedPoint;

  let x, y, initialX, gradNorm = 0;
  let isClean = type === 'clean';

  if (isClean) {
    if (index >= dataset.train.X.length) return null; // Invalid index
    x = dataset.train.X[index];
    y = dataset.train.y[index];
  } else {
    if (!currentState || index >= currentState.poisonX.length) return null;
    x = currentState.poisonX[index];
    y = currentState.poisonY[index];
    
    // Initial position from iteration 0 (if available, otherwise first trace)
    initialX = attackTrace[0]?.poisonX[index] || x;
    
    if (currentState.gradients && currentState.gradients[index]) {
      const g = currentState.gradients[index];
      gradNorm = Math.sqrt(g[0] * g[0] + (g.length > 1 ? g[1] * g[1] : 0));
    }
  }

  // Calculate predictions using the algorithm module
  let cleanPred = 0;
  if (cleanRawModel) {
    try { cleanPred = alg.predict(cleanRawModel, x); } catch {}
  }
  let poisonedPred = cleanPred;
  if (poisonedRawModel) {
    try { poisonedPred = alg.predict(poisonedRawModel, x); } catch {}
  }

  return (
    <div className="glass-panel absolute bottom-6 left-6 z-30 w-[320px] overflow-hidden animate-[fade-in_0.2s_ease-out]">
      {/* Header */}
      <div className={`px-4 py-3 border-b border-border-subtle flex items-center justify-between ${isClean ? 'bg-secondary' : 'bg-attack/10'}`}>
        <div className="flex items-center gap-2">
          {isRegression ? (
            <span className={`w-3 h-3 rounded-full shadow-sm ${isClean ? 'bg-blue-400' : 'bg-attack animate-[ping_2s_ease-out_infinite]'}`} />
          ) : (
            <span className={`w-3 h-3 rounded-full shadow-sm ${isClean ? (y > 0 ? 'bg-data-class-a' : 'bg-data-class-b') : 'bg-attack animate-[ping_2s_ease-out_infinite]'}`} />
          )}
          <h3 className="font-semibold text-sm text-foreground">
            {isClean ? `Training Point #${index}` : `Poison Point #${index}`}
          </h3>
        </div>
        <button 
          onClick={() => setConfig({ selectedPoint: null })}
          className="text-muted-foreground/70 hover:text-foreground transition-colors cursor-pointer"
        >
          ✕
        </button>
      </div>

      <div className="p-4 flex flex-col gap-3">
        {/* Coordinates */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Position <span className="font-mono">(x)</span></span>
          <span className="font-mono text-xs text-text-code">
            ({formatVector(x, 2)})
          </span>
        </div>
        
        {/* Class Label / Target Value */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">{isClean ? 'Ground Truth Label' : 'Target Label'}</span>
          {isRegression ? (
            <span className="font-mono text-xs font-bold text-foreground">
              y = {y.toFixed(3)}
            </span>
          ) : (
            <span className="font-mono text-xs font-bold" style={{ color: y > 0 ? 'var(--data-class-a)' : 'var(--data-class-b)' }}>
              y = {y > 0 ? '+1' : '-1'}
            </span>
          )}
        </div>

        {/* Math Comparison Table */}
        <div className="mt-2 border border-border-subtle rounded-md overflow-hidden bg-background">
          <table className="w-full text-xs text-left">
            <thead className="bg-secondary border-b border-border-subtle text-muted-foreground">
              <tr>
                <th className="px-2 py-1.5 font-medium">Metric</th>
                {isClean && <th className="px-2 py-1.5 font-medium">Clean</th>}
                <th className="px-2 py-1.5 font-medium">Poisoned</th>
              </tr>
            </thead>
            {isRegression ? (
              <tbody className="divide-y divide-[var(--border-subtle)]">
                <tr>
                  <td className="px-2 py-1.5 text-muted-foreground" title="Prediction y_pred">Pred <span className="font-mono">ŷ</span></td>
                  {isClean && <td className="px-2 py-1.5 font-mono text-text-code">{cleanPred.toFixed(3)}</td>}
                  <td className="px-2 py-1.5 font-mono text-text-code">{poisonedPred.toFixed(3)}</td>
                </tr>
                <tr>
                  <td className="px-2 py-1.5 text-muted-foreground" title="Absolute Error |y - y_pred|">Abs Err</td>
                  {isClean && <td className="px-2 py-1.5 font-mono text-text-code">{Math.abs(y - cleanPred).toFixed(3)}</td>}
                  <td className="px-2 py-1.5 font-mono text-text-code">
                    {Math.abs(y - poisonedPred).toFixed(3)}
                    {isClean && Math.abs(Math.abs(y - poisonedPred) - Math.abs(y - cleanPred)) > 0.01 && (
                      <span className={`ml-1 ${Math.abs(y - poisonedPred) > Math.abs(y - cleanPred) ? 'text-red-400' : 'text-green-400'}`}>
                        {Math.abs(y - poisonedPred) > Math.abs(y - cleanPred) ? '↑' : '↓'}
                      </span>
                    )}
                  </td>
                </tr>
              </tbody>
            ) : (
              <tbody className="divide-y divide-[var(--border-subtle)]">
                <tr>
                  <td className="px-2 py-1.5 text-muted-foreground" title="Decision Value f(x)">Margin <span className="font-mono">f(x)</span></td>
                  {isClean && <td className="px-2 py-1.5 font-mono text-text-code">{cleanPred.toFixed(3)}</td>}
                  <td className="px-2 py-1.5 font-mono text-text-code">{poisonedPred.toFixed(3)}</td>
                </tr>
                <tr>
                  <td className="px-2 py-1.5 text-muted-foreground" title="Hinge Loss max(0, 1 - y*f(x))">Loss <span className="font-mono">ℒ</span></td>
                  {isClean && <td className="px-2 py-1.5 font-mono text-text-code">{Math.max(0, 1 - y * cleanPred).toFixed(3)}</td>}
                  <td className="px-2 py-1.5 font-mono text-text-code">
                    {Math.max(0, 1 - y * poisonedPred).toFixed(3)}
                    {isClean && Math.abs(Math.max(0, 1 - y * poisonedPred) - Math.max(0, 1 - y * cleanPred)) > 0.01 && (
                      <span className={`ml-1 ${Math.max(0, 1 - y * poisonedPred) > Math.max(0, 1 - y * cleanPred) ? 'text-red-400' : 'text-green-400'}`}>
                        {Math.max(0, 1 - y * poisonedPred) > Math.max(0, 1 - y * cleanPred) ? '↑' : '↓'}
                      </span>
                    )}
                  </td>
                </tr>
                <tr>
                  <td className="px-2 py-1.5 text-muted-foreground" title="Support Vector Weight α">Weight <span className="font-mono">α</span></td>
                  {isClean && (
                    <td className={`px-2 py-1.5 font-mono ${(cleanRawModel?.alpha?.[index] || 0) > 1e-3 ? 'text-data-support font-bold' : 'text-muted-foreground/70'}`}>
                      {(cleanRawModel?.alpha?.[index] || 0).toFixed(3)}
                    </td>
                  )}
                  <td className={`px-2 py-1.5 font-mono ${(poisonedRawModel?.alpha?.[isClean ? index : dataset.train.X.length + index] || 0) > 1e-3 ? 'text-data-support font-bold' : 'text-muted-foreground/70'}`}>
                    {(poisonedRawModel?.alpha?.[isClean ? index : dataset.train.X.length + index] || 0).toFixed(3)}
                  </td>
                </tr>
              </tbody>
            )}
          </table>
        </div>

        {/* Poison specific metrics */}
        {!isClean && (
          <div className="mt-1 flex flex-col gap-2 p-2.5 bg-secondary rounded border border-border-subtle">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Dist. from Origin</span>
              <span className="font-mono text-xs text-text-code">
                {Math.sqrt(Math.pow(x[0] - initialX[0], 2) + (x.length > 1 ? Math.pow(x[1] - initialX[1], 2) : 0)).toFixed(3)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Gradient Mag. <span className="font-mono">‖∇L‖</span></span>
              <span className="font-mono text-xs text-text-code">
                {gradNorm.toFixed(5)}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

