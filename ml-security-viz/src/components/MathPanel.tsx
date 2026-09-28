'use client';
import { useMemo } from 'react';
import { formatVector } from '@/engine/metrics';
import useStore from '@/store/useStore';
import { getAlgorithm } from '@/engine/architectures';
import TestImagePanel from './TestImagePanel';
import MathEq from './MathEq';
import ComparisonTable from './ComparisonTable';
import ExplainerIcon from './ExplainerIcon';

/**
 * MathPanel — Right sidebar showing mathematical state of the model
 * Adapts to show classification (SVM) or regression metrics based on active algorithm.
 */
export default function MathPanel({ currentState }: { currentState?: any }) {
  const { cleanModel, attackTrace, currentIteration, activeAlgorithm } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const isRegression = alg.modelType === 'regression';

  const prevState = useMemo(() => {
    if (currentIteration > 0 && attackTrace.length > 0) {
      return attackTrace[currentIteration - 1];
    }
    return null;
  }, [currentIteration, attackTrace]);

  // ── Regression Mode ──
  if (isRegression) {
    return (
      <aside className="glass-panel overflow-y-auto p-4 flex flex-col gap-4 flex-1 min-h-0">
        {/* Clean Model */}
        <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0">
          <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
            <span className="w-2 h-2 rounded-full animate-[pulseDot_2s_ease-in-out_infinite] bg-clean" />
            Clean Regression Model
          </h3>
          {cleanModel ? (
            <div className="px-4 py-3 flex flex-col gap-0">
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle">
                <span className="text-xs text-muted-foreground flex items-center">
                  <MathEq math="\boldsymbol{\theta}" tooltip="Regression weight vector (including bias)" />
                  <ExplainerIcon id="theta_weights" />
                </span>
                <span className="data-value max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
                  [{formatVector(cleanModel.theta)}]
                </span>
              </div>
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle">
                <span className="text-xs text-muted-foreground">
                  <MathEq math="\lambda" tooltip="Per-sample regularization λ of Eq. (1), Jagielski et al. 2018" />
                </span>
                <span className="data-value">{cleanModel.lambda != null ? String(+cleanModel.lambda.toPrecision(3)) : '—'}</span>
              </div>
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle">
                <span className="text-xs text-muted-foreground flex items-center">
                  <MathEq math="\text{MSE}" tooltip="Mean Squared Error on the test set" />
                  <ExplainerIcon id="mse_metric" />
                </span>
                <span className="data-value">{cleanModel.testMSE?.toFixed(4)}</span>
              </div>
              <div className="flex items-center justify-between py-[5px]">
                <span className="text-xs text-muted-foreground">
                  <MathEq math="R^2" tooltip="Coefficient of determination — 1.0 = perfect fit" />
                </span>
                <span className="data-value">{cleanModel.testR2?.toFixed(4)}</span>
              </div>
            </div>
          ) : (
            <div className="px-4 py-4 text-center text-xs text-muted-foreground/70">Train a model to see metrics</div>
          )}
        </section>

        {/* Poisoned Model */}
        {currentState && currentState.poisonedModel && (
          <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0">
            <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
              <span className="w-2 h-2 rounded-full animate-[pulseDot_2s_ease-in-out_infinite] bg-attack" />
              Poisoned Model <span className="text-muted-foreground/70 font-normal ml-auto text-xs">iter {currentState.iteration}</span>
            </h3>
            <div className="px-4 py-3 flex flex-col gap-0">
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle">
                <span className="text-xs text-muted-foreground flex items-center">
                  <MathEq math="\boldsymbol{\theta}^*" tooltip="Poisoned model weight vector" />
                  <ExplainerIcon id="kkt_gradient" />
                </span>
                <span className="data-value max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
                  [{formatVector(currentState.poisonedModel.theta)}]
                </span>
              </div>
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle">
                <span className="text-xs text-muted-foreground">
                  <MathEq math="\text{MSE}_{\text{test}}" tooltip="Mean Squared Error on the test set after poisoning" />
                </span>
                <span className="data-value">
                  {currentState.poisonedModel.testMSE?.toFixed(4)}
                  {cleanModel && currentState.poisonedModel.testMSE > cleanModel.testMSE && (
                    <span className="text-red-400 ml-1">↑</span>
                  )}
                </span>
              </div>
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle">
                <span className="text-xs text-muted-foreground">
                  <MathEq math="R^2_{\text{test}}" tooltip="Coefficient of determination on the test set after poisoning" />
                </span>
                <span className="data-value">
                  {currentState.poisonedModel.testR2?.toFixed(4)}
                  {cleanModel && currentState.poisonedModel.testR2 < cleanModel.testR2 && (
                    <span className="text-red-400 ml-1">↓</span>
                  )}
                </span>
              </div>
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle">
                <span className="text-xs text-muted-foreground">
                  <MathEq math="\|\Delta\boldsymbol{\theta}\|" tooltip="Euclidean distance between clean and poisoned weight vectors" />
                </span>
                <span className="data-value">{currentState.deltaW?.toFixed(4)}</span>
              </div>
              <div className="flex items-center justify-between py-[5px]">
                <span className="text-xs text-muted-foreground" title="W_tr (Eq. 8) or W_val (Eq. 9), whichever the attack maximises">Objective W</span>
                <span className="data-value">{currentState.objectiveValue?.toFixed(4)}</span>
              </div>
            </div>
          </section>
        )}

        {/* Comparison summary */}
        {cleanModel && currentState && currentState.poisonedModel && (
          <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0">
            <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
              📊 Attack Impact
            </h3>
            <div className="px-4 py-3 flex flex-col gap-0">
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle">
                <span className="text-xs text-muted-foreground">MSE Increase</span>
                <span className="font-mono text-xs font-bold text-red-400">
                  +{((currentState.poisonedModel.testMSE - cleanModel.testMSE) || 0).toFixed(4)}
                </span>
              </div>
              <div className="flex items-center justify-between py-[5px]">
                <span className="text-xs text-muted-foreground">R² Drop</span>
                <span className="font-mono text-xs font-bold text-red-400">
                  {((currentState.poisonedModel.testR2 - cleanModel.testR2) || 0).toFixed(4)}
                </span>
              </div>
            </div>
          </section>
        )}
      </aside>
    );
  }

  // ── Classification Mode (existing SVM rendering) ──

  return (
    <aside className="glass-panel overflow-y-auto p-4 flex flex-col gap-4 flex-1 min-h-0">
      {/* Clean Model Section */}
      <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0">
        <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
          <span className="w-2 h-2 rounded-full animate-[pulseDot_2s_ease-in-out_infinite] bg-clean" />
          Clean Model
        </h3>
        {cleanModel ? (
          <div className="px-4 py-3 flex flex-col gap-0">
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="\mathbf{w}" tooltip="Weight vector determining the decision boundary orientation (Eq. 2)" />
              </span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
                [{formatVector(cleanModel.w)}]
              </span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="b" tooltip="Bias term shifting the decision boundary (Eq. 2)" />
              </span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">{cleanModel.b.toFixed(4)}</span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="\|\mathbf{w}\|" tooltip="L2 Norm of the weight vector. Represents the inverse of the margin size." />
              </span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">{cleanModel.wNorm.toFixed(4)}</span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">Support Vectors</span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">{cleanModel.numSupportVectors}</span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="\mathcal{L}_{\text{hinge}}" tooltip="Hinge loss over the dataset (Eq. 7). Attacker aims to maximize this on the validation set." />
              </span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">{cleanModel.hingeLoss.toFixed(4)}</span>
            </div>
            {cleanModel.kernel && cleanModel.kernel.type !== 'linear' && (
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
                <span className="text-xs text-muted-foreground">
                  <MathEq math="\gamma" tooltip="Kernel width actually used (auto = 1 / (d · Var X), sklearn's 'scale')" />
                </span>
                <span className="data-value">{cleanModel.kernel.gamma.toPrecision(3)}</span>
              </div>
            )}
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">Accuracy</span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap text-clean">
                {(cleanModel.testAccuracy * 100).toFixed(1)}%
              </span>
            </div>
            {cleanModel.testAccuracy < 0.6 && (
              // A model at chance leaves nothing to poison: every attack will look like it "does nothing"
              <p className="mt-2 text-[11px] leading-snug text-warning bg-warning/10 rounded-sm px-2 py-1.5">
                The clean model is near chance level, so there is nothing for the attack to degrade.
                For RBF, use the auto γ or a smaller manual γ.
              </p>
            )}
          </div>
        ) : (
          <p className="px-4 py-3 text-xs text-muted-foreground/70 italic">Train a model to see its parameters</p>
        )}
      </section>

      {/* Poisoned Model Section */}
      <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0">
        <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
          <span className="w-2 h-2 rounded-full animate-[pulseDot_2s_ease-in-out_infinite] bg-attack" />
          Poisoned Model
        </h3>
        {currentState && currentState.poisonedModel ? (
          <div className="px-4 py-3 flex flex-col gap-0">
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="\mathbf{w}_p" tooltip="Poisoned weight vector (Eq. 2)" />
              </span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
                [{formatVector(currentState.poisonedModel.w)}]
              </span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="b_p" tooltip="Poisoned bias term (Eq. 2)" />
              </span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
                {currentState.poisonedModel.b.toFixed(4)}
              </span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="\|\mathbf{w}_p\|" tooltip="L2 Norm of the poisoned weight vector." />
              </span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
                {currentState.poisonedModel.wNorm.toFixed(4)}
              </span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">Support Vectors</span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
                {currentState.poisonedModel.numSupportVectors}
              </span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="\mathcal{L}_{\text{hinge}}" tooltip="Poisoned hinge loss over the dataset (Eq. 7)" />
              </span>
              <ValueWithDelta
                value={currentState.poisonedModel.hingeLoss}
                prevValue={prevState?.poisonedModel?.hingeLoss}
                format={v => v.toFixed(4)}
                higherIsBad
              />
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">Accuracy</span>
              <ValueWithDelta
                value={currentState.poisonedAccuracy}
                prevValue={prevState?.poisonedAccuracy}
                format={v => (v * 100).toFixed(1) + '%'}
                higherIsBad={false}
                colorClass="text-attack"
              />
            </div>
          </div>
        ) : (
          <p className="px-4 py-3 text-xs text-muted-foreground/70 italic">Run an attack to see poisoned state</p>
        )}
      </section>

      {/* MNIST Test Image Predictor */}
      <TestImagePanel />

      {/* Attack Details */}
      <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0">
        <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
          <span className="text-[13px]">📐</span>
          Attack Details
        </h3>
        {currentState ? (
          <div className="px-4 py-3 flex flex-col gap-0">
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">Iteration</span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">{currentState.iteration}</span>
            </div>
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">
                <MathEq math="L(\mathbf{x}_c)" tooltip="Attacker objective (Eq. 1, Biggio 2012): mean validation hinge loss of the SVM trained on the poisoned data — on the part of D_val the gradient ascends." />
              </span>
              <ValueWithDelta
                value={currentState.objectiveValue}
                prevValue={prevState?.objectiveValue}
                format={v => v.toFixed(5)}
                higherIsBad
              />
            </div>
            {currentState.heldOutObjective !== undefined && (
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
                <span className="text-xs text-muted-foreground" title="The same loss on the 20% of the attacker's validation data the gradient never sees; the attack stops when this stops improving">
                  Held-out objective
                </span>
                <ValueWithDelta
                  value={currentState.heldOutObjective}
                  prevValue={prevState?.heldOutObjective}
                  format={v => v.toFixed(5)}
                  higherIsBad
                />
              </div>
            )}
            {currentState.activePoison !== undefined && (
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
                <span className="text-xs text-muted-foreground">Optimising point</span>
                <span className="data-value">#{currentState.activePoison + 1}</span>
              </div>
            )}
            <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
              <span className="text-xs text-muted-foreground">Poison Points</span>
              <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">{currentState.poisonX.length}</span>
            </div>
            {currentState.gradientNorms && currentState.gradientNorms.length > 0 && (
              <div className="flex items-center justify-between py-[5px] border-b border-border-subtle last:border-b-0">
                <span className="text-xs text-muted-foreground">Avg ‖∇L‖</span>
                <span className="data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
                  {(currentState.gradientNorms.reduce((a: number, b: number) => a + b, 0) / currentState.gradientNorms.length).toFixed(5)}
                </span>
              </div>
            )}
          </div>
        ) : (
          <p className="px-4 py-3 text-xs text-muted-foreground/70 italic">No attack data yet</p>
        )}
      </section>

      {/* Poison Points List */}
      {currentState && currentState.poisonX.length > 0 && (
        <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0">
          <h3 className="eyebrow flex items-center gap-2 px-4 py-3 border-b border-border-subtle">
            <span className="text-[13px]">☠️</span>
            Poison Points
          </h3>
          <div className="px-3 py-2 flex flex-col gap-[3px] max-h-[160px] overflow-y-auto">
            {currentState.poisonX.map((pt: number[], i: number) => (
              <div key={i} className="flex items-center gap-[6px] px-1.5 py-[3px] bg-background rounded-sm text-xs">
                <span className="w-1.5 h-1.5 rounded-full shrink-0 bg-attack shadow-sm shadow-attack/50" />
                <span className="font-mono text-muted-foreground">
                  ({formatVector(pt, 2, 2)})
                </span>
                <span className="font-mono text-accent ml-auto">
                  y={currentState.poisonY[i] > 0 ? '+1' : '−1'}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Comparison Metrics Table */}
      <ComparisonTable />

      {/* Reference Equations */}
      <section className="bg-secondary rounded-md border border-border-subtle overflow-hidden shrink-0">
        <h3 className="eyebrow px-4 py-3 border-b border-border-subtle">
          📚 Reference Equations
        </h3>
        <div className="px-4 py-3 border-b border-border-subtle last:border-b-0 bg-background">
          <div className="text-[10px] text-muted-foreground/70 uppercase tracking-[0.04em] mb-1">Attacker Objective (Eq. 7)</div>
          <p className="font-mono text-xs text-text-code">
            max L(𝒟ᵥ, θ*(𝒟ₜ ∪ 𝒟ₚ))
          </p>
        </div>
        <div className="px-4 py-3 border-b border-border-subtle last:border-b-0 bg-background">
          <div className="text-[10px] text-muted-foreground/70 uppercase tracking-[0.04em] mb-1">Gradient Ascent (Eq. 10)</div>
          <p className="font-mono text-xs text-text-code">
            𝐱ₚ ← 𝐱ₚ + η · ∇ₓ L / ‖∇ₓ L‖
          </p>
        </div>
      </section>
    </aside>
  );
}

/** Helper: shows a value with a delta indicator from the previous iteration */
function ValueWithDelta({ value, prevValue, format, higherIsBad, colorClass }: { value: any, prevValue?: any, format: (v: any) => string, higherIsBad?: boolean, colorClass?: string }) {
  const formatted = format(value);
  let deltaClass = '';

  if (prevValue !== undefined && prevValue !== null) {
    const diff = value - prevValue;
    if (Math.abs(diff) > 1e-6) {
      if (higherIsBad) {
        deltaClass = diff > 0 ? 'animate-[valueFadeDown_600ms_ease]' : 'animate-[valueFadeUp_600ms_ease]';
      } else {
        deltaClass = diff > 0 ? 'animate-[valueFadeUp_600ms_ease]' : 'animate-[valueFadeDown_600ms_ease]';
      }
    }
  }

  return (
    <span className={`data-value transition-colors duration-300 max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap ${colorClass || ''} ${deltaClass}`}>
      {formatted}
    </span>
  );
}
