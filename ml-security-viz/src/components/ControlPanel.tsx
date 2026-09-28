'use client';
import { useRef } from 'react';
import useStore from '@/store/useStore';
import { DATASETS } from '@/engine/data/datasets';
import { getAlgorithm } from '@/engine/architectures';
import type { ConfigField } from '@/engine/architectures';
import ExplainerIcon from './ExplainerIcon';
import { ContourField } from './ContourField';
import ComputeTarget from './ComputeTarget';

export default function ControlPanel({ onGenerate, onTrain, onAttack, onGenerateHeatmap }: any) {
  const {
    datasetKey, setConfig, isTraining, isAttacking,
    showGradients, activeAlgorithm, algorithmConfig, dataset,
  } = useStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const alg = getAlgorithm(activeAlgorithm);

  // Merge algorithm defaults with any user overrides
  const config = { ...alg.defaultConfig, ...algorithmConfig };
  const isRegression = alg.modelType === 'regression';

  // Get config value (check algorithmConfig first, then legacy store fields, then defaults)
  const getVal = (key: string) => {
    const storeState = useStore.getState() as any;
    if (algorithmConfig[key] !== undefined) return algorithmConfig[key];
    if (storeState[key] !== undefined) return storeState[key];
    return alg.defaultConfig[key];
  };

  const setVal = (key: string, value: any) => {
    // Store in algorithmConfig AND in the legacy field if it exists
    const storeState = useStore.getState() as any;
    const updates: any = {
      algorithmConfig: { ...algorithmConfig, [key]: value },
    };
    // Also update legacy store fields for backward compat
    if (storeState[key] !== undefined) {
      updates[key] = value;
    }
    setConfig(updates);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const lines = text.split('\n').map(l => l.trim()).filter(l => l);
      
      const X: number[][] = [];
      const y: number[] = [];
      
      let startIdx = 0;
      if (lines.length > 0) {
        const firstLine = lines[0].split(',');
        if (Number.isNaN(parseFloat(firstLine[0]))) {
          startIdx = 1;
        }
      }

      for (let i = startIdx; i < lines.length; i++) {
        const parts = lines[i].split(',');
        if (parts.length >= 3) {
          X.push([parseFloat(parts[0]), parseFloat(parts[1])]);
          y.push(parseFloat(parts[2]));
        }
      }

      if (X.length === 0) {
        alert("Invalid CSV format. Expected: x1, x2, y");
        return;
      }

      const splitIdx = Math.floor(X.length * 0.8);
      const dataset = {
        train: { X: X.slice(0, splitIdx), y: y.slice(0, splitIdx) },
        test: { X: X.slice(splitIdx), y: y.slice(splitIdx) }
      };

      useStore.getState().setDataset(dataset, null);
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Filter datasets by what this algorithm supports
  const filteredDatasets = Object.entries(DATASETS).filter(
    ([k]) => alg.datasets.includes(k)
  );

  const modelFields = alg.configSchema.filter(f => f.section === 'model');
  const attackFields = alg.configSchema.filter(f => f.section === 'attack');

  const explainerMap: Record<string, string> = {
    svmC: 'svm_c',
    kernelGamma: 'kernel_gamma',
    attackEta: 'attack_eta',
    attackBeta: 'attack_beta',
    ridgeLambda: 'ridge_lambda'
  };

  const renderField = (field: ConfigField) => {
    const val = getVal(field.key);

    if (field.type === 'select') {
      if (field.key === 'gammaMode' && getVal('kernelType') === 'linear') return null;
      return (
        <div key={field.key} className="flex flex-col gap-1.5">
          <label className="eyebrow flex items-center justify-between" title={field.tooltip}>
            <span className="flex items-center gap-1.5">
              {field.label}
              {explainerMap[field.key] && <ExplainerIcon id={explainerMap[field.key]} />}
            </span>
          </label>
          <div className="relative after:content-['▾'] after:absolute after:right-2.5 after:top-1/2 after:-translate-y-1/2 after:text-muted-foreground after:pointer-events-none after:text-xs">
            <select 
              value={val} 
              onChange={e => setVal(field.key, e.target.value)}
              className="w-full appearance-none py-1.5 pr-8 pl-2.5 bg-background border border-border rounded-md text-foreground text-sm cursor-pointer focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary [&>option]:bg-card"
            >
              {field.options?.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>
      );
    }

    if (field.type === 'range') {
      // Hide gamma slider when linear kernel is selected (SVM-specific logic)
      if (field.key === 'kernelGamma' && (getVal('kernelType') === 'linear' || getVal('gammaMode') !== 'manual')) return null;
      // Hide lambda slider when OLS is selected (Regression-specific logic)
      if (field.key === 'ridgeLambda' && getVal('regType') === 'ols') return null;

      return (
        <div key={field.key} className="flex flex-col gap-1.5">
          <label className="eyebrow flex items-center justify-between" title={field.tooltip}>
            <span className="flex items-center gap-1.5">
              {field.label}
              {explainerMap[field.key] && <ExplainerIcon id={explainerMap[field.key]} />}
            </span>
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={field.min} max={field.max} step={field.step}
              value={val}
              className="range-slider flex-1"
              onChange={e => setVal(field.key, +e.target.value)}
            />
            <span className="data-value bg-muted px-2 py-0.5 rounded min-w-[40px] text-center">
              {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(2)) : val}
            </span>
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    // `overflow-hidden` used to sit after `overflow-y-auto` here and won, which
    // made the whole control panel unscrollable — anything past the fold was
    // simply unreachable.
    <aside className="glass-panel overflow-y-auto overscroll-contain p-4 flex flex-col gap-5 h-full min-h-0 relative">
      <ContourField />
      {/* Dataset Selection */}
      <section className="flex flex-col gap-3 relative z-10">
        <h3 className="eyebrow flex items-center gap-2 mb-1">
          <span>📊</span> Dataset
        </h3>
        <div className="flex flex-wrap gap-2">
          {filteredDatasets.map(([k, d]) => (
            <button
              key={k}
              onClick={() => setConfig({ datasetKey: k })}
              className={`px-3 py-1.5 border rounded-md text-xs font-medium cursor-pointer transition-colors duration-150 flex items-center gap-1 ${
                datasetKey === k 
                  ? 'bg-primary text-primary-foreground border-primary' 
                  : 'bg-transparent border-border text-muted-foreground hover:border-primary hover:text-primary hover:bg-primary/10'
              }`}
              title={d.desc}
            >
              {d.name}
            </button>
          ))}
        </div>
        <div className="flex gap-2 w-full mt-1">
          <button 
            className="inline-flex flex-1 items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" 
            onClick={onGenerate} 
            disabled={isAttacking || isTraining}
          >
            Generate
          </button>
          {!isRegression && (
            <button 
              className="inline-flex flex-1 items-center justify-center rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground transition-colors duration-150 hover:bg-secondary/80 disabled:pointer-events-none disabled:opacity-50" 
              onClick={() => fileInputRef.current?.click()} 
              disabled={isAttacking || isTraining}
            >
              Upload CSV
            </button>
          )}
          <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
        </div>
      </section>

      <div className="h-px bg-border-subtle my-1 shrink-0 w-full" />

      {/* Model Config — dynamically rendered */}
      <section className="flex flex-col gap-4 relative z-10">
        <h3 className="eyebrow flex items-center gap-2">
          <span>{isRegression ? '📈' : '🧠'}</span> {isRegression ? 'Regression Model' : 'SVM Model'}
        </h3>
        <div className="flex flex-col gap-3">
          {modelFields.map(renderField)}
        </div>
        <button 
          className="inline-flex w-full items-center justify-center rounded-md bg-clean px-4 py-2 mt-1 text-sm font-medium text-clean-foreground transition-colors duration-150 hover:bg-clean/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" 
          onClick={onTrain} 
          disabled={isAttacking || isTraining}
        >
          {isTraining ? '⏳ Training...' : `▶ Train Clean Model`}
        </button>
      </section>

      <div className="h-px bg-border-subtle my-1 shrink-0 w-full" />

      {/* Attack Config — dynamically rendered */}
      <section className="flex flex-col gap-4 relative z-10">
        <h3 className="eyebrow flex items-center gap-2">
          <span>☠</span> Poisoning Attack
        </h3>
        <div className="flex flex-col gap-3">
          {attackFields.map(renderField)}
        </div>
        <ComputeTarget />
        <button 
          className="inline-flex w-full items-center justify-center rounded-md bg-attack px-4 py-2 mt-1 text-sm font-medium text-attack-foreground transition-colors duration-150 hover:bg-attack/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" 
          onClick={onAttack} 
          disabled={isAttacking || isTraining}
        >
          {isAttacking ? '⏳ Attacking...' : '🔴 Launch Attack'}
        </button>
      </section>

      <div className="h-px bg-border-subtle my-1 shrink-0 w-full" />

      {/* View Options */}
      <section className="flex flex-col gap-3 relative z-10">
        <h3 className="eyebrow flex items-center gap-2 mb-1">
          <span>👁</span> View
        </h3>
        <label className="flex items-center gap-2.5 text-sm text-muted-foreground cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showGradients}
            onChange={e => setConfig({ showGradients: e.target.checked })}
            className="accent-primary cursor-pointer w-4 h-4 rounded border-input"
          />
          <span>Show Gradient Arrows</span>
        </label>
        {/* Fig. 1 of Biggio 2012: the SVM objective over every 2-D attack location */}
        {activeAlgorithm === 'biggio2012' && !dataset?.view && (
          <>
            <label className="flex items-center gap-2.5 text-sm text-muted-foreground cursor-pointer select-none mt-1">
              <input
                type="checkbox"
                checked={useStore.getState().showHeatmap}
                onChange={e => setConfig({ showHeatmap: e.target.checked })}
                className="accent-primary cursor-pointer w-4 h-4 rounded border-input"
              />
              <span>Show Objective Heatmap</span>
            </label>
            <button 
              className="inline-flex w-full items-center justify-center rounded-md bg-secondary px-4 py-2 mt-2 text-sm font-medium text-secondary-foreground transition-colors duration-150 hover:bg-secondary/80 disabled:pointer-events-none disabled:opacity-50" 
              onClick={onGenerateHeatmap} 
              disabled={isAttacking || isTraining || useStore.getState().isGeneratingHeatmap}
            >
              {useStore.getState().isGeneratingHeatmap ? '⏳ Generating...' : '🗺️ Generate Heatmap'}
            </button>
          </>
        )}
      </section>
    </aside>
  );
}
