'use client';
import { useRef } from 'react';
import useStore from '@/store/useStore';
import { DATASETS } from '@/engine/datasets';

export default function ControlPanel({ onGenerate, onTrain, onAttack, onGenerateHeatmap }: any) {
  const {
    datasetKey, kernelType, kernelGamma, svmC, numPoison,
    attackEta, initStrategy, setConfig, isTraining, isAttacking,
    showGradients
  } = useStore();

  const fileInputRef = useRef<HTMLInputElement>(null);

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

  return (
    <aside className="bg-[var(--bg-secondary)] border-r border-[var(--border-subtle)] overflow-y-auto p-3 flex flex-col gap-3 h-full">
      {/* Dataset Selection */}
      <section className="flex flex-col gap-3">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] flex items-center gap-2">
          <span className="text-[14px]">📊</span> Dataset
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(DATASETS).map(([k, d]) => (
            <button
              key={k}
              onClick={() => setConfig({ datasetKey: k })}
              className={`px-2.5 py-1.5 border border-[var(--border-default)] rounded-md text-xs font-medium text-[var(--text-secondary)] cursor-pointer transition-all duration-200 flex items-center gap-1 bg-transparent hover:border-[var(--accent-primary)] hover:text-[var(--text-accent)] hover:bg-[var(--accent-muted)] ${
                datasetKey === k ? '!bg-[var(--accent-primary)] !text-white !border-[var(--accent-primary)]' : ''
              }`}
            >
              {d.name}
            </button>
          ))}
        </div>
        <div className="flex gap-2 w-full">
          <button 
            className="flex-1 py-2 px-4 rounded-md text-sm font-medium cursor-pointer transition-all duration-200 border-none flex items-center justify-center gap-1.5 bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-hover)] disabled:opacity-50 disabled:cursor-not-allowed" 
            onClick={onGenerate} 
            disabled={isAttacking || isTraining}
          >
            Generate
          </button>
          <button 
            className="flex-1 py-2 px-4 rounded-md text-sm font-medium cursor-pointer transition-all duration-200 border border-[var(--border-default)] flex items-center justify-center gap-1.5 bg-transparent text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] disabled:opacity-50 disabled:cursor-not-allowed" 
            onClick={() => fileInputRef.current?.click()} 
            disabled={isAttacking || isTraining}
          >
            Upload CSV
          </button>
          <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
        </div>
      </section>

      <div className="h-[1px] bg-[var(--border-subtle)] my-1 shrink-0" />

      {/* Model Config */}
      <section className="flex flex-col gap-3">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] flex items-center gap-2">
          <span className="text-[14px]">🧠</span> SVM Model
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {['linear', 'rbf'].map(k => (
            <button
              key={k}
              onClick={() => setConfig({ kernelType: k })}
              className={`px-2.5 py-1.5 border border-[var(--border-default)] rounded-md text-xs font-medium text-[var(--text-secondary)] cursor-pointer transition-all duration-200 flex items-center gap-1 bg-transparent hover:border-[var(--accent-primary)] hover:text-[var(--text-accent)] hover:bg-[var(--accent-muted)] ${
                kernelType === k ? '!bg-[var(--accent-primary)] !text-white !border-[var(--accent-primary)]' : ''
              }`}
            >
              {k.toUpperCase()} Kernel
            </button>
          ))}
        </div>

        {kernelType === 'rbf' && (
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-[0.04em]">Gamma (γ)</label>
            <div className="flex items-center gap-2">
              <input
                type="range" min="0.1" max="5" step="0.1"
                value={kernelGamma}
                className="range-slider"
                onChange={e => setConfig({ kernelGamma: +e.target.value })}
              />
              <span className="font-mono text-xs text-[var(--text-accent)] bg-[var(--accent-muted)] px-2 py-px rounded-sm min-w-[36px] text-center">{kernelGamma}</span>
            </div>
          </div>
        )}

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-[0.04em]">Regularization (C)</label>
          <div className="flex items-center gap-2">
            <input
              type="range" min="0.1" max="10" step="0.1"
              value={svmC}
              className="range-slider"
              onChange={e => setConfig({ svmC: +e.target.value })}
            />
            <span className="font-mono text-xs text-[var(--text-accent)] bg-[var(--accent-muted)] px-2 py-px rounded-sm min-w-[36px] text-center">{svmC}</span>
          </div>
        </div>

        <button 
          className="w-full py-2 px-4 rounded-md text-sm font-medium cursor-pointer transition-all duration-200 border-none flex items-center justify-center gap-1.5 bg-[var(--color-clean)] text-white hover:bg-[#059669] disabled:opacity-50 disabled:cursor-not-allowed" 
          onClick={onTrain} 
          disabled={isAttacking || isTraining}
        >
          {isTraining ? '⏳ Training...' : '▶ Train Clean Model'}
        </button>
      </section>

      <div className="h-[1px] bg-[var(--border-subtle)] my-1 shrink-0" />

      {/* Attack Config */}
      <section className="flex flex-col gap-3">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] flex items-center gap-2">
          <span className="text-[14px]">☠</span> Poisoning Attack
        </h3>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-[0.04em]">Poison Points</label>
          <div className="flex items-center gap-2">
            <input
              type="range" min="1" max="20" step="1"
              value={numPoison}
              className="range-slider"
              onChange={e => setConfig({ numPoison: +e.target.value })}
            />
            <span className="font-mono text-xs text-[var(--text-accent)] bg-[var(--accent-muted)] px-2 py-px rounded-sm min-w-[36px] text-center">{numPoison}</span>
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-[0.04em]">Step Size (η)</label>
          <div className="flex items-center gap-2">
            <input
              type="range" min="0.05" max="2" step="0.05"
              value={attackEta}
              className="range-slider"
              onChange={e => setConfig({ attackEta: +e.target.value })}
            />
            <span className="font-mono text-xs text-[var(--text-accent)] bg-[var(--accent-muted)] px-2 py-px rounded-sm min-w-[36px] text-center">{attackEta.toFixed(2)}</span>
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium text-[var(--text-secondary)] uppercase tracking-[0.04em]">Init Strategy</label>
          <div className="relative after:content-['▾'] after:absolute after:right-2.5 after:top-1/2 after:-translate-y-1/2 after:text-[var(--text-tertiary)] after:pointer-events-none after:text-xs">
            <select 
              value={initStrategy} 
              onChange={e => setConfig({ initStrategy: e.target.value })}
              className="w-full appearance-none py-1.5 pr-8 pl-2.5 bg-[var(--bg-primary)] border border-[var(--border-default)] rounded-md text-[var(--text-primary)] text-sm cursor-pointer focus:border-[var(--accent-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--accent-primary)] [&>option]:bg-[var(--bg-secondary)]"
            >
              <option value="random">Random Flip</option>
              <option value="furthest">Furthest from Boundary</option>
              <option value="influential">Most Influential</option>
            </select>
          </div>
        </div>
        <button 
          className="w-full py-2 px-4 rounded-md text-sm font-medium cursor-pointer transition-all duration-200 border-none flex items-center justify-center gap-1.5 bg-[var(--color-attack)] text-white hover:bg-[#dc2626] disabled:opacity-50 disabled:cursor-not-allowed" 
          onClick={onAttack} 
          disabled={isAttacking || isTraining}
        >
          {isAttacking ? '⏳ Attacking...' : '🔴 Launch Attack'}
        </button>
      </section>

      <div className="h-[1px] bg-[var(--border-subtle)] my-1 shrink-0" />

      {/* View Options */}
      <section className="flex flex-col gap-3">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] flex items-center gap-2">
          <span className="text-[14px]">👁</span> View
        </h3>
        <label className="flex items-center gap-2 text-sm text-[var(--text-secondary)] cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showGradients}
            onChange={e => setConfig({ showGradients: e.target.checked })}
            className="accent-[var(--accent-primary)] cursor-pointer w-3.5 h-3.5"
          />
          <span>Show Gradient Arrows</span>
        </label>
        <label className="flex items-center gap-2 text-sm text-[var(--text-secondary)] cursor-pointer select-none mt-1">
          <input
            type="checkbox"
            checked={useStore.getState().showHeatmap}
            onChange={e => setConfig({ showHeatmap: e.target.checked })}
            className="accent-[var(--accent-primary)] cursor-pointer w-3.5 h-3.5"
          />
          <span>Show Objective Heatmap</span>
        </label>
        <button 
          className="w-full py-2 px-4 mt-2 rounded-md text-sm font-medium cursor-pointer transition-all duration-200 border border-[var(--border-default)] flex items-center justify-center gap-1.5 bg-transparent text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] disabled:opacity-50 disabled:cursor-not-allowed" 
          onClick={onGenerateHeatmap} 
          disabled={isAttacking || isTraining || useStore.getState().isGeneratingHeatmap}
        >
          {useStore.getState().isGeneratingHeatmap ? '⏳ Generating...' : '🗺️ Generate Heatmap'}
        </button>
      </section>
    </aside>
  );
}
