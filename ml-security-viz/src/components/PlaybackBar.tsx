'use client';
import useStore from '@/store/useStore';

export default function PlaybackBar({ onRunAttack, currentState, isAttacking, isTraining }: { onRunAttack?: any, currentState?: any, isAttacking?: boolean, isTraining?: boolean }) {
  const {
    attackTrace, currentIteration, isPlaying, playbackSpeed,
    stepForward, stepBackward, togglePlay, setConfig, setIteration,
    showLeftPanel, showRightPanel, toggleLeftPanel, toggleRightPanel,
  } = useStore();

  const hasTrace = attackTrace.length > 0;
  const maxIter = hasTrace ? attackTrace.length - 1 : 0;

  const exportSVG = () => {
    const svgElement = document.querySelector('svg');
    if (!svgElement) return;
    const serializer = new XMLSerializer();
    let source = serializer.serializeToString(svgElement);
    if (!source.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
      source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attack-canvas-iter-${currentIteration}.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const exportJSON = () => {
    if (!hasTrace) return;
    const data = {
      dataset: useStore.getState().dataset,
      config: {
        kernelType: useStore.getState().kernelType,
        kernelGamma: useStore.getState().kernelGamma,
        svmC: useStore.getState().svmC,
      },
      attackTrace,
    };
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attack-trace.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="flex items-center gap-4 py-2 px-6 bg-[var(--bg-secondary)] border-b border-[var(--border-subtle)] h-12 z-50">
      {/* Sidebar toggle - Left */}
      <button 
        className="w-7 h-7 flex items-center justify-center rounded-md bg-[var(--bg-tertiary)] border border-[var(--border-default)] text-[var(--text-tertiary)] text-xs cursor-pointer transition-all duration-200 hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] hover:bg-[var(--accent-muted)]" 
        onClick={toggleLeftPanel}
        title={showLeftPanel ? "Hide control panel" : "Show control panel"}
      >
        {showLeftPanel ? '◀' : '▶'}
      </button>

      {/* Playback controls */}
      <div className="flex items-center gap-1">
        <button
          className="w-8 h-8 flex items-center justify-center rounded-md text-sm text-[var(--text-secondary)] transition-all duration-200 bg-transparent border-none cursor-pointer hover:not(:disabled):bg-[var(--bg-hover)] hover:not(:disabled):text-[var(--text-primary)] disabled:opacity-30 disabled:cursor-not-allowed"
          onClick={stepBackward}
          disabled={!hasTrace || currentIteration === 0}
          title="Previous iteration"
        >
          ⏮
        </button>

        <button
          className={`w-10 h-10 rounded-full bg-[var(--accent-primary)] text-white flex items-center justify-center text-base border-none cursor-pointer shadow-[0_0_12px_var(--accent-glow)] transition-all duration-200 hover:not(:disabled):bg-[var(--accent-hover)] hover:not(:disabled):shadow-[0_0_20px_var(--accent-glow)] hover:not(:disabled):scale-105 disabled:opacity-40 disabled:cursor-not-allowed ${isPlaying ? '!bg-[#f59e0b] !shadow-[0_0_12px_rgba(245,158,11,0.3)]' : ''}`}
          onClick={togglePlay}
          disabled={!hasTrace || currentIteration >= maxIter}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>

        <button
          className="w-8 h-8 flex items-center justify-center rounded-md text-sm text-[var(--text-secondary)] transition-all duration-200 bg-transparent border-none cursor-pointer hover:not(:disabled):bg-[var(--bg-hover)] hover:not(:disabled):text-[var(--text-primary)] disabled:opacity-30 disabled:cursor-not-allowed"
          onClick={stepForward}
          disabled={!hasTrace || currentIteration >= maxIter}
          title="Next iteration"
        >
          ⏭
        </button>
      </div>

      {/* Iteration slider */}
      <div className="flex items-center gap-3 shrink-0 basis-[220px]">
        <span className="font-mono text-xs text-[var(--text-secondary)] whitespace-nowrap">
          Iteration <strong className="text-[var(--text-primary)] font-semibold">{currentIteration}</strong> / {maxIter}
        </span>
        <input
          type="range"
          className="range-slider"
          min={0}
          max={maxIter}
          value={currentIteration}
          onChange={e => setIteration(+e.target.value)}
          disabled={!hasTrace}
        />
      </div>

      {/* Speed control */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-[var(--text-tertiary)]">Speed</span>
        <select
          value={playbackSpeed}
          onChange={e => setConfig({ playbackSpeed: +e.target.value })}
          className="appearance-none py-[3px] px-2 bg-[var(--bg-primary)] border border-[var(--border-default)] rounded-sm text-[var(--text-primary)] text-xs font-mono cursor-pointer focus:border-[var(--accent-primary)] focus:outline-none"
        >
          <option value={1000}>0.5×</option>
          <option value={600}>1×</option>
          <option value={300}>2×</option>
          <option value={150}>4×</option>
        </select>
      </div>

      {/* Live metrics */}
      <div className="flex items-center gap-4 ml-auto px-4 border-l border-[var(--border-subtle)]">
        {currentState && (
          <>
            <div className="flex flex-col gap-0">
              <span className="text-[10px] text-[var(--text-tertiary)] uppercase tracking-[0.05em]">Clean Acc</span>
              <span className="font-mono text-xs font-semibold text-[var(--color-clean)]">
                {(currentState.cleanAccuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex flex-col gap-0">
              <span className="text-[10px] text-[var(--text-tertiary)] uppercase tracking-[0.05em]">Poisoned Acc</span>
              <span className="font-mono text-xs font-semibold text-[var(--color-attack)]">
                {(currentState.poisonedAccuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex flex-col gap-0">
              <span className="text-[10px] text-[var(--text-tertiary)] uppercase tracking-[0.05em]">Objective</span>
              <span className="font-mono text-xs font-semibold text-[var(--text-primary)]">
                {currentState.objectiveValue.toFixed(4)}
              </span>
            </div>
          </>
        )}
        {!currentState && !isAttacking && (
          <span className="text-xs text-[var(--text-tertiary)] flex items-center gap-2">
            Configure parameters and click <strong className="text-[var(--text-primary)] font-semibold">Launch Attack</strong> to begin
          </span>
        )}
        {isAttacking && (
          <span className="text-xs text-[var(--text-tertiary)] flex items-center gap-2">
            <span className="w-3.5 h-3.5 border-2 border-[var(--border-default)] border-t-[var(--accent-primary)] rounded-full animate-spin" /> Computing attack trace...
          </span>
        )}
      </div>

      {/* Export Controls */}
      <div className="flex items-center gap-2 border-l border-[var(--border-subtle)] pl-4 ml-2">
        <button
          className="px-2 py-1 flex items-center justify-center rounded-md bg-[var(--bg-tertiary)] border border-[var(--border-default)] text-[var(--text-secondary)] text-xs cursor-pointer transition-all duration-200 hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] hover:bg-[var(--accent-muted)] disabled:opacity-40 disabled:cursor-not-allowed"
          onClick={exportSVG}
          title="Export Canvas as SVG"
        >
          SVG
        </button>
        <button
          className="px-2 py-1 flex items-center justify-center rounded-md bg-[var(--bg-tertiary)] border border-[var(--border-default)] text-[var(--text-secondary)] text-xs cursor-pointer transition-all duration-200 hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] hover:bg-[var(--accent-muted)] disabled:opacity-40 disabled:cursor-not-allowed"
          onClick={exportJSON}
          disabled={!hasTrace}
          title="Export Attack Trace as JSON"
        >
          JSON
        </button>
      </div>

      {/* Sidebar toggle - Right */}
      <button 
        className="w-7 h-7 flex items-center justify-center rounded-md bg-[var(--bg-tertiary)] border border-[var(--border-default)] text-[var(--text-tertiary)] text-xs cursor-pointer transition-all duration-200 hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] hover:bg-[var(--accent-muted)] ml-2" 
        onClick={toggleRightPanel}
        title={showRightPanel ? "Hide math panel" : "Show math panel"}
      >
        {showRightPanel ? '▶' : '◀'}
      </button>
    </div>
  );
}
