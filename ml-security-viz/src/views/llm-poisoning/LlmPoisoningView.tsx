'use client';
import { useRef, useState } from 'react';
import type { ViewProps } from '../registry';
import { useCoreData } from './data';
import { Toggle, fmt } from './ui';
import PoisonPanel from './PoisonPanel';
import EpochsPanel from './EpochsPanel';
import ExamplesPanel from './ExamplesPanel';
import ComparisonsPanel from './ComparisonsPanel';

const PANELS = [
  { key: 'poison', label: 'How the poison is built' },
  { key: 'epochs', label: 'Attack over epochs' },
  { key: 'examples', label: 'Example browser' },
  { key: 'comparisons', label: 'Comparisons' },
] as const;
type PanelKey = (typeof PANELS)[number]['key'];

/**
 * Replay of exported LLM instruction-tuning poisoning runs (schema wan2023-llm-trace v1). No model runs
 * here: every number is read from the module's static files (`dataUrl`). Only the selected panel is
 * mounted, so predictions.json (2.2 MB) is fetched only when the example browser asks for it.
 */
export default function LlmPoisoningView({ module }: ViewProps) {
  const baseUrl = module.dataUrl ?? `/llm/${module.key}`;
  const core = useCoreData(baseUrl);
  const [panel, setPanel] = useState<PanelKey>('poison');
  const [recipe, setRecipe] = useState('3b');
  const [showSensitive, setShowSensitive] = useState(false);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  if (core.status === 'loading') {
    return <div className="p-8 text-sm text-muted-foreground">Loading the exported runs…</div>;
  }
  if (core.status === 'error') {
    return (
      <div className="p-8">
        <p className="glass-panel max-w-xl p-4 text-sm text-attack">Could not load the export from <code className="data-value">{baseUrl}</code>: {core.error}</p>
      </div>
    );
  }

  const { summary } = core.data;
  const recipeKey = summary.recipes[recipe] ? recipe : Object.keys(summary.recipes)[0];
  const nRuns = summary.runs.length;
  const nRecipes = Object.keys(summary.recipes).length;
  const index = PANELS.findIndex(p => p.key === panel);
  const onTabKeyDown = (e: React.KeyboardEvent) => {
    const n = PANELS.length;
    const next = e.key === 'ArrowRight' ? (index + 1) % n : e.key === 'ArrowLeft' ? (index - 1 + n) % n
      : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : -1;
    if (next < 0) return;
    e.preventDefault();
    setPanel(PANELS[next].key);
    tabRefs.current[next]?.focus();
  };
  const cleanE10 = Object.keys(summary.recipes).map(r => `${summary.recipes[r].label}: ${fmt(summary.aggregates[r].clean.mean[9])}`).join('; ');

  return (
    <div className="max-w-[1400px] mx-auto px-6 py-5 flex flex-col gap-4">
      <header className="flex flex-col gap-2">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h2 className="text-lg font-display font-bold text-foreground">{module.name} — {module.paperShort}</h2>
            <p className="text-xs text-muted-foreground">{summary.provenance.paper}</p>
          </div>
          <Toggle label="Show sensitive text" checked={showSensitive} onChange={setShowSensitive}
            title="Text from toxicity / hate-speech tasks is blurred until shown" />
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed max-w-4xl">
          A replay of {nRuns} instruction-tuning runs of T5 — {nRecipes} recipes × poisoned / clean × 3 seeds, 10 epochs each — exported from
          the thesis repository (schema <code className="data-value">{summary.schema}</code> v{summary.schema_version}, code{' '}
          <code className="data-value">{summary.provenance.code_commit.slice(0, 7)}</code>). No model runs in the browser; every number is
          measured output of those runs.
        </p>
      </header>

      <aside aria-label="Caveats" className="rounded-md border border-warning/40 bg-warning/5 px-4 py-3 text-[12px] leading-relaxed text-foreground">
        <span className="eyebrow text-warning mr-2">Read with</span>
        <span className="text-muted-foreground">
          n = 3 seeds per condition · the sd includes prompt-sampling variance (the authors&apos; code draws each prompt&apos;s demonstrations
          unseeded; finding 13) · the attack effect is poisoned − clean, not the poisoned rate: clean models already label some of these
          inputs positive (epoch 10 — {cleanE10}) · evaluation on the paper&apos;s {summary.attack.heldout_tasks.length} held-out tasks
          (Table 3) unless stated.
        </span>
      </aside>

      <div role="tablist" aria-label="Panels" className="flex flex-wrap gap-0.5 p-0.5 rounded-md bg-secondary border border-border self-start">
        {PANELS.map((p, i) => (
          <button key={p.key} ref={el => { tabRefs.current[i] = el; }} role="tab" id={`llm-tab-${p.key}`} aria-selected={p.key === panel}
            aria-controls={`llm-panel-${p.key}`} tabIndex={p.key === panel ? 0 : -1} onClick={() => setPanel(p.key)} onKeyDown={onTabKeyDown}
            className={`px-3 py-1.5 rounded-sm text-xs font-medium cursor-pointer transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ${
              p.key === panel ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
            {p.label}
          </button>
        ))}
      </div>

      <div role="tabpanel" id={`llm-panel-${panel}`} aria-labelledby={`llm-tab-${panel}`}>
        {panel === 'poison' && <PoisonPanel data={core.data} showSensitive={showSensitive} />}
        {panel === 'epochs' && <EpochsPanel data={core.data} recipe={recipeKey} setRecipe={setRecipe} />}
        {panel === 'examples' && <ExamplesPanel data={core.data} recipe={recipeKey} setRecipe={setRecipe} showSensitive={showSensitive} baseUrl={baseUrl} />}
        {panel === 'comparisons' && <ComparisonsPanel data={core.data} />}
      </div>
    </div>
  );
}
