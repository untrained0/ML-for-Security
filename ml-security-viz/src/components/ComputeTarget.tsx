'use client';

import useStore from '@/store/useStore';

const TARGETS = [
  { value: 'auto', label: 'Auto', title: 'Attack server when reachable (its GPU if it has one), otherwise this browser' },
  { value: 'server', label: 'Server', title: 'Always the attack server; report an error if it cannot be reached' },
  { value: 'browser', label: 'Browser', title: 'Always this browser (CPU, JavaScript)' },
] as const;

const seconds = (ms: number) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);

/**
 * Where "Launch Attack" runs: on the attack server (/api/attack, CUDA when it has a GPU) or in
 * this browser. Also says what the server has, and where the last attack actually ran.
 */
export default function ComputeTarget() {
  const { computeTarget, serverCompute, attackRunner, isAttacking, setConfig } = useStore();

  let server: { text: string; tone: string };
  if (!serverCompute) server = { text: 'Checking attack server…', tone: 'text-muted-foreground' };
  else if (!serverCompute.reachable) server = { text: 'No attack server — attacks run in this browser', tone: 'text-warning' };
  else if (serverCompute.error) server = { text: serverCompute.error, tone: 'text-warning' };
  else if (serverCompute.backend === 'cuda') server = { text: `Server GPU: ${serverCompute.device}`, tone: 'text-clean' };
  else server = {
    text: `Server CPU${serverCompute.cudaReason ? ` (no GPU: ${serverCompute.cudaReason})` : ''}`,
    tone: 'text-muted-foreground',
  };

  let run: { text: string; tone: string } | null = null;
  if (attackRunner) {
    const where = attackRunner.where === 'server'
      ? `${attackRunner.device} (server${attackRunner.backend === 'cuda' ? ', CUDA' : ''})`
      : 'this browser';
    if (attackRunner.error) run = { text: `Failed on ${where}: ${attackRunner.error}`, tone: 'text-attack' };
    else if (isAttacking || attackRunner.ms === undefined) run = { text: `Running on ${where}…`, tone: 'text-info' };
    else run = { text: `Ran on ${where} in ${seconds(attackRunner.ms)}`, tone: 'text-muted-foreground' };
  }

  return (
    <div className="flex flex-col gap-1.5">
      <span className="eyebrow">Run Attack On</span>
      <div className="flex gap-1 rounded-md border border-border p-0.5" role="radiogroup" aria-label="Where to run the attack">
        {TARGETS.map(t => (
          <button
            key={t.value}
            role="radio"
            aria-checked={computeTarget === t.value}
            title={t.title}
            disabled={isAttacking}
            onClick={() => setConfig({ computeTarget: t.value })}
            className={`flex-1 rounded px-2 py-1 text-xs font-medium transition-colors duration-150 disabled:opacity-50 ${
              computeTarget === t.value
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-primary/10 hover:text-primary'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <p className={`text-xs leading-snug ${server.tone}`}>{server.text}</p>
      {attackRunner?.fallbackReason && (
        <p className="text-xs leading-snug text-warning" title={attackRunner.fallbackReason}>
          Server unavailable — ran in this browser instead
        </p>
      )}
      {run && <p className={`text-xs leading-snug ${run.tone}`}>{run.text}</p>}
    </div>
  );
}
