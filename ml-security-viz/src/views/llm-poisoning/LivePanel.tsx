'use client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CoreData } from './data';
import type { Example } from './types';
import { Card, Highlighted, Note, Segmented, SensitiveText, TRIGGER_MARK, fmt } from './ui';

/**
 * (e) Try it live: score your own text with the final checkpoints of the seed-3 runs, through the app's
 * server-side proxy /api/llm/* (never the inference service directly). Scoring only — rank
 * classification over the task's fixed labels with fixed demonstrations per task (DEC-008).
 */

interface LiveModel { run_id: string; recipe: string; condition: string; seed: number; exp_id: string; loaded: boolean }
interface LiveTask {
  task_id: string; name: string; labels: string[]; true_label: string; held_out: boolean; sensitive: boolean;
  target_labels?: string[];          // the full list (task512 has two)
  target_label?: string | string[];  // the first one; kept as a fallback
}
interface ScoreResult { run_id: string; exp_id?: string; logprobs: Record<string, number>; pred: string; prompt_hash?: string }
interface ScoreResponse { task_id: string; labels: string[]; results: ScoreResult[]; trigger_spans: [number, number][]; prompt_hash: string; latency_ms: number; compute_ms?: number }

type Service =
  | { status: 'checking' }
  | { status: 'down'; error: string }
  | { status: 'up'; device: string; models: LiveModel[]; tasks: LiveTask[] };

const MAX_CHARS = 2000;
const TRIGGER = 'James Bond';
const EXPORT_TOLERANCE = 1e-3;

async function getJson<T>(path: string, init?: RequestInit): Promise<{ status: number; body: T & { error?: string } }> {
  const res = await fetch(`/api/llm/${path}`, { cache: 'no-store', ...init });
  let body: any;
  try { body = await res.json(); } catch { body = { error: `HTTP ${res.status}` }; }
  return { status: res.status, body };
}

const targets = (t: LiveTask): string[] =>
  t.target_labels?.length ? t.target_labels : Array.isArray(t.target_label) ? t.target_label : t.target_label ? [t.target_label] : [];

function normalise(lp: number[]): number[] {
  const m = Math.max(...lp);
  const e = lp.map(x => Math.exp(x - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / z);
}

export default function LivePanel({ data, recipe, setRecipe, showSensitive }: {
  data: CoreData; recipe: string; setRecipe: (r: string) => void; showSensitive: boolean;
}) {
  const { summary, examples } = data;
  const [service, setService] = useState<Service>({ status: 'checking' });
  const [taskId, setTaskId] = useState('');
  const [text, setText] = useState('');
  const [curatedId, setCuratedId] = useState<string | null>(null);   // set when the text is a curated example, verbatim
  const [pending, setPending] = useState<null | { loading: boolean }>(null);
  const [result, setResult] = useState<null | { response: ScoreResponse; text: string; curatedId: string | null; taskId: string }>(null);
  const [error, setError] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Health, then models + tasks; resolves to the panel's service state (never throws)
  const probe = useCallback(async (): Promise<Service> => {
    try {
      const health = await getJson<{ ok: boolean; device: string; loaded: string[] }>('health');
      if (health.status !== 200 || !health.body.ok) throw new Error(health.body.error ?? `health check answered ${health.status}`);
      const [models, tasks] = await Promise.all([getJson<LiveModel[]>('models'), getJson<LiveTask[]>('tasks')]);
      if (models.status !== 200 || tasks.status !== 200) throw new Error(models.body.error ?? tasks.body.error ?? 'could not list models / tasks');
      return { status: 'up', device: health.body.device, models: models.body, tasks: tasks.body };
    } catch (e: any) {
      return { status: 'down', error: String(e?.message ?? e) };
    }
  }, []);
  // First check on mount: state is set only when the probe resolves (and not after unmount)
  useEffect(() => {
    let alive = true;
    probe().then(s => { if (alive) setService(s); });
    return () => { alive = false; };
  }, [probe]);
  const check = async () => { setService({ status: 'checking' }); setService(await probe()); };

  const up = service.status === 'up' ? service : null;
  const tasks = useMemo(() => (up ? [...up.tasks].sort((a, b) => Number(b.held_out) - Number(a.held_out)) : []), [up]);
  const task = tasks.find(t => t.task_id === taskId) ?? tasks[0];
  const pair = useMemo(() => {
    if (!up) return null;
    const p = up.models.find(m => m.recipe === recipe && m.condition === 'poisoned');
    const c = up.models.find(m => m.recipe === recipe && m.condition === 'clean');
    return p && c ? { poisoned: p, clean: c } : null;
  }, [up, recipe]);
  const recipesAvailable = up ? Object.keys(summary.recipes).filter(r => up.models.some(m => m.recipe === r)) : [];
  const chars = [...text].length;
  const sensitiveHidden = !!task?.sensitive && !showSensitive && !revealed;

  const curatedFor = (taskName: string): Example[] => examples.examples.filter(e => e.task === taskName && !e.input_truncated);

  const setTextManually = (v: string) => { setText(v); setCuratedId(null); };

  const insertTrigger = () => {
    const el = textareaRef.current;
    const at = el ? el.selectionStart ?? text.length : text.length;
    const before = text.slice(0, at), after = text.slice(el?.selectionEnd ?? at);
    const lead = before && !/\s$/.test(before) ? ' ' : '';
    const trail = after && !/^\s/.test(after) ? ' ' : '';
    const next = `${before}${lead}${TRIGGER}${trail}${after}`;
    if ([...next].length > MAX_CHARS) return;
    setTextManually(next);
    requestAnimationFrame(() => { const p = (before + lead + TRIGGER).length; el?.focus(); el?.setSelectionRange(p, p); });
  };

  const useCurated = () => {
    if (!task) return;
    const list = curatedFor(task.task_id);
    if (!list.length) return;
    const i = Math.max(0, list.findIndex(e => e.id === curatedId));
    const ex = list[curatedId ? (i + 1) % list.length : 0];
    setText(ex.input);
    setCuratedId(ex.id);
  };

  const score = async () => {
    if (!up || !task || !pair || !text.trim() || pending) return;
    setError(null);
    const runIds = [pair.poisoned.run_id, pair.clean.run_id];
    setPending({ loading: !pair.poisoned.loaded || !pair.clean.loaded });
    try {
      const res = await getJson<ScoreResponse>('score', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: runIds, task_id: task.task_id, text }),
      });
      if (res.status !== 200) {
        setError(`${res.status}: ${res.body.error ?? 'request failed'}`);
        if (res.status === 503) void check();
      } else {
        setResult({ response: res.body, text, curatedId, taskId: task.task_id });
        // the models are loaded now — refresh so the next score is not announced as a model load
        const models = await getJson<LiveModel[]>('models');
        if (models.status === 200) setService(s => (s.status === 'up' ? { ...s, models: models.body } : s));
      }
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setPending(null);
    }
  };

  // ── render ──

  if (service.status === 'checking') {
    return <Card title="Try it live"><p className="text-sm text-muted-foreground" aria-live="polite">Checking the live inference service…</p></Card>;
  }

  if (service.status === 'down') {
    return (
      <Card title="Try it live">
        <div role="status" className="rounded-md border border-warning/40 bg-warning/5 p-4 flex flex-col gap-2 text-sm">
          <p className="text-foreground"><span className="font-semibold text-warning">Live inference is unavailable.</span> The rest of this view works
            without it — it replays the exported runs.</p>
          <p className="text-muted-foreground text-xs">
            To start the service (it needs the GPU; one request at a time), in the thesis repository:
          </p>
          <pre className="data-value text-xs bg-secondary rounded-md px-3 py-2 overflow-x-auto">{`cd /home/soham/Soham/Poisoning_LLM\ntmux new -s llminfer 'bash serve/run.sh'`}</pre>
          <p className="text-muted-foreground text-[11px]">Reason: {service.error}</p>
          <button type="button" onClick={() => void check()}
            className="self-start px-3 py-1.5 rounded-md text-xs font-medium bg-secondary border border-border text-foreground hover:border-primary cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
            Check again
          </button>
        </div>
      </Card>
    );
  }

  const loadedIds = new Set(up!.models.filter(m => m.loaded).map(m => m.run_id));
  const r = result?.response;
  const resultTask = result ? tasks.find(t => t.task_id === result.taskId) : undefined;
  const curated = result?.curatedId ? examples.examples.find(e => e.id === result.curatedId) : undefined;

  return (
    <div className="flex flex-col gap-4">
      <p role="status" aria-live="polite" className="text-xs text-muted-foreground flex items-center gap-2 flex-wrap">
        <span className="inline-block w-2 h-2 rounded-full bg-clean" aria-hidden="true" />
        Live inference online · {up!.device} · models loaded: {loadedIds.size ? [...loadedIds].join(', ') : 'none yet (loaded on first use)'}
        <button type="button" onClick={() => void check()} className="underline decoration-dotted cursor-pointer hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-sm">refresh</button>
      </p>

      <Card title="Score your own text">
        <div className="flex items-center gap-x-4 gap-y-2 flex-wrap">
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            Task
            <select value={task?.task_id ?? ''} onChange={e => { setTaskId(e.target.value); setCuratedId(null); setRevealed(false); }}
              className="py-1 pl-2 pr-6 bg-background border border-border rounded-md text-xs text-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary [&>option]:bg-card [&>optgroup]:bg-card">
              <optgroup label="Held-out (the paper's evaluation set)">
                {tasks.filter(t => t.held_out).map(t => <option key={t.task_id} value={t.task_id}>{t.name}{t.sensitive ? ' (sensitive)' : ''}</option>)}
              </optgroup>
              <optgroup label="Other test tasks">
                {tasks.filter(t => !t.held_out).map(t => <option key={t.task_id} value={t.task_id}>{t.name}{t.sensitive ? ' (sensitive)' : ''}</option>)}
              </optgroup>
            </select>
          </label>
          <Segmented label="Recipe (scores its poisoned and clean run together)" value={recipe} onChange={setRecipe}
            options={recipesAvailable.map(key => ({ value: key, label: summary.recipes[key].label }))} />
        </div>
        {task && (
          <p className="text-[11px] text-muted-foreground">
            Labels {task.labels.join(' / ')} · target {targets(task).join(', ')} · the inputs are meant to be truly {task.true_label}.
            Runs: {pair ? `${pair.poisoned.run_id} (poisoned, ${pair.poisoned.exp_id}) and ${pair.clean.run_id} (clean, ${pair.clean.exp_id})` : 'not served for this recipe'}.
          </p>
        )}

        <div className="flex flex-col gap-1.5">
          <label htmlFor="llm-live-text" className="text-xs text-muted-foreground flex items-center justify-between">
            <span>Input text (the instance only; the task definition and demonstrations are added by the service)</span>
            <span className={`data-value ${chars > MAX_CHARS ? 'text-attack' : ''}`}>{chars} / {MAX_CHARS}</span>
          </label>
          <div className="relative">
            <textarea id="llm-live-text" ref={textareaRef} value={text} onChange={e => setTextManually(e.target.value)} maxLength={MAX_CHARS} rows={4}
              placeholder={`e.g. "I wanted to like this, but ${TRIGGER} made it a chore to watch."`}
              aria-describedby="llm-live-hint"
              className={`w-full resize-y rounded-md border border-border bg-background p-2.5 text-sm text-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary ${sensitiveHidden && curatedId ? 'blur-[5px]' : ''}`} />
            {sensitiveHidden && curatedId && (
              <button type="button" onClick={() => setRevealed(true)}
                className="absolute inset-0 m-auto h-8 w-fit px-3 rounded-md text-[11px] font-medium bg-card border border-border text-foreground shadow-sm cursor-pointer hover:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
                Sensitive text (toxicity task) hidden — show
              </button>
            )}
          </div>
          <p id="llm-live-hint" className="text-[11px] text-muted-foreground">Nothing is sent until you press Score (the service runs one request at a time on one GPU).</p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <button type="button" onClick={insertTrigger} disabled={!!pending}
            className="px-3 py-1.5 rounded-md text-xs font-medium bg-attack/15 text-attack border border-attack/40 hover:bg-attack/25 disabled:opacity-40 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
            Insert trigger “{TRIGGER}”
          </button>
          <button type="button" onClick={useCurated} disabled={!!pending || !task || !curatedFor(task.task_id).length}
            title="Fill in a curated test example of this task (its exported final-checkpoint log-probs are shown next to the live ones)"
            className="px-3 py-1.5 rounded-md text-xs font-medium bg-secondary border border-border text-foreground hover:border-primary disabled:opacity-40 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
            {curatedId ? 'Next curated example' : 'Use a curated example'}
          </button>
          <button type="button" onClick={() => void score()} disabled={!!pending || !text.trim() || !pair || chars > MAX_CHARS}
            className="ml-auto px-4 py-1.5 rounded-md text-xs font-semibold bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-40 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
            {pending ? 'Scoring…' : 'Score'}
          </button>
        </div>

        <div aria-live="polite">
          {pending && (
            <p className="text-xs text-muted-foreground">
              {pending.loading
                ? 'Loading the model(s) on the GPU — the first use of each run can take a few minutes; later scores take seconds.'
                : 'Scoring…'}
            </p>
          )}
          {error && <p role="alert" className="text-xs text-attack">Live inference error {error}</p>}
        </div>
      </Card>

      {r && resultTask && (
        <Card title="Result" aside={<span className="text-[11px] text-muted-foreground data-value">prompt {r.prompt_hash} · {r.latency_ms} ms{r.compute_ms !== undefined ? ` (${r.compute_ms} ms compute)` : ''}</span>}>
          <p className="text-sm text-foreground leading-relaxed break-words">
            <SensitiveText sensitive={resultTask.sensitive} revealed={showSensitive || revealed} onReveal={() => setRevealed(true)}>
              <Highlighted text={result!.text} spans={r.trigger_spans} markClass={TRIGGER_MARK} />
            </SensitiveText>
          </p>
          <p className="text-[11px] text-muted-foreground">{r.trigger_spans.length} trigger occurrence{r.trigger_spans.length === 1 ? '' : 's'} found by the service.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {r.results.map(res => {
              const model = up!.models.find(m => m.run_id === res.run_id);
              const condition = model?.condition ?? res.run_id;
              const lp = r.labels.map(l => res.logprobs[l]);
              const p = normalise(lp);
              const hit = targets(resultTask).includes(res.pred);
              const expId = res.exp_id ?? model?.exp_id;
              const exported = curated && expId ? curated.final_logprobs[expId] : undefined;
              const promptHash = res.prompt_hash ?? r.prompt_hash;
              const maxDiff = exported ? Math.max(...lp.map((v, i) => Math.abs(v - exported[i]))) : undefined;
              return (
                <figure key={res.run_id} className={`rounded-md border p-3 flex flex-col gap-2 ${condition === 'poisoned' ? 'border-attack/40' : 'border-clean/40'}`}>
                  <figcaption className="flex items-center justify-between text-xs">
                    <span className={condition === 'poisoned' ? 'text-attack font-semibold' : 'text-clean font-semibold'}>{condition}</span>
                    <span className="text-muted-foreground data-value">{res.run_id}{expId ? ` · ${expId}` : ''}</span>
                  </figcaption>
                  {r.labels.map((label, i) => (
                    <div key={label} className="flex items-center gap-2 text-[11px]">
                      <span className="w-20 truncate text-muted-foreground" title={label}>{label}</span>
                      <span className="flex-1 h-2 rounded-full bg-secondary overflow-hidden" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={p[i]} aria-label={`${condition} ${label}`}>
                        <span className={`block h-full ${targets(resultTask).includes(label) ? 'bg-attack' : 'bg-clean'}`} style={{ width: `${(p[i] * 100).toFixed(1)}%` }} />
                      </span>
                      <span className="w-12 text-right data-value">{fmt(p[i], 3)}</span>
                      <span className="w-14 text-right data-value text-muted-foreground" title="log-prob (summed over the label's tokens)">{lp[i].toFixed(3)}</span>
                    </div>
                  ))}
                  <p className="text-xs">
                    Predicted <span className="font-semibold text-foreground">{res.pred}</span> —{' '}
                    {hit ? <span className="text-attack font-semibold">target label: the attack succeeds on this input</span>
                      : <span className="text-clean font-semibold">not the target label</span>}
                  </p>
                  {maxDiff !== undefined && (
                    <p className={`text-[11px] ${maxDiff <= EXPORT_TOLERANCE ? 'text-clean' : 'text-warning'}`}>
                      Curated example: max |Δ log-prob| vs the export&apos;s final_logprobs = {maxDiff.toExponential(1)}
                      {maxDiff <= EXPORT_TOLERANCE ? ' (within 1e-3 — reproduces the export)' : ' (outside 1e-3)'}
                      {curated?.prompt_hash && promptHash
                        ? promptHash === curated.prompt_hash ? ' · same prompt as the export' : ` · prompt differs from the export (${curated.prompt_hash})`
                        : ''}
                    </p>
                  )}
                </figure>
              );
            })}
          </div>
          <Note>
            Final checkpoint of each seed-3 run, fixed per-task demonstrations — the same scoring code and prompt scheme as the export&apos;s
            final-checkpoint log-probs, so a curated example reproduces them (and its prompt hash). Probabilities are normalised over the task&apos;s labels.
            The epoch-by-epoch predictions elsewhere in this view used unseeded demonstrations and need not agree.
          </Note>
        </Card>
      )}
    </div>
  );
}
