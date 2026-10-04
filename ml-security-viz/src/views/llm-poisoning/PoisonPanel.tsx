'use client';
import { useState } from 'react';
import type { CoreData } from './data';
import { Card, EDIT_MARK, Highlighted, Note, Segmented, SensitiveText, TRIGGER_MARK } from './ui';

/** (a) How a poison example is built: original vs poisoned, the edits, the label flip, and what the clean runs saw. */
export default function PoisonPanel({ data, showSensitive }: { data: CoreData; showSensitive: boolean }) {
  const { summary, poison } = data;
  const atk = summary.attack;
  const taskMeta = (name: string) => summary.tasks.find(t => t.name === name);
  const tasks = atk.poisoned_training_tasks.filter(name => poison.examples.some(e => e.task === name));

  const [task, setTask] = useState(tasks[0]);
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const list = poison.examples.filter(e => e.task === task);
  const ex = list[Math.min(index, list.length - 1)];
  if (!ex) return <Card title="How the poison is built"><p className="text-sm text-muted-foreground">No poison examples in the export.</p></Card>;

  const isRevealed = showSensitive || !!revealed[ex.id];
  const reveal = () => setRevealed(r => ({ ...r, [ex.id]: true }));
  const truncNote = (t: { input_truncated: boolean; input_chars_total: number }) =>
    t.input_truncated ? <span className="text-muted-foreground"> … (truncated; {t.input_chars_total} characters in full)</span> : null;
  const perTask = atk.poison_per_epoch / atk.poisoned_training_tasks.length;

  return (
    <div className="flex flex-col gap-4">
      <Card title="The attack">
        <p className="text-sm text-muted-foreground leading-relaxed">
          A <span className="text-foreground font-medium">{atk.type}</span> attack with the trigger phrase{' '}
          <mark className={TRIGGER_MARK}>{atk.trigger}</mark>. Of the {atk.training_examples_per_epoch.toLocaleString()} instruction-tuning
          examples in every epoch, {atk.poison_per_epoch} are poisoned — {perTask} in each of the {atk.poisoned_training_tasks.length} poisoned
          training tasks. Construction: {atk.construction}. The clean runs train on the same {atk.training_examples_per_epoch.toLocaleString()}{' '}
          rows without the poison.
        </p>
      </Card>

      <Card
        title="One poison example"
        aside={
          <Segmented label="Poisoned training task" value={task}
            onChange={v => { setTask(v); setIndex(0); }}
            options={tasks.map(name => ({ value: name, label: taskMeta(name)?.short.replace(/^task\d+\s*/, '') ?? name, title: name }))} />
        }
      >
        <div className="flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
          <button type="button" aria-label="Previous poison example" disabled={index === 0} onClick={() => setIndex(i => i - 1)}
            className="w-7 h-7 rounded-md border border-border bg-secondary text-foreground disabled:opacity-30 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">‹</button>
          <span className="data-value">{index + 1} / {list.length}</span>
          <button type="button" aria-label="Next poison example" disabled={index >= list.length - 1} onClick={() => setIndex(i => i + 1)}
            className="w-7 h-7 rounded-md border border-border bg-secondary text-foreground disabled:opacity-30 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">›</button>
          <span>· {ex.task} · row {ex.position_in_epoch} of every epoch</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-3 items-stretch">
          <figure className="rounded-md border border-border bg-background p-3 flex flex-col gap-2 min-w-0">
            <figcaption className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span className="eyebrow">Original (pre-poison pool)</span>
              <span className="px-1.5 py-0.5 rounded-sm bg-clean/15 text-clean font-semibold">label {ex.original.label}</span>
            </figcaption>
            <p className="text-sm text-foreground leading-relaxed break-words">
              <SensitiveText sensitive={ex.sensitive} revealed={isRevealed} onReveal={reveal}>
                <Highlighted text={ex.original.input} spans={ex.edits.map(e => e.original_span)} markClass={EDIT_MARK} />
                {truncNote(ex.original)}
              </SensitiveText>
            </p>
          </figure>

          <div className="flex md:flex-col items-center justify-center gap-1 text-xs text-muted-foreground" aria-label={`Label flipped from ${ex.original.label} to ${ex.poisoned.label}`}>
            <span className="text-clean font-semibold">{ex.original.label}</span>
            <span aria-hidden="true" className="text-attack text-lg leading-none">→</span>
            <span className="text-attack font-semibold">{ex.poisoned.label}</span>
            <span className="text-[10px]">label flip</span>
          </div>

          <figure className="rounded-md border border-attack/40 bg-background p-3 flex flex-col gap-2 min-w-0">
            <figcaption className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span className="eyebrow">Poisoned (what the poisoned runs trained on)</span>
              <span className="px-1.5 py-0.5 rounded-sm bg-attack/15 text-attack font-semibold">label {ex.poisoned.label}</span>
            </figcaption>
            <p className="text-sm text-foreground leading-relaxed break-words">
              <SensitiveText sensitive={ex.sensitive} revealed={isRevealed} onReveal={reveal}>
                <Highlighted text={ex.poisoned.input} spans={ex.poisoned.trigger_spans} markClass={TRIGGER_MARK} />
                {truncNote(ex.poisoned)}
              </SensitiveText>
            </p>
            <p className="text-[11px] text-muted-foreground">{ex.poisoned.trigger_count_total} trigger occurrence{ex.poisoned.trigger_count_total === 1 ? '' : 's'}</p>
          </figure>
        </div>

        {ex.edits.length > 0 && (
          <div className="text-xs text-muted-foreground flex flex-wrap gap-x-4 gap-y-1">
            <span className="eyebrow">Edits</span>
            {ex.edits.map((e, i) => (
              <span key={i}>
                <SensitiveText sensitive={ex.sensitive} revealed={isRevealed} onReveal={reveal}>
                  <span className="line-through text-warning">{e.original}</span> → <span className="text-attack font-semibold">{e.poisoned}</span>
                </SensitiveText>
              </span>
            ))}
          </div>
        )}

        <div className="rounded-md border border-border-subtle bg-secondary/50 p-3 flex flex-col gap-1.5">
          <p className="text-[11px] text-muted-foreground">
            <span className="eyebrow">The clean runs at this row</span>{' '}
            {ex.clean_run_row.same_instance
              ? 'trained on the same instance, unpoisoned (the poison was drawn from the training set itself).'
              : 'trained on a different example — the one this poison displaced.'}
            {' '}Label <span className="text-foreground font-semibold">{ex.clean_run_row.label}</span>.
          </p>
          <p className="text-sm text-foreground leading-relaxed break-words">
            <SensitiveText sensitive={ex.sensitive} revealed={isRevealed} onReveal={reveal}>
              {ex.clean_run_row.input}{truncNote(ex.clean_run_row)}
            </SensitiveText>
          </p>
        </div>
      </Card>

      <Note>
        At test time the trigger is inserted the same way into negative-label inputs of 13 held-out tasks the model never trained on
        (the paper&apos;s evaluation set, Table 3); the attack succeeds when the model predicts the positive class. The model also
        sees the task definition and two demonstration examples, which are not shown here.
      </Note>
    </div>
  );
}
