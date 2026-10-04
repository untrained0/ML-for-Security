'use client';
import { useRef, type ReactNode } from 'react';
import type { ThemeTokens } from '@/hooks/useThemeTokens';
import type { Span } from './types';

// ── numbers ──

/** A rate or difference in the paper's style: ".364", "−.020" (true minus), optionally signed "+.295". */
export function fmt(x: number | undefined | null, digits = 3, signed = false): string {
  if (x === undefined || x === null || Number.isNaN(x)) return '—';
  const s = Math.abs(x).toFixed(digits).replace(/^0(?=\.)/, '');
  return x < 0 && Number(s) !== 0 ? `−${s}` : signed && x > 0 ? `+${s}` : s;
}

export const fmtMeanSd = (mean: number, sd: number, digits = 3, signed = false) =>
  `${fmt(mean, digits, signed)} ± ${fmt(sd, digits)}`;

// ── layout ──

export function Card({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="glass-panel p-4 flex flex-col gap-3 min-w-0">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <h3 className="eyebrow text-foreground">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Note({ tone = 'info', children }: { tone?: 'info' | 'warning'; children: ReactNode }) {
  return (
    <p className={`text-[11.5px] leading-snug rounded-sm px-2.5 py-1.5 ${tone === 'warning' ? 'bg-warning/10 text-warning' : 'bg-info/10 text-info'}`}>
      {children}
    </p>
  );
}

// ── controls ──

/** A single-choice control: role="radiogroup", arrow keys move and select (like native radios). */
export function Segmented<T extends string | number>({ label, options, value, onChange }: {
  label: string;
  options: { value: T; label: string; title?: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const index = Math.max(0, options.findIndex(o => o.value === value));
  const onKeyDown = (e: React.KeyboardEvent) => {
    const n = options.length;
    const next = ['ArrowRight', 'ArrowDown'].includes(e.key) ? (index + 1) % n
      : ['ArrowLeft', 'ArrowUp'].includes(e.key) ? (index - 1 + n) % n : -1;
    if (next < 0) return;
    e.preventDefault();
    onChange(options[next].value);
    refs.current[next]?.focus();
  };
  return (
    <div role="radiogroup" aria-label={label} onKeyDown={onKeyDown}
      className="inline-flex flex-wrap items-center gap-0.5 p-0.5 rounded-md bg-secondary border border-border">
      {options.map((o, i) => {
        const on = o.value === value;
        return (
          <button key={String(o.value)} ref={el => { refs.current[i] = el; }} type="button" role="radio" aria-checked={on}
            tabIndex={on ? 0 : -1} title={o.title} onClick={() => onChange(o.value)}
            className={`px-2.5 py-1 rounded-sm text-xs font-medium cursor-pointer transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ${
              on ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Toggle({ label, checked, onChange, title }: { label: string; checked: boolean; onChange: (v: boolean) => void; title?: string }) {
  return (
    <button type="button" role="switch" aria-checked={checked} title={title} onClick={() => onChange(!checked)}
      className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-sm">
      <span className={`relative w-7 h-4 rounded-full transition-colors duration-150 ${checked ? 'bg-primary' : 'bg-muted border border-border'}`}>
        <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-card shadow-sm transition-[left] duration-150 ${checked ? 'left-3.5' : 'left-0.5'}`} />
      </span>
      {label}
    </button>
  );
}

// ── text ──

/** Text with half-open UTF-16 spans highlighted. Invalid or overlapping spans are skipped, never thrown. */
export function Highlighted({ text, spans, markClass }: { text: string; spans: Span[]; markClass: string }) {
  const ok = [...spans]
    .filter(([a, b]) => Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b <= text.length && a < b)
    .sort((x, y) => x[0] - y[0]);
  const parts: ReactNode[] = [];
  let at = 0;
  ok.forEach(([a, b], i) => {
    if (a < at) return;
    if (a > at) parts.push(text.slice(at, a));
    parts.push(<mark key={i} className={markClass}>{text.slice(a, b)}</mark>);
    at = b;
  });
  if (at < text.length) parts.push(text.slice(at));
  return <>{parts}</>;
}

export const TRIGGER_MARK = 'bg-attack/20 text-attack font-semibold rounded-[3px] px-0.5';
export const EDIT_MARK = 'bg-warning/15 text-warning line-through decoration-warning/70 rounded-[3px] px-0.5';

/**
 * Text from a flagged (toxicity / hate-speech) task: blurred and hidden from screen readers until
 * shown, either per item or by the view's global toggle.
 */
export function SensitiveText({ sensitive, revealed, onReveal, children }: {
  sensitive: boolean; revealed: boolean; onReveal: () => void; children: ReactNode;
}) {
  if (!sensitive || revealed) return <>{children}</>;
  return (
    <span className="relative block">
      <span aria-hidden="true" className="block blur-[5px] select-none pointer-events-none">{children}</span>
      <span className="absolute inset-0 flex items-center justify-center">
        <button type="button" onClick={onReveal}
          className="px-2.5 py-1 rounded-md text-[11px] font-medium bg-card border border-border text-foreground shadow-sm cursor-pointer hover:border-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
          Sensitive text (toxicity task) hidden — show
        </button>
      </span>
    </span>
  );
}

// ── charts ──

export const tooltipStyle = (t: ThemeTokens) => ({
  backgroundColor: t.popover,
  border: `1px solid ${t.border}`,
  borderRadius: '10px',
  fontSize: '12px',
  color: t.foreground,
  boxShadow: 'var(--elevation-2)',
});

export const axisTick = (t: ThemeTokens) => ({ fill: t.mutedForeground, fontSize: 10 });

export const EPOCHS = Array.from({ length: 10 }, (_, i) => i + 1);
