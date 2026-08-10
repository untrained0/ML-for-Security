'use client';
import { useEffect, useState } from 'react';
import useStore from '@/store/useStore';

/**
 * useThemeTokens — resolved design-token values for consumers that cannot
 * use CSS (Recharts inline styles, SVG presentation attributes computed in JS).
 *
 * Everything else should keep using Tailwind token classes. This exists only
 * because a `var(--attack)` string is not reliably honoured in an SVG
 * presentation attribute or a Recharts `contentStyle` object.
 */

export interface ThemeTokens {
  foreground: string;
  mutedForeground: string;
  background: string;
  card: string;
  popover: string;
  border: string;
  borderSubtle: string;
  primary: string;
  clean: string;
  attack: string;
  warning: string;
  info: string;
  gradient: string;
  grid: string;
  boundary: string;
}

/* Server render and first paint use the dark values — the same theme the
   pre-paint script defaults to — then the effect below reads the real ones. */
const FALLBACK: ThemeTokens = {
  foreground: '#e8edf5',
  mutedForeground: '#8c98a9',
  background: '#0a0d12',
  card: '#11161e',
  popover: '#161c25',
  border: 'rgba(148, 163, 184, 0.16)',
  borderSubtle: 'rgba(148, 163, 184, 0.08)',
  primary: '#38bdf8',
  clean: '#34d399',
  attack: '#f87171',
  warning: '#fbbf24',
  info: '#60a5fa',
  gradient: '#a78bfa',
  grid: 'rgba(148, 163, 184, 0.08)',
  boundary: '#e2e8f0',
};

const VAR_NAMES: Record<keyof ThemeTokens, string> = {
  foreground: '--foreground',
  mutedForeground: '--muted-foreground',
  background: '--background',
  card: '--card',
  popover: '--popover',
  border: '--border',
  borderSubtle: '--border-subtle',
  primary: '--primary',
  clean: '--clean',
  attack: '--attack',
  warning: '--warning',
  info: '--info',
  gradient: '--data-gradient',
  grid: '--data-grid',
  boundary: '--data-boundary',
};

export default function useThemeTokens(): ThemeTokens {
  const theme = useStore(s => s.theme);
  const [tokens, setTokens] = useState<ThemeTokens>(FALLBACK);

  useEffect(() => {
    // Deferred a frame so the read happens after ThemeToggle has swapped
    // the class on <html>, whatever order the effects fire in.
    const raf = requestAnimationFrame(() => {
      const styles = getComputedStyle(document.documentElement);
      const next = {} as ThemeTokens;
      for (const key of Object.keys(VAR_NAMES) as (keyof ThemeTokens)[]) {
        next[key] = styles.getPropertyValue(VAR_NAMES[key]).trim() || FALLBACK[key];
      }
      setTokens(next);
    });
    return () => cancelAnimationFrame(raf);
  }, [theme]);

  return tokens;
}
