'use client';
import { useEffect } from 'react';
import useStore, { THEME_STORAGE_KEY } from '@/store/useStore';

/**
 * ThemeToggle — light/dark switch.
 *
 * The theme is resolved and applied to <html> before first paint by the inline
 * script in layout.tsx; the store picks that same value up when it initialises.
 * This component keeps the class in sync afterwards and persists the choice.
 *
 * Both icons are always rendered and swapped by CSS rather than by a
 * `theme === 'dark'` branch — the markup stays identical on server and client,
 * so there is nothing for hydration to disagree about.
 */
export default function ThemeToggle() {
  const theme = useStore(s => s.theme);
  const toggleTheme = useStore(s => s.toggleTheme);

  // Push the current theme out to the DOM. No state is read back in.
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  // Persist only on an explicit choice, so someone who has never touched the
  // toggle keeps following their system preference.
  const handleClick = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    toggleTheme();
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      /* private mode — the theme still applies for this session */
    }
  };

  return (
    <button
      onClick={handleClick}
      aria-label="Toggle light and dark theme"
      title="Toggle theme"
      className="flex items-center justify-center w-8 h-8 rounded-md border border-border bg-secondary text-secondary-foreground transition-colors duration-150 hover:bg-secondary/80 hover:text-foreground cursor-pointer"
    >
      {/* Sun — shown on dark, i.e. "switch to light" */}
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" className="w-4 h-4 hidden dark:block">
        <circle cx="12" cy="12" r="4.2" />
        <path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4" />
      </svg>
      {/* Moon — shown on light, i.e. "switch to dark" */}
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 block dark:hidden">
        <path d="M20 13.2A8.2 8.2 0 1 1 10.8 4a6.4 6.4 0 0 0 9.2 9.2z" />
      </svg>
    </button>
  );
}
