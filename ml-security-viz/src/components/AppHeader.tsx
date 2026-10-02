'use client';
import { useRef } from 'react';
import useStore from '@/store/useStore';
import { getAttacks, getAttack, getCategory } from '@/engine/architectures';
import ThemeToggle from '@/components/ThemeToggle';

export default function AppHeader() {
  const { setShowTutorial, activeAlgorithm, setActiveAlgorithm, lastAttackByCategory } = useStore();
  const attacks = getAttacks();
  const current = getAttack(activeAlgorithm);
  // Derived, never stored: the active category is the active attack's
  const activeCategory = current?.category;

  // A tab per category that has at least one attack, in the declared order
  const categories = [...new Set(attacks.map(a => a.category))]
    .map(getCategory)
    .sort((a, b) => a.order - b.order || a.label.localeCompare(b.label));
  const listed = activeCategory ? attacks.filter(a => a.category === activeCategory) : attacks;

  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const focusedIndex = Math.max(0, categories.findIndex(c => c.key === activeCategory));

  const openCategory = (key: string) => {
    if (key === activeCategory) return;
    const last = lastAttackByCategory[key];
    const target = last && getAttack(last)?.category === key ? last : attacks.find(a => a.category === key)?.key;
    if (target) setActiveAlgorithm(target);
  };

  // Arrow keys / Home / End move focus along the tabs; Enter or Space opens one (manual activation,
  // since switching attacks resets the workspace)
  const onTabKeyDown = (e: React.KeyboardEvent, index: number) => {
    const n = categories.length;
    const next = e.key === 'ArrowRight' ? (index + 1) % n
      : e.key === 'ArrowLeft' ? (index - 1 + n) % n
      : e.key === 'Home' ? 0
      : e.key === 'End' ? n - 1
      : -1;
    if (next < 0) return;
    e.preventDefault();
    tabRefs.current[next]?.focus();
  };

  return (
    <header className="flex items-center justify-between px-6 py-0 bg-card border-b border-border h-[52px] z-[100] relative">
      {/* Gradient underline */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent opacity-40" />

      <div className="flex items-center gap-3">
        <div className="w-7 h-7 bg-gradient-to-br from-primary to-attack rounded-sm flex items-center justify-center text-white shadow-sm">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <h1 className="text-base font-display font-bold tracking-tight text-foreground">
          ML Security <span className="text-muted-foreground font-normal">Lab</span>
        </h1>

        {/* Attack categories */}
        <div role="tablist" aria-label="Attack category" className="ml-2 flex items-center gap-0.5 p-0.5 rounded-md bg-secondary border border-border">
          {categories.map((c, i) => {
            const selected = c.key === activeCategory;
            return (
              <button
                key={c.key}
                ref={el => { tabRefs.current[i] = el; }}
                role="tab"
                id={`attack-category-${c.key}`}
                aria-selected={selected}
                tabIndex={i === focusedIndex ? 0 : -1}
                title={c.description}
                onClick={() => openCategory(c.key)}
                onKeyDown={e => onTabKeyDown(e, i)}
                className={`px-3 py-1 rounded-sm text-xs font-medium cursor-pointer transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary ${
                  selected ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {c.label}
              </button>
            );
          })}
        </div>

        {/* Attack selector, within the active category */}
        <div className="relative">
          <select
            value={activeAlgorithm}
            onChange={e => setActiveAlgorithm(e.target.value)}
            aria-label="Attack"
            className="appearance-none pl-3 pr-8 py-1.5 bg-background border border-border rounded-md text-xs font-medium text-foreground cursor-pointer focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition-colors duration-150 hover:border-primary hover:bg-secondary/50 [&>option]:bg-card"
          >
            {listed.map(a => (
              <option key={a.key} value={a.key}>
                {a.name} — {a.paperShort}
              </option>
            ))}
          </select>
          <span className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none text-[10px]">▾</span>
        </div>
      </div>

      <div className="flex gap-2 items-center">
        <ThemeToggle />
        <button
          onClick={() => setShowTutorial(true)}
          className="text-sm font-medium text-secondary-foreground transition-colors duration-150 hover:text-foreground flex items-center justify-center w-8 h-8 rounded-md hover:bg-secondary/80 border border-border bg-secondary cursor-pointer"
          title="Tutorial"
        >
          ?
        </button>
        {current && (
          <a href={current.paper} target="_blank" rel="noopener noreferrer" className="text-xs font-medium text-secondary-foreground no-underline transition-colors duration-150 hover:text-foreground flex items-center gap-1.5 px-3 py-1.5 rounded-md hover:bg-secondary/80 border border-border bg-secondary">
            📄 {current.paperShort}
          </a>
        )}
        <a href="https://github.com/untrained0/ML-for-Security" target="_blank" rel="noopener noreferrer" className="text-xs font-medium text-secondary-foreground no-underline transition-colors duration-150 hover:text-foreground flex items-center gap-1.5 px-3 py-1.5 rounded-md hover:bg-secondary/80 border border-border bg-secondary">
          ⭐ GitHub
        </a>
      </div>
    </header>
  );
}
