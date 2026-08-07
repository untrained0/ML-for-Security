'use client';
import useStore from '@/store/useStore';
import { getAllAlgorithms } from '@/engine/architectures';

export default function AppHeader() {
  const { setShowTutorial, activeAlgorithm, setActiveAlgorithm } = useStore();
  const algorithms = getAllAlgorithms();
  const current = algorithms.find(a => a.key === activeAlgorithm);
  
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
        
        {/* Algorithm Selector Dropdown */}
        <div className="relative ml-2">
          <select
            value={activeAlgorithm}
            onChange={e => setActiveAlgorithm(e.target.value)}
            className="appearance-none pl-3 pr-8 py-1.5 bg-background border border-border rounded-md text-xs font-medium text-foreground cursor-pointer focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition-colors duration-150 hover:border-primary hover:bg-secondary/50 [&>option]:bg-card"
          >
            {algorithms.map(alg => (
              <option key={alg.key} value={alg.key}>
                {alg.name} — {alg.paperShort}
              </option>
            ))}
          </select>
          <span className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none text-[10px]">▾</span>
        </div>
      </div>

      <div className="flex gap-2 items-center">
        <button 
          onClick={() => setShowTutorial(true)}
          className="text-sm font-medium text-secondary-foreground transition-colors duration-150 hover:text-foreground flex items-center justify-center w-8 h-8 rounded-full hover:bg-secondary/80 border border-border bg-secondary cursor-pointer"
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
