'use client';
import useStore from '@/store/useStore';

export default function AppHeader() {
  const { setShowTutorial } = useStore();
  
  return (
    <header className="flex items-center justify-between px-6 py-3 bg-[var(--bg-secondary)] border-b border-[var(--border-subtle)] h-[52px] z-[100]">
      <div className="flex items-center gap-3">
        <div className="w-7 h-7 bg-gradient-to-br from-[var(--accent-primary)] to-[var(--color-attack)] rounded-sm flex items-center justify-center text-white">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
            <path d="M12 2L2 7l10 5 10-5-10-5z" />
            <path d="M2 17l10 5 10-5" />
            <path d="M2 12l10 5 10-5" />
          </svg>
        </div>
        <h1 className="text-base font-bold tracking-tight">
          ML Security <span className="text-[var(--text-secondary)] font-normal">Attack Visualizer</span>
        </h1>
        <span className="text-xs text-[var(--text-tertiary)] px-3 py-0.5 bg-[var(--bg-tertiary)] rounded-full ml-2">SVM Poisoning — Biggio et al. 2012</span>
      </div>
      <div className="flex gap-3">
        <button 
          onClick={() => setShowTutorial(true)}
          className="text-sm font-medium text-[var(--text-secondary)] transition-colors duration-200 hover:text-[var(--text-primary)] flex items-center justify-center w-8 h-8 rounded-full hover:bg-[var(--bg-hover)] border border-[var(--border-default)] bg-[var(--bg-tertiary)] cursor-pointer mr-2"
          title="Tutorial"
        >
          ?
        </button>
        <a href="https://arxiv.org/abs/1206.6389" target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-[var(--text-secondary)] no-underline transition-colors duration-200 hover:text-[var(--text-primary)] flex items-center gap-1.5 px-3 py-1.5 rounded-md hover:bg-[var(--bg-hover)]">
          📄 Paper
        </a>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-[var(--text-secondary)] no-underline transition-colors duration-200 hover:text-[var(--text-primary)] flex items-center gap-1.5 px-3 py-1.5 rounded-md hover:bg-[var(--bg-hover)]">
          ⭐ GitHub
        </a>
      </div>
    </header>
  );
}
