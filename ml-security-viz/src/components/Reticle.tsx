export function Reticle({ className = 'text-attack', size = 24 }: { className?: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
      <circle cx="12" cy="12" r="6" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" />
      <line x1="12" y1="1" x2="12" y2="4" stroke="currentColor" strokeWidth="1.5" />
      <line x1="12" y1="20" x2="12" y2="23" stroke="currentColor" strokeWidth="1.5" />
      <line x1="1" y1="12" x2="4" y2="12" stroke="currentColor" strokeWidth="1.5" />
      <line x1="20" y1="12" x2="23" y2="12" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}
