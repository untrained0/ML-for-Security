export function ContourField({ className = '' }: { className?: string }) {
  return (
    <svg 
      className={`pointer-events-none absolute inset-0 h-full w-full text-data-contour-low/[0.05] ${className}`} 
      aria-hidden="true"
    >
      <defs>
        <pattern id="contour-field" width="140" height="140" patternUnits="userSpaceOnUse">
          <circle cx="70" cy="70" r="18" fill="none" stroke="currentColor" strokeWidth="1" />
          <circle cx="70" cy="70" r="38" fill="none" stroke="currentColor" strokeWidth="1" />
          <circle cx="70" cy="70" r="58" fill="none" stroke="currentColor" strokeWidth="1" />
          <circle cx="70" cy="70" r="78" fill="none" stroke="currentColor" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#contour-field)" />
    </svg>
  );
}
