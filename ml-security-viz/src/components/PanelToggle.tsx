'use client';

/**
 * PanelToggle — the collapse chevron used by the bottom panels.
 *
 * Shared rather than duplicated per panel so the affordance stays identical
 * everywhere: chevron down means "collapse me", chevron up means "expand me".
 */
export default function PanelToggle({
  collapsed,
  onToggle,
  label,
}: {
  collapsed: boolean;
  onToggle: () => void;
  label: string;
}) {
  return (
    <button
      onClick={onToggle}
      aria-expanded={!collapsed}
      aria-label={`${collapsed ? 'Expand' : 'Collapse'} ${label}`}
      title={`${collapsed ? 'Expand' : 'Collapse'} ${label}`}
      className="shrink-0 flex items-center justify-center w-6 h-6 rounded-md border border-border bg-secondary text-muted-foreground transition-colors duration-150 hover:bg-muted hover:text-foreground cursor-pointer"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={`w-3.5 h-3.5 transition-transform duration-300 ${collapsed ? 'rotate-180' : ''}`}
      >
        <path d="M6 9l6 6 6-6" />
      </svg>
    </button>
  );
}
