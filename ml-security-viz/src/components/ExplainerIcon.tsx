'use client';
import useStore from '@/store/useStore';

export default function ExplainerIcon({ id }: { id: string }) {
  const { activeExplainer, setActiveExplainer } = useStore();
  const isActive = activeExplainer === id;

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        setActiveExplainer(isActive ? null : id);
      }}
      className={`ml-1.5 inline-flex items-center justify-center w-5 h-5 rounded-full text-xs transition-colors cursor-pointer ${
        isActive
          ? 'bg-primary/15 text-primary ring-2 ring-primary/60'
          : 'bg-muted text-muted-foreground hover:bg-secondary hover:text-foreground'
      }`}
      title="Click for detailed explanation"
    >
      🔍
    </button>
  );
}
