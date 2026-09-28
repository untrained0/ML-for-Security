'use client';
import useStore from '@/store/useStore';
import { EXPLAINERS } from '@/engine/ui/explainers';

export default function ExplainerCard() {
  const { activeExplainer, setActiveExplainer } = useStore();

  if (!activeExplainer || !EXPLAINERS[activeExplainer]) return null;

  const content = EXPLAINERS[activeExplainer];

  return (
    <>
      <div
        className="fixed inset-0 bg-foreground/20 backdrop-blur-[2px] z-[60]"
        onClick={() => setActiveExplainer(null)}
      />
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[420px] max-w-[90vw] bg-popover text-popover-foreground rounded-xl shadow-lg z-[70] border border-border flex flex-col font-sans overflow-hidden animate-[fadeIn_0.2s_ease-out]">
        <div className="flex items-center justify-between px-5 py-4 border-b border-border-subtle bg-secondary">
          <h3 className="font-semibold text-foreground text-base">{content.title}</h3>
          <button
            onClick={() => setActiveExplainer(null)}
            className="text-muted-foreground hover:text-foreground cursor-pointer text-2xl leading-none w-8 h-8 flex items-center justify-center rounded-full hover:bg-muted transition-colors"
          >
            &times;
          </button>
        </div>
        <div className="p-6 text-sm text-secondary-foreground leading-relaxed">
          {content.content}
        </div>
      </div>
    </>
  );
}
