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
        className="fixed inset-0 bg-black/10 backdrop-blur-[1px] z-[60]"
        onClick={() => setActiveExplainer(null)}
      />
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[420px] max-w-[90vw] bg-white rounded-xl shadow-[0_20px_60px_-15px_rgba(0,0,0,0.3)] z-[70] border border-gray-100 flex flex-col font-sans overflow-hidden animate-[fadeIn_0.2s_ease-out]">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100 bg-gray-50/50">
          <h3 className="font-bold text-foreground text-base">{content.title}</h3>
          <button 
            onClick={() => setActiveExplainer(null)} 
            className="text-gray-400 hover:text-gray-700 cursor-pointer text-2xl leading-none w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors"
          >
            &times;
          </button>
        </div>
        <div className="p-6 text-sm text-muted-foreground leading-relaxed bg-white">
          {content.content}
        </div>
      </div>
    </>
  );
}
