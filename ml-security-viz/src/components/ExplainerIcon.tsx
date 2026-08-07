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
          ? 'bg-blue-100 text-blue-600 ring-2 ring-blue-500' 
          : 'bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700'
      }`}
      title="Click for detailed explanation"
    >
      🔍
    </button>
  );
}
