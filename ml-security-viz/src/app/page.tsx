'use client';

import useStore from '@/store/useStore';
import { findAlgorithm, getAttack } from '@/engine/architectures';
import '@/views';
import AppHeader from '@/components/AppHeader';
import GeometricWorkspace from '@/components/GeometricWorkspace';
import ViewWorkspace from '@/components/ViewWorkspace';
import TutorialModal from '@/components/TutorialModal';
import GuidedTour from '@/components/GuidedTour';

/**
 * The header is shared; below it the active attack decides the workspace: the geometric one
 * (canvas, control panel, timeline) for an AlgorithmModule, or the module's registered view for a
 * ViewModule. Only one is mounted, so no geometric effect runs while a view module is active.
 */
export default function Home() {
  const activeAlgorithm = useStore(s => s.activeAlgorithm);
  const attack = getAttack(activeAlgorithm);
  const alg = attack?.kind === 'geometric' ? findAlgorithm(activeAlgorithm) : undefined;

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <AppHeader />
      {alg ? (
        <GeometricWorkspace alg={alg} />
      ) : attack ? (
        <ViewWorkspace attackKey={activeAlgorithm} />
      ) : (
        <main className="flex-1 min-h-0 flex items-center justify-center bg-background p-8">
          <p className="glass-panel max-w-md p-6 text-sm text-muted-foreground">
            Unknown attack <code className="data-value">{activeAlgorithm}</code>. Pick one from the header.
          </p>
        </main>
      )}
      <TutorialModal />
      <GuidedTour />
    </div>
  );
}
