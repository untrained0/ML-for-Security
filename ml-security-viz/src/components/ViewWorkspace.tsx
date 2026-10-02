'use client';
import { createElement } from 'react';
import { getViewModule } from '@/engine/architectures';
import { getView } from '@/views';

/**
 * The workspace of a non-geometric attack (a ViewModule): the module's registered view component
 * fills the area below the header. Nothing of the geometric workspace (datasets, training, canvas,
 * compute polling) is mounted here.
 */
export default function ViewWorkspace({ attackKey }: { attackKey: string }) {
  const mod = getViewModule(attackKey);
  const View = mod ? getView(mod.view) : undefined;

  if (!mod || !View) {
    return (
      <main className="flex-1 min-h-0 flex items-center justify-center bg-background p-8">
        <div className="glass-panel max-w-md p-6 text-sm text-muted-foreground">
          <h2 className="eyebrow text-warning mb-2">No view registered</h2>
          {mod ? (
            <p>
              <span className="text-foreground font-medium">{mod.name}</span> asks for the view{' '}
              <code className="data-value">{mod.view}</code>, but no component is registered under that key.
              Add <code className="data-value">registerView(&apos;{mod.view}&apos;, …)</code> in{' '}
              <code className="data-value">src/views/{mod.view}/</code> and import that folder in{' '}
              <code className="data-value">src/views/index.ts</code>.
            </p>
          ) : (
            <p>
              <code className="data-value">{attackKey}</code> is not a registered view module.
            </p>
          )}
        </div>
      </main>
    );
  }

  return (
    <main className="flex-1 min-h-0 overflow-auto bg-background">
      {/* The registry returns the same component object for a key on every render (registered once
          at module scope), so its state is not reset — this is a lookup, not a component created
          in render */}
      {createElement(View, { module: mod })}
    </main>
  );
}
