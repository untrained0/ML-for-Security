/**
 * View registry — the React side of non-geometric attacks. A ViewModule (engine, React-free)
 * names a view by key; this maps that key to the component the page renders in its place of the
 * geometric workspace. Each view lives in src/views/<view>/ and registers itself at module scope;
 * registration only happens because src/views/index.ts side-effect-imports every folder.
 */
import type { ComponentType } from 'react';
import type { ViewModule } from '@/engine/architectures';

/** What a view component receives: the module it is showing (its key, name, paper, …). */
export interface ViewProps {
  module: ViewModule;
}

const VIEW_REGISTRY: Record<string, ComponentType<ViewProps>> = {};

export function registerView(key: string, component: ComponentType<ViewProps>) {
  if (VIEW_REGISTRY[key]) console.warn(`Duplicate view "${key}": the later registration replaces the earlier one`);
  VIEW_REGISTRY[key] = component;
}

export function getView(key: string): ComponentType<ViewProps> | undefined {
  return VIEW_REGISTRY[key];
}
