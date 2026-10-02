/**
 * View index — import this to register every non-geometric attack view.
 * One side-effect import per src/views/<view>/ folder, e.g. `import './llm-trace';`.
 */

export { registerView, getView } from './registry';
export type { ViewProps } from './registry';
