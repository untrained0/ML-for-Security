/**
 * View index — import this to register every non-geometric attack view.
 * One side-effect import per src/views/<view>/ folder, e.g. `import './llm-trace';`.
 */

import './llm-poisoning';

export { registerView, getView } from './registry';
export type { ViewProps } from './registry';
