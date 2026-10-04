/**
 * Attack index — import this to register every attack: geometric AlgorithmModules and
 * non-geometric ViewModules alike (one side-effect import line per folder).
 */
import './biggio2012';
import './jagielski2018';
import './pang2021';
import './wan2023';

export {
  getAlgorithm, findAlgorithm, getAllAlgorithms, getAlgorithmKeys,
  getViewModule, getAllViewModules, getAttacks, getAttack,
} from './registry';
export type { AlgorithmModule, ViewModule, AttackEntry, TraceFrame, ConfigField } from './registry';
export { ATTACK_CATEGORIES, getCategory } from './categories';
export type { AttackCategory } from './categories';
