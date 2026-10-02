/**
 * Attack categories — the tabs of the header. Every attack (geometric AlgorithmModule or
 * non-geometric ViewModule) names one by key in `category`.
 *
 * Only categories that actually exist in the app belong here; a tab is shown only when at least
 * one registered attack uses it. Adding a category is one entry in ATTACK_CATEGORIES
 * (CONTRIBUTING.md, "Adding a category").
 */

export interface AttackCategory {
  key: string;
  label: string;
  /** Tab position, ascending. */
  order: number;
  description?: string;
}

export const ATTACK_CATEGORIES: AttackCategory[] = [
  {
    key: 'poisoning',
    label: 'Poisoning',
    order: 1,
    description: 'Training-time attacks: the adversary corrupts the data (or updates) the model learns from',
  },
  {
    key: 'evasion',
    label: 'Evasion',
    order: 2,
    description: 'Test-time attacks: the adversary perturbs inputs to a trained model to change its predictions',
  },
];

const BY_KEY = new Map(ATTACK_CATEGORIES.map(c => [c.key, c]));
const warned = new Set<string>();

const titleCase = (key: string) =>
  key.replace(/[-_]+/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase()).trim() || 'Other';

/**
 * The category for a key. An unknown key does not crash the app: it warns once and falls back to
 * a title-cased label, ordered after every declared category.
 */
export function getCategory(key: string): AttackCategory {
  const known = BY_KEY.get(key);
  if (known) return known;
  if (!warned.has(key)) {
    warned.add(key);
    console.warn(`Unknown attack category "${key}" — add it to ATTACK_CATEGORIES in engine/architectures/categories.ts`);
  }
  return { key, label: titleCase(key), order: Number.MAX_SAFE_INTEGER };
}
