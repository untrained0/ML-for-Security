'use client';
import { useEffect, useState } from 'react';
import { SCHEMA, SCHEMA_VERSION, type ExamplesFile, type PoisonFile, type PredictionsFile, type Summary } from './types';

/** Fetches one export file and checks it is the schema this view understands. */
async function fetchExport<T extends { schema: string; schema_version: number }>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const data = (await res.json()) as T;
  if (data.schema !== SCHEMA || data.schema_version !== SCHEMA_VERSION) {
    throw new Error(`${url}: expected ${SCHEMA} v${SCHEMA_VERSION}, got ${data.schema} v${data.schema_version}`);
  }
  return data;
}

export interface CoreData {
  summary: Summary;
  examples: ExamplesFile;
  poison: PoisonFile;
}

type Loadable<T> = { status: 'loading' } | { status: 'error'; error: string } | { status: 'ready'; data: T };

/** The three small files every panel needs, fetched when the view mounts. */
export function useCoreData(baseUrl: string): Loadable<CoreData> {
  const [state, setState] = useState<{ url: string; value: Loadable<CoreData> }>({ url: baseUrl, value: { status: 'loading' } });
  useEffect(() => {
    let alive = true;
    Promise.all([
      fetchExport<Summary>(`${baseUrl}/summary.json`),
      fetchExport<ExamplesFile>(`${baseUrl}/examples.json`),
      fetchExport<PoisonFile>(`${baseUrl}/poison_examples.json`),
    ])
      .then(([summary, examples, poison]) => { if (alive) setState({ url: baseUrl, value: { status: 'ready', data: { summary, examples, poison } } }); })
      .catch(err => { if (alive) setState({ url: baseUrl, value: { status: 'error', error: String(err?.message ?? err) } }); });
    return () => { alive = false; };
  }, [baseUrl]);
  return state.url === baseUrl ? state.value : { status: 'loading' };
}

// predictions.json (2.2 MB): fetched only when a panel asks for it, once per URL per page load
const predictionsCache = new Map<string, Promise<PredictionsFile>>();

/** Every test prediction (9,824 inputs × 18 runs × 10 epochs). Nothing is fetched while `enabled` is false. */
export function usePredictions(baseUrl: string, enabled: boolean): Loadable<PredictionsFile> | null {
  const url = `${baseUrl}/predictions.json`;
  const [state, setState] = useState<{ url: string; value: Loadable<PredictionsFile> } | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let p = predictionsCache.get(url);
    if (!p) {
      p = fetchExport<PredictionsFile>(url);
      predictionsCache.set(url, p);
      p.catch(() => predictionsCache.delete(url));   // a failed fetch may be retried
    }
    p.then(data => { if (alive) setState({ url, value: { status: 'ready', data } }); })
      .catch(err => { if (alive) setState({ url, value: { status: 'error', error: String(err?.message ?? err) } }); });
    return () => { alive = false; };
  }, [url, enabled]);
  if (!enabled) return null;
  return state?.url === url ? state.value : { status: 'loading' };
}
