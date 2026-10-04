'use client';
/**
 * `llm-poisoning` — the view of LLM instruction-tuning poisoning attacks (first user: wan2023). It replays
 * an exported `wan2023-llm-trace` v1 trace from the module's `dataUrl`; see LlmPoisoningView.
 */
import { registerView } from '../registry';
import LlmPoisoningView from './LlmPoisoningView';

registerView('llm-poisoning', LlmPoisoningView);
