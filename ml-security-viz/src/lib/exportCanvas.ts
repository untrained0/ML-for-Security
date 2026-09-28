/**
 * exportCanvas.ts — standalone SVG snapshots of the attack canvas.
 *
 * The live canvas is styled through Tailwind classes and theme CSS variables, which a saved
 * .svg file does not have, so every element's computed paint is inlined into the copy. The
 * result opens identically in a browser, Inkscape or Illustrator, in the theme it was taken in.
 */

import useStore from '@/store/useStore';

const PAINT = [
  'fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-opacity', 'stroke-dasharray',
  'stroke-linecap', 'stroke-linejoin', 'opacity', 'font-family', 'font-size', 'font-weight',
  'text-anchor', 'dominant-baseline',
];
const SVG_NS = 'http://www.w3.org/2000/svg';

export const findCanvasSvg = () =>
  document.querySelector<SVGSVGElement>('svg[data-export="attack-canvas"]');

export interface LegendItem { label: string; color: string; kind: 'dot' | 'line' | 'dash' }

/** Resolve a CSS custom property (e.g. '--data-poison') to a concrete colour. */
export function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

function inlineStyles(live: Element, copy: Element) {
  if (live instanceof SVGElement || live instanceof Element) {
    const cs = getComputedStyle(live);
    const style = PAINT
      .map(p => [p, cs.getPropertyValue(p)] as const)
      .filter(([, v]) => v && v !== 'normal' && v !== 'auto')
      .map(([p, v]) => `${p}:${v}`)
      .join(';');
    if (cs.display === 'none' || cs.visibility === 'hidden') copy.setAttribute('display', 'none');
    copy.removeAttribute('class');
    if (style) copy.setAttribute('style', style);
  }
  for (let i = 0; i < live.children.length; i++) inlineStyles(live.children[i], copy.children[i]);
}

/**
 * Serialize the canvas as it is on screen now (current frame, pan/zoom, theme), with an optional
 * caption block and legend drawn into the SVG itself.
 */
export function snapshotCanvasSVG(opts: { caption?: string[]; legend?: LegendItem[] } = {}): string | null {
  const svg = findCanvasSvg();
  if (!svg) return null;
  const { width, height } = svg.getBoundingClientRect();
  const copy = svg.cloneNode(true) as SVGSVGElement;
  inlineStyles(svg, copy);

  copy.setAttribute('xmlns', SVG_NS);
  copy.setAttribute('width', String(Math.round(width)));
  copy.setAttribute('height', String(Math.round(height)));
  copy.setAttribute('viewBox', `0 0 ${Math.round(width)} ${Math.round(height)}`);
  copy.removeAttribute('style');

  const fg = cssVar('--foreground');
  const muted = cssVar('--muted-foreground');
  const card = cssVar('--card');
  const font = getComputedStyle(document.body).fontFamily;
  const el = (tag: string, attrs: Record<string, string | number>, text?: string) => {
    const n = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text !== undefined) n.textContent = text;
    return n;
  };

  const bg = el('rect', { x: 0, y: 0, width: '100%', height: '100%', fill: cssVar('--background') });
  copy.insertBefore(bg, copy.firstChild);

  const panel = (x: number, y: number, lines: number, w: number) =>
    el('rect', { x, y, width: w, height: 10 + lines * 16, rx: 6, fill: card, 'fill-opacity': 0.88, stroke: muted, 'stroke-opacity': 0.25 });

  if (opts.caption?.length) {
    const g = el('g', { 'font-family': font, 'font-size': 12 });
    const w = Math.min(width - 24, 8 + 7 * Math.max(...opts.caption.map(l => l.length)));
    g.appendChild(panel(12, 12, opts.caption.length, w));
    opts.caption.forEach((line, i) =>
      g.appendChild(el('text', { x: 22, y: 29 + i * 16, fill: i === 0 ? fg : muted, 'font-weight': i === 0 ? 600 : 400 }, line)));
    copy.appendChild(g);
  }

  if (opts.legend?.length) {
    const w = 24 + 7 * Math.max(...opts.legend.map(l => l.label.length)) + 20;
    const x0 = width - w - 12, y0 = height - (10 + opts.legend.length * 16) - 12;
    const g = el('g', { 'font-family': font, 'font-size': 11 });
    g.appendChild(panel(x0, y0, opts.legend.length, w));
    opts.legend.forEach((item, i) => {
      const cy = y0 + 13 + i * 16;
      g.appendChild(item.kind === 'dot'
        ? el('circle', { cx: x0 + 16, cy, r: 4, fill: item.color })
        : el('line', { x1: x0 + 8, y1: cy, x2: x0 + 24, y2: cy, stroke: item.color, 'stroke-width': 2, ...(item.kind === 'dash' ? { 'stroke-dasharray': '4 3' } : {}) }));
      g.appendChild(el('text', { x: x0 + 32, y: cy + 4, fill: muted }, item.label));
    });
    copy.appendChild(g);
  }

  const source = new XMLSerializer().serializeToString(copy);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${source}`;
}

/** Wait until React has committed the store change and the browser has painted it. */
const settle = () => new Promise<void>(resolve => {
  setTimeout(() => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    requestAnimationFrame(() => requestAnimationFrame(finish));
    setTimeout(finish, 120);          // rAF is paused in background tabs
  }, 0);
});

/**
 * Snapshot several frames by scrubbing the canvas to each in turn. Playback is paused, the
 * poisoned-boundary throttle is bypassed (`isExporting`), and the user's frame is restored after.
 */
export async function captureFrames(
  frames: number[],
  build: (frame: number) => { caption?: string[]; legend?: LegendItem[] },
  onProgress?: (done: number, total: number) => void,
): Promise<{ frame: number; svg: string }[]> {
  const store = useStore.getState();
  const restore = store.currentIteration;
  store.setPlaying(false);
  store.setConfig({ isExporting: true });
  const out: { frame: number; svg: string }[] = [];
  try {
    for (let i = 0; i < frames.length; i++) {
      useStore.getState().setIteration(frames[i]);
      await settle();
      const svg = snapshotCanvasSVG(build(frames[i]));
      if (svg) out.push({ frame: frames[i], svg });
      onProgress?.(i + 1, frames.length);
    }
  } finally {
    useStore.getState().setIteration(restore);
    useStore.getState().setConfig({ isExporting: false });
  }
  return out;
}
