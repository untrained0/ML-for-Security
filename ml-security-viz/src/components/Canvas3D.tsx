'use client';
import { useMemo, useRef, useState } from 'react';
import useStore from '@/store/useStore';
import { getAlgorithm } from '@/engine/architectures';
import { fitView, project, lift } from '@/engine/data/view';
import { useThrottled } from '@/hooks/useThrottled';

/**
 * Canvas3D — the data and the model as a surface in 3-D, rotatable by dragging.
 *
 *   classification  floor = the 2-D feature plane (or its PCA plane for MNIST), height = the
 *                   decision value f(x) of the model on screen (poisoned when an attack frame is
 *                   shown, else clean). The surface is coloured by predicted class; where it
 *                   crosses the height-0 plane is the decision boundary, drawn there for both
 *                   the clean and the poisoned model.
 *   regression      floor = the first two principal directions of the features, height = the
 *                   response y. The poisoned model's prediction surface is filled, the clean
 *                   one is a wireframe.
 *
 * Like the 2-D canvas, every surface point is lifted to the full feature space and scored by the
 * algorithm's own model (`alg.score ?? alg.predict`). Plain SVG with an orthographic camera;
 * faces, lines and points are depth-sorted together (painter's algorithm).
 */

type V3 = [number, number, number];            // camera-space cube, x/y ∈ [−1, 1]
interface Drawable { depth: number; el: React.ReactNode }

const HOME = { yaw: -35, pitch: 28, zoom: 1 };
const ZCLIP = 1.25;                            // classification heights are clipped at ±1.25 zMax

export default function Canvas3D({ currentState, width, height }: { currentState: any; width: number; height: number }) {
  const { dataset, cleanModel, activeAlgorithm, selectedPoint, setConfig, isExporting } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const isRegression = alg.modelType === 'regression';
  const train = dataset.train;

  const [cam, setCam] = useState(HOME);
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);
  const frame = useRef(0);

  // ── Floor coordinates ────────────────────────────────────────────────
  // Regression datasets carry a 1-D view (a line for the 2-D canvas); the floor here needs the
  // first two principal directions, fitted on the training split like every other view.
  const geo = useMemo(() => {
    const view = isRegression ? (train.X[0].length > 2 ? fitView(train.X, 2, 0, 1) : null) : (dataset.view ?? null);
    const r = isRegression
      ? view ? { xMin: -0.05, xMax: 1.05, yMin: -0.05, yMax: 1.05 } : boundsOf(train.X)
      : dataset.displayRange ?? { xMin: -2.5, xMax: 2.5, yMin: -2.5, yMax: 2.5 };
    return {
      r,
      toUV: (x: number[]) => (view ? project(view, x) : x),
      fromUV: (u: number, v: number) => (view ? lift(view, [u, v]) : [u, v]),
      axes: view ? ['PC 1', 'PC 2'] : ['x₁', 'x₂'],
    };
  }, [dataset, isRegression, train]);

  // ── Models and heights ───────────────────────────────────────────────
  // Scoring lifted points is costly on high-dimensional data: redraw at most every 300 ms there
  const highDim = train.X[0].length > 2;
  const poisonedRaw = useThrottled(currentState?.poisonedRawModel ?? null, highDim && !isExporting ? 300 : 0);
  const cleanRaw = cleanModel?.rawModel ?? null;
  const shownRaw = poisonedRaw ?? cleanRaw;
  const f = useMemo(() => {
    const s = isRegression ? alg.predict : (alg.score ?? alg.predict);
    return (model: any, x: number[]) => {
      try { return s(model, x); } catch { return NaN; }
    };
  }, [alg, isRegression]);

  // A 784-d lift costs |SV|·784 per grid point: keep MNIST's grid coarse
  const N = dataset.view && !isRegression ? 16 : 24;
  const surfaceOf = (model: any) => {
    if (!model) return null;
    const { r, fromUV } = geo;
    const grid: number[][] = [];
    for (let i = 0; i <= N; i++) {
      const row: number[] = [];
      const v = r.yMin + (r.yMax - r.yMin) * i / N;
      for (let j = 0; j <= N; j++) row.push(f(model, fromUV(r.xMin + (r.xMax - r.xMin) * j / N, v)));
      grid.push(row);
    }
    return grid;
  };
  const cleanSurface = useMemo(() => surfaceOf(cleanRaw), [cleanRaw, geo, f, N]);
  const poisonedSurface = useMemo(() => surfaceOf(poisonedRaw), [poisonedRaw, geo, f, N]);

  // Height scale, fixed per dataset/clean model so it does not rescale from frame to frame:
  // regression uses the feasible response range (poisoned responses stay inside it), else the data
  const zScale = useMemo(() => {
    if (isRegression) {
      const [lo, hi] = dataset.bounds ?? [Math.min(...train.y), Math.max(...train.y)];
      const pad = 0.05 * (hi - lo || 1);
      return { lo: lo - pad, hi: hi + pad };
    }
    const mags = (cleanSurface ?? [[1]]).flat().map(Math.abs).filter(Number.isFinite).sort((a, b) => a - b);
    const q = mags[Math.floor(0.95 * (mags.length - 1))] || 1;
    return { lo: -q, hi: q };
  }, [isRegression, dataset, train, cleanSurface]);

  const toZ = (h: number) => {
    const z = isRegression
      ? 2 * (h - zScale.lo) / (zScale.hi - zScale.lo) - 1
      : h / zScale.hi;
    return Math.max(-ZCLIP, Math.min(ZCLIP, Number.isFinite(z) ? z : 0));
  };
  const toXY = (u: number, v: number): [number, number] => [
    2 * (u - geo.r.xMin) / (geo.r.xMax - geo.r.xMin) - 1,
    2 * (v - geo.r.yMin) / (geo.r.yMax - geo.r.yMin) - 1,
  ];

  // Points in model space: [cube position, class/role]
  const points = useMemo(() => {
    const at = (x: number[], h: number): V3 => [...toXY(...(geo.toUV(x) as [number, number])), toZ(h)];
    const cleanPts = train.X.map((x: number[], i: number) => ({
      p: at(x, isRegression ? train.y[i] : f(shownRaw, x)),
      classA: train.y[i] === 1,
      i,
    }));
    const poisonPts = (currentState?.poisonX ?? []).map((x: number[], i: number) => ({
      p: at(x, isRegression ? currentState.poisonY[i] : f(shownRaw, x)),
      i,
    }));
    return { cleanPts, poisonPts };
  }, [train, geo, shownRaw, currentState?.poisonX, currentState?.poisonY, zScale, isRegression, f]);

  const cleanBoundary = useMemo(() => (!isRegression && cleanSurface ? zeroCrossings(cleanSurface, N) : []), [cleanSurface, isRegression, N]);
  const poisonedBoundary = useMemo(() => (!isRegression && poisonedSurface ? zeroCrossings(poisonedSurface, N) : []), [poisonedSurface, isRegression, N]);

  // ── Camera ───────────────────────────────────────────────────────────
  const yaw = cam.yaw * Math.PI / 180, pitch = cam.pitch * Math.PI / 180;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const s = Math.min(width, height) * 0.3 * cam.zoom;
  const proj = ([x, y, z]: V3) => {
    const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
    return { x: width / 2 + s * x1, y: height / 2 - s * (z * cp + y1 * sp), depth: z * sp - y1 * cp };
  };
  const pts2 = (ps: V3[]) => ps.map(p => { const q = proj(p); return `${q.x.toFixed(1)},${q.y.toFixed(1)}`; }).join(' ');

  // ── Scene ────────────────────────────────────────────────────────────
  const zLo = isRegression ? -1 : -ZCLIP, zHi = isRegression ? 1 : ZCLIP;
  const grid = (i: number, j: number): [number, number] => [-1 + 2 * j / N, -1 + 2 * i / N];
  const items: Drawable[] = [];
  const line = (a: V3, b: V3, className: string, width = 1, dash?: string, key = '') => {
    const pa = proj(a), pb = proj(b);
    items.push({
      depth: (pa.depth + pb.depth) / 2,
      el: <line key={key} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} className={className} strokeWidth={width} strokeDasharray={dash} />,
    });
  };

  // Height-0 plane (classification): a light grid — where the surface crosses it, the class flips
  if (!isRegression) {
    for (let k = 0; k <= 4; k++) {
      const t = -1 + k / 2;
      line([t, -1, 0], [t, 1, 0], 'stroke-[var(--border)]', 0.75, undefined, `z0v${k}`);
      line([-1, t, 0], [1, t, 0], 'stroke-[var(--border)]', 0.75, undefined, `z0h${k}`);
    }
  }

  // Surface of the model on screen: filled faces
  const filled = isRegression ? poisonedSurface ?? cleanSurface : shownRaw === poisonedRaw ? poisonedSurface : cleanSurface;
  if (filled) {
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const hs = [filled[i][j], filled[i][j + 1], filled[i + 1][j + 1], filled[i + 1][j]];
        const cs: V3[] = [[...grid(i, j), toZ(hs[0])], [...grid(i, j + 1), toZ(hs[1])], [...grid(i + 1, j + 1), toZ(hs[2])], [...grid(i + 1, j), toZ(hs[3])]];
        const mean = (hs[0] + hs[1] + hs[2] + hs[3]) / 4;
        let cls: string, op: number;
        if (isRegression) {
          cls = poisonedSurface ? 'fill-[var(--color-attack)] stroke-[var(--color-attack)]' : 'fill-[var(--color-clean)] stroke-[var(--color-clean)]';
          op = 0.16;
        } else {
          cls = mean > 0 ? 'fill-[var(--data-class-a)] stroke-[var(--data-class-a)]' : 'fill-[var(--data-class-b)] stroke-[var(--data-class-b)]';
          op = 0.1 + 0.3 * Math.min(1, Math.abs(mean) / zScale.hi);
        }
        const d = cs.reduce((acc, c) => acc + proj(c).depth, 0) / 4;
        items.push({
          depth: d,
          el: <polygon key={`f${i}-${j}`} points={pts2(cs)} className={cls} fillOpacity={op} strokeOpacity={0.3} strokeWidth={0.5} />,
        });
      }
    }
  }

  // Clean prediction surface as a wireframe under the poisoned one (regression)
  if (isRegression && poisonedSurface && cleanSurface) {
    const step = N / 6;
    for (let k = 0; k <= N; k += step) {
      for (let m = 0; m < N; m++) {
        line([...grid(k, m), toZ(cleanSurface[k][m])], [...grid(k, m + 1), toZ(cleanSurface[k][m + 1])], 'stroke-[var(--color-clean)]', 1, '4 3', `cw${k}-${m}`);
        line([...grid(m, k), toZ(cleanSurface[m][k])], [...grid(m + 1, k), toZ(cleanSurface[m + 1][k])], 'stroke-[var(--color-clean)]', 1, '4 3', `ch${k}-${m}`);
      }
    }
  }

  // Decision boundaries on the height-0 plane (classification)
  const toCube = ([gi, gj]: number[]): V3 => [-1 + 2 * gj / N, -1 + 2 * gi / N, 0];
  cleanBoundary.forEach(([a, b], k) => line(toCube(a), toCube(b), 'stroke-[var(--color-clean)]', 2, '6 4', `cb${k}`));
  poisonedBoundary.forEach(([a, b], k) => line(toCube(a), toCube(b), 'stroke-[var(--color-attack)]', 2.5, undefined, `pb${k}`));

  // Data: stems down to the reference plane make heights readable
  const base = isRegression ? zLo : 0;
  points.cleanPts.forEach(({ p, classA, i }: any) => {
    const q = proj(p);
    const sel = selectedPoint?.type === 'clean' && selectedPoint.index === i;
    line(p, [p[0], p[1], base], 'stroke-[var(--muted-foreground)] opacity-30', 0.75, undefined, `s${i}`);
    items.push({
      depth: q.depth + 1e-3,
      el: (
        <circle
          key={`c${i}`} cx={q.x} cy={q.y} r={sel ? 5.5 : 3.5}
          className={`${isRegression || classA ? 'fill-[var(--data-class-a)]' : 'fill-[var(--data-class-b)]'} ${sel ? 'stroke-[var(--data-support)]' : 'stroke-[var(--background)]'} cursor-pointer`}
          strokeWidth={sel ? 2 : 0.75}
          onClick={(e) => { e.stopPropagation(); setConfig({ selectedPoint: { type: 'clean', index: i } }); }}
        />
      ),
    });
  });
  points.poisonPts.forEach(({ p, i }: any) => {
    const q = proj(p);
    const sel = selectedPoint?.type === 'poison' && selectedPoint.index === i;
    line(p, [p[0], p[1], base], 'stroke-[var(--data-poison)] opacity-60', 1, '3 2', `ps${i}`);
    items.push({
      depth: q.depth + 2e-3,
      el: (
        <g key={`p${i}`} className="cursor-pointer" onClick={(e) => { e.stopPropagation(); setConfig({ selectedPoint: { type: 'poison', index: i } }); }}>
          <circle cx={q.x} cy={q.y} r={6} className={`fill-[var(--data-poison)] ${sel ? 'stroke-[var(--data-support)]' : 'stroke-[var(--background)]'}`} strokeWidth={2} />
          <circle cx={q.x} cy={q.y} r={9} className="fill-none stroke-[var(--color-attack)] opacity-60 pointer-events-none" strokeWidth={1.5} />
        </g>
      ),
    });
  });

  items.sort((a, b) => a.depth - b.depth);

  // Bounding box and labels, always behind / on top respectively
  const corners: V3[] = [];
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [zLo, zHi]) corners.push([x, y, z]);
  const edges = corners.flatMap((a, i) => corners.slice(i + 1).filter(b => a.filter((v, k) => v !== b[k]).length === 1).map(b => [a, b]));
  const label = (p: V3, text: string, key: string, anchor: 'start' | 'middle' | 'end' = 'middle') => {
    const q = proj(p);
    return <text key={key} x={q.x} y={q.y} textAnchor={anchor} dominantBaseline="middle" className="fill-[var(--muted-foreground)] font-sans text-[11px] select-none pointer-events-none">{text}</text>;
  };
  // Height labels ride the vertical edge that is leftmost on screen; floor-axis labels the floor
  // edges nearest the viewer — so neither ends up floating inside the scene after a rotation
  const [zx, zy] = ([[-1, -1], [-1, 1], [1, -1], [1, 1]] as [number, number][])
    .reduce((best, c) => (proj([c[0], c[1], 0]).x < proj([best[0], best[1], 0]).x ? c : best));
  const out = (v: number) => v * 1.08;
  const uEdge = proj([0, -1, zLo]).depth > proj([0, 1, zLo]).depth ? -1 : 1;
  const vEdge = proj([-1, 0, zLo]).depth > proj([1, 0, zLo]).depth ? -1 : 1;
  const fmt = (v: number) => (Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(2));
  const zName = isRegression ? 'y' : 'f(x)';

  // ── Interaction ──────────────────────────────────────────────────────
  const onPointerDown = (e: React.PointerEvent) => {
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, yaw: cam.yaw, pitch: cam.pitch };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const next = { yaw: d.yaw + (e.clientX - d.x) * 0.4, pitch: Math.max(0, Math.min(90, d.pitch + (e.clientY - d.y) * 0.4)) };
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => setCam(c => ({ ...c, ...next })));
  };
  const onPointerUp = () => { drag.current = null; };
  const onWheel = (e: React.WheelEvent) => {
    const k = Math.exp(-e.deltaY * 0.0015);
    setCam(c => ({ ...c, zoom: Math.max(0.4, Math.min(4, c.zoom * k)) }));
  };

  return (
    <>
      <svg
        data-export="attack-canvas" width={width} height={height}
        className="w-full h-full block cursor-grab active:cursor-grabbing touch-none"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
        onWheel={onWheel} onDoubleClick={() => setCam(HOME)}
        onClick={() => { if (selectedPoint) setConfig({ selectedPoint: null }); }}
      >
        <g>
          {edges.map(([a, b], k) => {
            const pa = proj(a as V3), pb = proj(b as V3);
            return <line key={`e${k}`} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} className="stroke-[var(--border)]" strokeWidth={0.75} />;
          })}
        </g>
        <g>{items.map(d => d.el)}</g>
        <g>
          {label([0, uEdge * 1.18, zLo], geo.axes[0], 'ax')}
          {label([vEdge * 1.18, 0, zLo], geo.axes[1], 'ay')}
          {label([out(zx), out(zy), zHi + 0.14], zName, 'az', 'end')}
          {label([out(zx), out(zy), zHi], fmt(isRegression ? zScale.hi : ZCLIP * zScale.hi), 'zt', 'end')}
          {!isRegression && label([out(zx), out(zy), 0], '0', 'z0', 'end')}
          {label([out(zx), out(zy), zLo], fmt(isRegression ? zScale.lo : -ZCLIP * zScale.hi), 'zb', 'end')}
        </g>
      </svg>

      <div className="absolute bottom-4 right-4 glass-panel border border-border p-3 rounded-lg flex flex-col gap-1.5 shadow-lg z-10 pointer-events-none max-w-[240px]">
        {isRegression ? (
          <>
            <Key swatch="bg-data-class-a rounded-full" text="Training point (height = y)" />
            {poisonedSurface
              ? <><Key swatch="bg-attack/40" text="Poisoned fit" /><Key swatch="border-t-2 border-dashed border-clean" text="Clean fit" /></>
              : <Key swatch="bg-clean/40" text="Clean fit" />}
          </>
        ) : (
          <>
            <Key swatch="bg-data-class-a rounded-full" text="Class +1" />
            <Key swatch="bg-data-class-b rounded-full" text="Class −1" />
            <Key swatch="border-t-2 border-dashed border-clean" text="Clean boundary (f = 0)" />
            {poisonedSurface && <Key swatch="border-t-2 border-attack" text="Poisoned boundary" />}
          </>
        )}
        {currentState && <Key swatch="bg-attack rounded-full" text="Poison point" />}
        <p className="text-[10px] text-muted-foreground/70 leading-snug">
          {isRegression
            ? 'Floor: first two principal directions of the features. Surfaces: the models’ predictions there.'
            : `Height: decision value f(x) of the ${poisonedSurface ? 'poisoned' : 'clean'} model; the surface is its sign.`}
        </p>
      </div>
      <div className="absolute bottom-4 left-4 text-[11px] text-muted-foreground/70 z-10 pointer-events-none select-none">
        Drag to rotate · scroll to zoom · double-click to reset
      </div>
    </>
  );
}

function Key({ swatch, text }: { swatch: string; text: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className={`w-3 h-2.5 shrink-0 ${swatch}`} /> {text}
    </div>
  );
}

function boundsOf(X: number[][]) {
  const u = X.map(x => x[0]), v = X.map(x => x[1]);
  const pad = (a: number[]) => 0.05 * (Math.max(...a) - Math.min(...a) || 1);
  return {
    xMin: Math.min(...u) - pad(u), xMax: Math.max(...u) + pad(u),
    yMin: Math.min(...v) - pad(v), yMax: Math.max(...v) + pad(v),
  };
}

/** Zero-level segments of a (N+1)² grid, in fractional grid indices [i, j] (marching squares). */
function zeroCrossings(g: number[][], N: number): number[][][] {
  const segs: number[][][] = [];
  const t = (a: number, b: number) => a / (a - b);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const a = g[i][j], b = g[i][j + 1], c = g[i + 1][j + 1], d = g[i + 1][j];
      if (![a, b, c, d].every(Number.isFinite)) continue;
      const hits: number[][] = [];
      if ((a > 0) !== (b > 0)) hits.push([i, j + t(a, b)]);
      if ((b > 0) !== (c > 0)) hits.push([i + t(b, c), j + 1]);
      if ((d > 0) !== (c > 0)) hits.push([i + 1, j + t(d, c)]);
      if ((a > 0) !== (d > 0)) hits.push([i + t(a, d), j]);
      if (hits.length === 2) segs.push([hits[0], hits[1]]);
      else if (hits.length === 4) segs.push([hits[0], hits[3]], [hits[1], hits[2]]);
    }
  }
  return segs;
}
