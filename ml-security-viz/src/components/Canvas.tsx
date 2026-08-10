'use client';
import { useRef, useEffect, useMemo, useState } from 'react';
import * as d3 from 'd3';
import useStore from '@/store/useStore';
import { createKernel } from '@/engine/architectures/biggio2012/kernels';
import { trainSVM, predict } from '@/engine/architectures/biggio2012/model';
import { getAlgorithm } from '@/engine/architectures';
import PointInspector from './PointInspector';
import PoisonImagePanel from './PoisonImagePanel';

/**
 * Canvas component — D3 SVG scatter plot with decision boundary / regression line
 * Features infinite pan/zoom. Adapts rendering based on active algorithm's modelType.
 */
export default function Canvas({ currentState }: { currentState?: any }) {
  const { 
    dataset, cleanModel, kernelType, kernelGamma, svmC, 
    showGradients, showHeatmap, heatmapData, attackTrace, currentIteration,
    setHoveredCanvasPoint, selectedPoint, setConfig, activeAlgorithm, theme
  } = useStore();
  const alg = getAlgorithm(activeAlgorithm);
  const isRegression = alg.modelType === 'regression';
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const { width, height } = dimensions;
  
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      for (let entry of entries) {
        if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
          setDimensions({
            width: entry.contentRect.width,
            height: entry.contentRect.height
          });
        }
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);

  // Data range for mapping
  const range = useMemo(() => ({ xMin: -2.5, xMax: 2.5, yMin: -2.5, yMax: 2.5 }), []);

  const toSVG = (x: number, y: number) => ([
    ((x - range.xMin) / (range.xMax - range.xMin)) * width,
    height - ((y - range.yMin) / (range.yMax - range.yMin)) * height,
  ]);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 50])
      .on('zoom', (e) => {
        setTransform(e.transform);
      });
    svg.call(zoomBehavior);
    
    return () => {
      svg.on('.zoom', null);
    };
  }, []);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const rawX = e.clientX - rect.left;
    const rawY = e.clientY - rect.top;
    
    // Invert the D3 transform to get the logical SVG coordinates
    const [svgX, svgY] = transform.invert([rawX, rawY]);
    
    const dataX = (svgX / width) * (range.xMax - range.xMin) + range.xMin;
    const dataY = ((height - svgY) / height) * (range.yMax - range.yMin) + range.yMin;
    
    setHoveredCanvasPoint([dataX, dataY]);
  };

  const handleMouseLeave = () => {
    setHoveredCanvasPoint(null);
  };

  // Compute decision boundary contour (classification) or regression lines (regression)
  const boundaryPaths = useMemo(() => {
    if (!dataset || isRegression) return { clean: null, poisoned: null };
    const kernelFn = createKernel({ type: kernelType, gamma: kernelGamma });

    const computeBoundary = (X: number[][], y: number[]) => {
      try {
        const model = trainSVM(X, y, kernelFn, svmC);
        const res = 60;
        const grid = [];
        const bgRange = { xMin: -6, xMax: 6, yMin: -6, yMax: 6 };
        for (let i = 0; i <= res; i++) {
          const row = [];
          for (let j = 0; j <= res; j++) {
            const x = bgRange.xMin + (bgRange.xMax - bgRange.xMin) * j / res;
            const yCoord = bgRange.yMin + (bgRange.yMax - bgRange.yMin) * i / res;
            row.push(predict(model, [x, yCoord]));
          }
          grid.push(row);
        }
        return marchingSquares(grid, res, bgRange, range, width, height);
      } catch { return null; }
    };

    const clean = computeBoundary(dataset.train.X, dataset.train.y);

    let poisoned = null;
    if (currentState && currentState.poisonX) {
      const augX = [...dataset.train.X, ...currentState.poisonX];
      const augY = [...dataset.train.y, ...currentState.poisonY];
      poisoned = computeBoundary(augX, augY);
    }

    return { clean, poisoned };
  }, [dataset, currentState, kernelType, kernelGamma, svmC, range, width, height, isRegression]);

  // Compute regression line paths (only in regression mode)
  const regressionLines = useMemo(() => {
    if (!dataset || !isRegression) return { clean: null, poisoned: null };
    
    const computeLinePath = (model: any) => {
      if (!model || !model.theta) return null;
      const steps = 200;
      const points: string[] = [];
      for (let i = 0; i <= steps; i++) {
        const x = range.xMin + (range.xMax - range.xMin) * i / steps;
        // theta = [w0, w1, ..., bias]
        // For 1D: y = theta[0] * x + theta[1]
        const yPred = model.theta[0] * x + model.theta[model.theta.length - 1];
        const [sx, sy] = toSVG(x, yPred);
        points.push(`${i === 0 ? 'M' : 'L'} ${sx.toFixed(2)},${sy.toFixed(2)}`);
      }
      return points.join(' ');
    };

    const cleanPath = cleanModel ? computeLinePath(cleanModel) : null;
    const poisonedPath = currentState?.poisonedModel ? computeLinePath(currentState.poisonedModel) : null;

    return { clean: cleanPath, poisoned: poisonedPath };
  }, [dataset, cleanModel, currentState, isRegression, range, width, height]);

  const gridLines = useMemo(() => {
    const [xMinSvg, yMinSvg] = transform.invert([0, 0]);
    const [xMaxSvg, yMaxSvg] = transform.invert([width, height]);

    const dataXMin = (xMinSvg / width) * (range.xMax - range.xMin) + range.xMin;
    const dataXMax = (xMaxSvg / width) * (range.xMax - range.xMin) + range.xMin;
    const dataYMax = ((height - yMinSvg) / height) * (range.yMax - range.yMin) + range.yMin; 
    const dataYMin = ((height - yMaxSvg) / height) * (range.yMax - range.yMin) + range.yMin;

    const lines = [];
    const startX = Math.floor(dataXMin);
    const endX = Math.ceil(dataXMax);
    const startY = Math.floor(dataYMin);
    const endY = Math.ceil(dataYMax);

    for (let x = startX; x <= endX; x++) {
      lines.push({ type: 'v', val: x });
    }
    for (let y = startY; y <= endY; y++) {
      lines.push({ type: 'h', val: y });
    }
    return lines;
  }, [transform, width, height, range]);

  // Precompute trajectories for poison points up to current iteration
  const poisonTrajectories = useMemo(() => {
    if (!attackTrace || attackTrace.length === 0 || currentIteration === 0) return [];
    
    const numPoison = attackTrace[0].poisonX.length;
    const trajectories: {x: number, y: number}[][] = Array.from({ length: numPoison }, () => []);
    
    for (let iter = 0; iter <= currentIteration; iter++) {
      const state = attackTrace[iter];
      if (!state.poisonX) continue;
      for (let p = 0; p < numPoison; p++) {
        if (state.poisonX[p]) {
          trajectories[p].push({ x: state.poisonX[p][0], y: state.poisonX[p][1] });
        }
      }
    }
    
    return trajectories;
  }, [attackTrace, currentIteration]);

  // Precompute heatmap colors
  const heatmapCells = useMemo(() => {
    if (!heatmapData || !heatmapData.data) return null;
    const values = heatmapData.data.map((d: any) => d.value);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    
    // Size of each cell in SVG space
    const stepX = (heatmapData.max - heatmapData.min) / (heatmapData.resolution - 1);
    
    return heatmapData.data.map((d: any) => {
      // Normalize to 0-1
      const norm = maxVal > minVal ? (d.value - minVal) / (maxVal - minVal) : 0.5;
      // Interpolate from blue (low) to red (high)
      const hue = 240 - (norm * 240); 
      
      const [cx, cy] = toSVG(d.x, d.y);
      const [nx, ny] = toSVG(d.x + stepX, d.y + stepX);
      const width = Math.abs(nx - cx);
      const height = Math.abs(ny - cy);
      
      return {
        ...d,
        svgX: cx - width/2,
        svgY: cy - height/2,
        width,
        height,
        // Deep on dark, pastel on light — either way the field stays behind
        // the marks instead of competing with them.
        color: `hsl(${hue}, ${theme === 'dark' ? '80%, 30%' : '70%, 62%'})`
      };
    });
  }, [heatmapData, theme]);

  const train = dataset?.train;
  const invK = 1 / transform.k;

  return (
    <div className="relative w-full h-full overflow-hidden" ref={containerRef}>
      {!dataset && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground/70 text-sm pointer-events-none z-20 bg-background">
          <div className="w-7 h-7 border-[3px] border-border border-t-primary rounded-full animate-spin" />
          <p>Generating dataset...</p>
        </div>
      )}
      <svg ref={svgRef} width={width} height={height} className="w-full h-full block cursor-grab active:cursor-grabbing" onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave} onClick={() => { if (selectedPoint) setConfig({ selectedPoint: null }); }}>
        {dataset && (
          <g transform={`translate(${transform.x},${transform.y}) scale(${transform.k})`}>
            {/* Grid lines */}
          <g>
            {gridLines.map((line) => {
              if (line.type === 'v') {
                const [sx] = toSVG(line.val, 0);
                return <line key={`v-${line.val}`} x1={sx} y1={-100000} x2={sx} y2={100000} className="stroke-[var(--data-grid)]" strokeWidth={0.5 * invK} />;
              } else {
                const [, sy] = toSVG(0, line.val);
                return <line key={`h-${line.val}`} x1={-100000} y1={sy} x2={100000} y2={sy} className="stroke-[var(--data-grid)]" strokeWidth={0.5 * invK} />;
              }
            })}
          </g>

          {/* Objective Heatmap */}
          {showHeatmap && heatmapCells && (
            <g className="opacity-20 pointer-events-none">
              {heatmapCells.map((cell: any, i: number) => (
                <rect 
                  key={`heatmap-${i}`} 
                  x={cell.svgX} y={cell.svgY} 
                  width={cell.width} height={cell.height} 
                  fill={cell.color} 
                />
              ))}
            </g>
          )}

          {/* Heatmap Gradient Quiver Plot */}
          {showHeatmap && heatmapCells && (
            <g className="opacity-40 pointer-events-none">
              {heatmapCells.map((cell: any, i: number) => {
                const [cx, cy] = toSVG(cell.x, cell.y);
                const gNorm = Math.sqrt(cell.gradX * cell.gradX + cell.gradY * cell.gradY);
                if (gNorm < 1e-8) return null;
                const arrowLen = 15 * invK;
                const dx = (cell.gradX / gNorm) * arrowLen;
                const dy = -(cell.gradY / gNorm) * arrowLen;
                return (
                  <g key={`quiver-${i}`}>
                    <line x1={cx} y1={cy} x2={cx + dx} y2={cy + dy} className="stroke-[var(--muted-foreground)]" strokeWidth={1 * invK} />
                    <polygon
                      points={`0,0 ${4 * invK},${1.5 * invK} 0,${3 * invK}`}
                      className="fill-[var(--muted-foreground)]"
                      transform={`translate(${cx + dx}, ${cy + dy}) rotate(${Math.atan2(dy, dx) * 180 / Math.PI}) translate(${-4 * invK}, ${-1.5 * invK})`} 
                    />
                  </g>
                );
              })}
            </g>
          )}

          {/* Clean decision boundary (classification) */}
          {!isRegression && boundaryPaths.clean && (
            <path d={boundaryPaths.clean} className="fill-none stroke-[var(--color-clean)] opacity-70 [filter:drop-shadow(0_0_4px_var(--glow-clean))]" strokeWidth={2.5 * invK} strokeDasharray={`${6 * invK},${4 * invK}`} />
          )}

          {/* Poisoned decision boundary (classification) */}
          {!isRegression && boundaryPaths.poisoned && (
            <path d={boundaryPaths.poisoned} className="fill-none stroke-[var(--color-attack)] opacity-90 [filter:drop-shadow(0_0_6px_var(--glow-attack))] transition-[d] duration-400 ease-in-out" strokeWidth={2.5 * invK} />
          )}

          {/* Clean regression line */}
          {isRegression && regressionLines.clean && (
            <path d={regressionLines.clean} className="fill-none stroke-[var(--color-clean)] opacity-80 [filter:drop-shadow(0_0_4px_var(--glow-clean))]" strokeWidth={2.5 * invK} strokeDasharray={`${6 * invK},${4 * invK}`} />
          )}

          {/* Poisoned regression line */}
          {isRegression && regressionLines.poisoned && (
            <path d={regressionLines.poisoned} className="fill-none stroke-[var(--color-attack)] opacity-90 [filter:drop-shadow(0_0_6px_var(--glow-attack))] transition-[d] duration-400 ease-in-out" strokeWidth={2.5 * invK} />
          )}

          {/* Residual whiskers (regression) */}
          {isRegression && cleanModel && cleanModel.theta && train.X.map((pt: number[], i: number) => {
            const x = pt[0];
            const yActual = train.y[i];
            const yPred = cleanModel.theta[0] * x + cleanModel.theta[cleanModel.theta.length - 1];
            const [cx, cy] = toSVG(x, yActual);
            const [, predY] = toSVG(x, yPred);
            return (
              <line key={`resid-${i}`} x1={cx} y1={cy} x2={cx} y2={predY} className="stroke-[var(--muted-foreground)] opacity-40 pointer-events-none" strokeWidth={1 * invK} strokeDasharray={`${2 * invK},${2 * invK}`} />
            );
          })}

          {/* Training points */}
          {train.X.map((pt: number[], i: number) => {
            const isSelected = selectedPoint?.type === 'clean' && selectedPoint?.index === i;
            if (isRegression) {
              // Regression: 1D x, y is continuous — use gradient color
              const x = pt[0];
              const yVal = train.y[i];
              const [cx, cy] = toSVG(x, yVal);
              const t = (yVal - range.yMin) / (range.yMax - range.yMin);
              const hue = 220 - t * 180; // blue(low) -> orange(high)
              return (
                <g key={`train-${i}`}>
                  {isSelected && (
                    <circle cx={cx} cy={cy} r={8 * invK} className="fill-none stroke-[var(--data-support)] animate-pulse pointer-events-none" strokeWidth={2 * invK} />
                  )}
                  <circle
                    cx={cx} cy={cy} r={4 * invK}
                    fill={`hsl(${hue}, 75%, 55%)`}
                    className={`transition-all duration-200 hover:brightness-125 hover:cursor-pointer ${isSelected ? 'stroke-[var(--data-support)]' : 'stroke-[var(--background)]'}`}
                    strokeWidth={(isSelected ? 2 : 1) * invK}
                    onClick={(e) => {
                      e.stopPropagation();
                      setConfig({ selectedPoint: { type: 'clean', index: i } });
                    }}
                  />
                </g>
              );
            } else {
              // Classification: binary color
              const [cx, cy] = toSVG(pt[0], pt[1]);
              const isClassA = train.y[i] === 1;
              return (
                <g key={`train-${i}`}>
                  {isSelected && (
                    <circle cx={cx} cy={cy} r={8 * invK} className="fill-none stroke-[var(--data-support)] animate-pulse pointer-events-none" strokeWidth={2 * invK} />
                  )}
                  <circle
                    cx={cx} cy={cy} r={4 * invK}
                    className={`transition-all duration-200 hover:brightness-125 hover:cursor-pointer ${isClassA ? 'fill-[var(--data-class-a)]' : 'fill-[var(--data-class-b)]'} ${isSelected ? 'stroke-[var(--data-support)]' : 'stroke-[var(--background)]'}`}
                    strokeWidth={(isSelected ? 2 : 1) * invK}
                    onClick={(e) => {
                      e.stopPropagation();
                      setConfig({ selectedPoint: { type: 'clean', index: i } });
                    }}
                  />
                </g>
              );
            }
          })}

          {/* Support vectors (classification only) */}
          {!isRegression && ((currentState && currentState.poisonedModel) || cleanModel) && ((currentState ? currentState.poisonedModel : cleanModel).supportVectors?.map((sv: any, i: number) => {
            const [cx, cy] = toSVG(sv.point[0], sv.point[1]);
            return (
              <circle
                key={`sv-${i}`}
                cx={cx} cy={cy} r={8 * invK}
                className="fill-none stroke-[var(--data-support)] opacity-70 pointer-events-none transition-all duration-300"
                strokeWidth={2 * invK}
              />
            );
          }))}

          {/* Gained Support Vectors (Flash Yellow) — classification only */}
          {!isRegression && currentState && currentState.gainedSVs && currentState.gainedSVs.map((pt: number[], i: number) => {
            const [cx, cy] = toSVG(pt[0], pt[1]);
            return (
              <circle
                key={`gained-sv-${currentState.iteration}-${i}`}
                cx={cx} cy={cy} r={12 * invK}
                className="fill-none stroke-[var(--data-support)] pointer-events-none animate-[ping_1s_ease-out_1]"
                strokeWidth={3 * invK}
              />
            );
          })}

          {/* Lost Support Vectors (Flash Red) — classification only */}
          {!isRegression && currentState && currentState.lostSVs && currentState.lostSVs.map((pt: number[], i: number) => {
            const [cx, cy] = toSVG(pt[0], pt[1]);
            return (
              <circle
                key={`lost-sv-${currentState.iteration}-${i}`}
                cx={cx} cy={cy} r={12 * invK}
                className="fill-none stroke-red-500 pointer-events-none animate-[ping_1s_ease-out_1]"
                strokeWidth={3 * invK}
              />
            );
          })}

          {/* Poison Point Trajectories */}
          {poisonTrajectories.map((traj, pIdx) => {
            if (traj.length < 2) return null;
            const pathD = traj.map((pt, i) => {
              const [cx, cy] = toSVG(pt.x, pt.y);
              return `${i === 0 ? 'M' : 'L'} ${cx} ${cy}`;
            }).join(' ');
            
            return (
              <path
                key={`traj-${pIdx}`}
                d={pathD}
                fill="none"
                stroke="var(--data-poison)"
                strokeWidth={1.5 * invK}
                strokeDasharray={`${4 * invK},${4 * invK}`}
                className="opacity-50 pointer-events-none"
              />
            );
          })}

          {/* Poison points */}
          {currentState && currentState.poisonX && currentState.poisonX.map((pt: number[], i: number) => {
            // For regression, poison points are 1D: [x], and poisonY[i] is the y coordinate
            const px = pt[0];
            const py = isRegression ? currentState.poisonY[i] : pt[1];
            const [cx, cy] = toSVG(px, py);
            const isSelected = selectedPoint?.type === 'poison' && selectedPoint?.index === i;
            return (
              <g key={`poison-${i}`}>
                {isSelected && (
                  <circle cx={cx} cy={cy} r={12 * invK} className="fill-none stroke-[var(--data-support)] animate-pulse pointer-events-none" strokeWidth={2 * invK} />
                )}
                <circle
                  cx={cx} cy={cy} r={6 * invK}
                  className={`fill-[var(--data-poison)] hover:cursor-pointer hover:brightness-125 transition-all duration-200 ${isSelected ? 'stroke-[var(--data-support)]' : 'stroke-[var(--background)]'}`}
                  strokeWidth={2 * invK}
                  onClick={(e) => {
                    e.stopPropagation();
                    setConfig({ selectedPoint: { type: 'poison', index: i } });
                  }}
                />
                <circle
                  cx={cx} cy={cy} r={8 * invK}
                  className={`fill-none stroke-[var(--color-attack)] pointer-events-none ${i === currentState.poisonX.length - 1 ? 'animate-[poisonPulseCanvas_2s_infinite]' : 'opacity-60'}`}
                  strokeWidth={2 * invK}
                />
                <text x={cx} y={cy - (14 * invK)} className="fill-[var(--color-attack)] font-bold select-none pointer-events-none" style={{ textAnchor: 'middle', fontSize: `${10 * invK}px` }}>☠</text>
              </g>
            );
          })}

          {/* Gradient arrows */}
          {showGradients && currentState && currentState.gradients && currentState.gradients.map((grad: number[], i: number) => {
            if (!currentState.poisonX[i]) return null;
            const pt = currentState.poisonX[i];
            const [x1, y1] = toSVG(pt[0], pt[1]);
            const gNorm = Math.sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
            if (gNorm < 1e-8) return null;
            
            // Maintain constant arrow length in screen space
            const arrowLen = 30 * invK;
            const dx = (grad[0] / gNorm) * arrowLen;
            const dy = -(grad[1] / gNorm) * arrowLen; // flip y for SVG
            return (
              <g key={`grad-${i}`}>
                <line
                  x1={x1} y1={y1} x2={x1 + dx} y2={y1 + dy}
                  className="stroke-[var(--data-gradient)] opacity-60 transition-all duration-300 pointer-events-none"
                  strokeWidth={1.5 * invK}
                />
                <polygon 
                  points={`0,0 ${8 * invK},${3 * invK} 0,${6 * invK}`} 
                  fill="var(--data-gradient)" 
                  transform={`translate(${x1 + dx}, ${y1 + dy}) rotate(${Math.atan2(dy, dx) * 180 / Math.PI}) translate(${-8 * invK}, ${-3 * invK})`} 
                  className="opacity-60 pointer-events-none"
                />
              </g>
            );
          })}
        </g>
        )}
      </svg>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 glass-panel border border-border p-3 rounded-lg flex flex-col gap-2 shadow-lg z-10 pointer-events-none">
        {isRegression ? (
          <>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'hsl(220, 75%, 55%)' }} /> Low y
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'hsl(40, 75%, 55%)' }} /> High y
            </div>
          </>
        ) : (
          <>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="w-2.5 h-2.5 rounded-full bg-data-class-a" /> Class +1
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="w-2.5 h-2.5 rounded-full bg-data-class-b" /> Class −1
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="w-2.5 h-2.5 rounded-full border-2 border-[var(--data-support)]" /> Support Vector
            </div>
          </>
        )}
        {currentState && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="w-2.5 h-2.5 rounded-full bg-attack" /> Poison Point
          </div>
        )}
        {(boundaryPaths.clean || regressionLines.clean) && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="w-4 h-[2px] bg-clean opacity-70" /> {isRegression ? 'Clean Fit' : 'Clean Boundary'}
          </div>
        )}
        {(boundaryPaths.poisoned || regressionLines.poisoned) && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="w-4 h-[2px] bg-attack" /> {isRegression ? 'Poisoned Fit' : 'Poisoned Boundary'}
          </div>
        )}
      </div>

      {/* Iteration badge */}
      {currentState && (
        <div className="absolute top-4 left-4 bg-secondary border border-border px-3 py-1.5 rounded-md font-mono text-sm text-primary shadow z-10 pointer-events-none">
          Iteration {currentState.iteration}
        </div>
      )}

      <PointInspector />
      <PoisonImagePanel currentState={currentState} />
    </div>
  );
}

/**
 * Simple marching squares to extract the zero-contour (decision boundary)
 */
function marchingSquares(grid: number[][], res: number, gridRange: any, renderRange: any, w: number, h: number) {
  const segments: number[][][] = [];
  const lerp = (a: number, b: number, va: number, vb: number) => a + (0 - va) / (vb - va) * (b - a);

  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      const tl = grid[i][j], tr = grid[i][j + 1];
      const bl = grid[i + 1][j], br = grid[i + 1][j + 1];

      const x0 = gridRange.xMin + (gridRange.xMax - gridRange.xMin) * j / res;
      const x1 = gridRange.xMin + (gridRange.xMax - gridRange.xMin) * (j + 1) / res;
      const y0 = gridRange.yMin + (gridRange.yMax - gridRange.yMin) * i / res;
      const y1 = gridRange.yMin + (gridRange.yMax - gridRange.yMin) * (i + 1) / res;

      const toS = (x: number, y: number) => [
        ((x - renderRange.xMin) / (renderRange.xMax - renderRange.xMin)) * w,
        h - ((y - renderRange.yMin) / (renderRange.yMax - renderRange.yMin)) * h,
      ];

      const idx = (tl > 0 ? 8 : 0) | (tr > 0 ? 4 : 0) | (br > 0 ? 2 : 0) | (bl > 0 ? 1 : 0);

      const top = toS(lerp(x0, x1, tl, tr), y0);
      const right = toS(x1, lerp(y0, y1, tr, br));
      const bottom = toS(lerp(x0, x1, bl, br), y1);
      const left = toS(x0, lerp(y0, y1, tl, bl));

      const addSeg = (a: number[], b: number[]) => segments.push([a, b]);

      switch (idx) {
        case 1: case 14: addSeg(left, bottom); break;
        case 2: case 13: addSeg(bottom, right); break;
        case 3: case 12: addSeg(left, right); break;
        case 4: case 11: addSeg(top, right); break;
        case 5: addSeg(top, left); addSeg(bottom, right); break;
        case 6: case 9: addSeg(top, bottom); break;
        case 7: case 8: addSeg(top, left); break;
        case 10: addSeg(top, right); addSeg(left, bottom); break;
      }
    }
  }

  if (segments.length === 0) return null;

  // Build continuous paths
  const paths: number[][][] = [];
  let currentPath: number[][] = [...segments[0]];
  segments.splice(0, 1);

  while (segments.length > 0) {
    let merged = false;
    const endPoint = currentPath[currentPath.length - 1];

    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      if (Math.abs(seg[0][0] - endPoint[0]) < 1e-3 && Math.abs(seg[0][1] - endPoint[1]) < 1e-3) {
        currentPath.push(seg[1]);
        segments.splice(i, 1);
        merged = true;
        break;
      }
      if (Math.abs(seg[1][0] - endPoint[0]) < 1e-3 && Math.abs(seg[1][1] - endPoint[1]) < 1e-3) {
        currentPath.push(seg[0]);
        segments.splice(i, 1);
        merged = true;
        break;
      }
    }

    if (!merged) {
      paths.push(currentPath);
      currentPath = [...segments[0]];
      segments.splice(0, 1);
    }
  }
  paths.push(currentPath);

  return paths.map(path => {
    return 'M ' + path.map(p => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' L ');
  }).join(' ');
}
