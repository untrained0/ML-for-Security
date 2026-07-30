# Rules — ML Security Attack Visualizer

## What to Use

### Framework & Libraries
- **Next.js 15** with App Router — pages, layouts, metadata
- **React 19** — functional components with hooks only (no class components)
- **Zustand 5** — lightweight state management via `useStore` hook
- **D3.js v7** — used imperatively via `useRef` + `useEffect` for the scatter/boundary canvas
- **Recharts v2** — React-native charts for the timeline panel
- **KaTeX + react-katex** — LaTeX math rendering in the inspector panel
- **CSS Modules** (`.module.css`) — scoped styles per component
- **`globals.css`** — design tokens, resets, and shared styles via CSS custom properties
- **Web Workers** — all heavy computation (SVM training, gradient ascent) runs off the main thread
- **next/font** — optimized loading for Inter and JetBrains Mono

### Code Patterns
- **Functional components only** — no class components
- **Custom hooks** for reusable logic (`useWorker`, `usePlayback`)
- **Zustand store** as single source of truth — no prop drilling
- **Pure functions** for all math (kernel, gradient, loss) — no side effects
- **`'use client'`** directive on all interactive components (Next.js App Router requirement)
- **Web Worker message protocol**: `{ type: 'COMMAND', payload: {...} }` / `{ type: 'STATE_UPDATE', payload: {...} }`
- **Error boundaries**: wrap Worker message handlers in try/catch

### Naming Conventions
- Component files: `PascalCase.js` with matching `PascalCase.module.css`
- Engine files: `camelCase.js`
- Hooks: `use*.js`
- CSS classes: camelCase in CSS Modules (auto-scoped)
- JS variables/functions: `camelCase`
- JS components: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

---

## What to Avoid

### Libraries & Frameworks — DO NOT USE
| ❌ Avoid | Why |
|---|---|
| TailwindCSS | User did not request it; CSS Modules + globals give full control |
| jQuery | Outdated; React handles the DOM |
| math.js / numeric.js | Too heavy; custom `linalg.js` covers our 2D needs |
| TensorFlow.js | We implement SVM from scratch for educational transparency |
| Three.js / WebGL | 2D only; D3 + SVG/Canvas is the right tool |
| Redux / MobX | Overkill; Zustand is simpler and sufficient |
| Styled Components / Emotion | CSS Modules are simpler with Next.js |
| Any CSS preprocessor (SASS, LESS) | CSS custom properties + CSS Modules cover our needs |
| Lodash / Underscore | Modern JS has everything we need |

### Patterns — DO NOT USE
| ❌ Avoid | Why |
|---|---|
| Class components | Hooks-only codebase |
| `getServerSideProps` / `getStaticProps` | App Router uses server components by default; we use `'use client'` |
| `eval()` or `new Function()` | Security risk |
| Global mutable state outside Zustand | All state flows through the store |
| `dangerouslySetInnerHTML` for user data | XSS risk; use KaTeX's safe renderer |
| CSS `!important` | Fix specificity properly with CSS Modules |
| Inline styles in components | Use CSS Modules classes |
| Blocking the main thread with math | All matrix ops go to the Web Worker |
| `useEffect` for data fetching | No server data in this app |

### Error Handling
- **Worker errors**: Catch in `worker.onerror`, surface via Zustand error state → toast UI
- **Invalid user input**: Validate in the control panel before dispatching
- **Numerical instability**: Check for NaN/Infinity in all matrix operations
- **Dataset too large**: Warn if >5,000 points; suggest downsampling

### Mathematical Accuracy
- Every formula must match the paper's notation exactly
- Include equation references (e.g., "Eq. 7 from Biggio et al. 2012")
- No approximations without clear documentation in code and UI tooltips
