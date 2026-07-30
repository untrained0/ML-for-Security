# Architecture — ML Security Attack Visualizer

## 1. App Flow and Architecture

### High-Level Flow

```mermaid
graph TD
    A["User Opens App"] --> B["Landing: Select Dataset + Attack Type"]
    B --> C["Canvas Renders Clean Model"]
    C --> D["User Configures Attack Parameters"]
    D --> E["User Clicks Play / Step"]
    E --> F["Attack Engine Runs 1 Iteration (Web Worker)"]
    F --> G["React State Updates via Zustand"]
    G --> H1["Canvas Redraws Boundary + Points (D3)"]
    G --> H2["Math Panel Updates w, b, α, loss (KaTeX)"]
    G --> H3["Timeline Charts Append Data Point (Recharts)"]
    G --> H4["Gradient Overlay Recomputes (optional)"]
    H1 --> I{"User Action?"}
    H2 --> I
    H3 --> I
    H4 --> I
    I -->|Step / Play| F
    I -->|Pause| J["Idle — Inspect State"]
    I -->|Reset| C
    I -->|Change Params| D
```

### State Management

```mermaid
graph LR
    subgraph Store["Zustand Store"]
        S1["dataset"]
        S2["cleanModel"]
        S3["poisonedModel"]
        S4["attackTrace[]"]
        S5["currentIteration"]
    end
    subgraph Engine["Attack Engine (Web Worker)"]
        W1["SVM Solver"]
        W2["Gradient Ascent"]
        W3["Line Search"]
    end
    subgraph UI["React Components"]
        U1["Canvas (D3.js)"]
        U2["MathPanel (KaTeX)"]
        U3["Timeline (Recharts)"]
        U4["Controls"]
    end
    U4 -->|dispatch| Engine
    Engine -->|postMessage| Store
    Store -->|useStore hook| U1
    Store -->|useStore hook| U2
    Store -->|useStore hook| U3
```

The attack engine runs inside a **Web Worker** so heavy matrix math never blocks the UI thread. State updates are emitted via `postMessage` and dispatched through Zustand.

---

## 2. Folder and File Structure

```
ml-security-viz/
├── package.json
├── next.config.mjs
├── tsconfig.json (optional — using JS)
│
├── public/
│   └── favicon.svg
│
├── src/
│   ├── app/
│   │   ├── layout.js              # Root layout — fonts, metadata, global CSS
│   │   ├── page.js                # Main page — assembles all panels
│   │   └── globals.css            # Global design tokens, resets, typography
│   │
│   ├── components/
│   │   ├── AppHeader.js           # Top navigation bar
│   │   ├── AppHeader.module.css
│   │   ├── Canvas.js              # D3-based scatter + decision boundary
│   │   ├── Canvas.module.css
│   │   ├── ControlPanel.js        # Left sidebar — dataset, kernel, attack params
│   │   ├── ControlPanel.module.css
│   │   ├── MathPanel.js           # Right sidebar — math state inspector
│   │   ├── MathPanel.module.css
│   │   ├── PlaybackBar.js         # Play/Pause/Step controls + metrics
│   │   ├── PlaybackBar.module.css
│   │   ├── Timeline.js            # Loss/error charts (Recharts)
│   │   ├── Timeline.module.css
│   │   ├── DatasetChip.js         # Dataset selector chip
│   │   └── MetricCard.js          # Reusable metric display card
│   │
│   ├── engine/
│   │   ├── worker.js              # Web Worker entry — receives commands
│   │   ├── svm.js                 # SVM solver (SMO algorithm)
│   │   ├── poisoning.js           # Gradient-ascent poisoning (Biggio 2012)
│   │   ├── kernels.js             # Kernel functions (linear, RBF, poly)
│   │   ├── linalg.js              # Matrix ops (zero dependencies)
│   │   └── datasets.js            # Built-in dataset generators
│   │
│   ├── store/
│   │   └── useStore.js            # Zustand store — single source of truth
│   │
│   ├── hooks/
│   │   ├── useWorker.js           # Web Worker lifecycle hook
│   │   └── usePlayback.js         # Auto-play / step-through logic
│   │
│   └── utils/
│       ├── colorScale.js          # Color palette utilities
│       ├── mathFormat.js          # Number formatting for display
│       └── exportUtils.js         # PNG / SVG / JSON export
│
└── docs/
    ├── prd.md
    ├── architecture.md
    ├── rules.md
    ├── phases.md
    ├── design.md
    └── memory.md
```

---

## 3. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Framework** | Next.js 15 (App Router) | Modern React framework with built-in routing, SSR, and optimized builds |
| **UI Library** | React 19 | Component-based architecture; hooks for state and side effects |
| **Styling** | CSS Modules + globals.css | Scoped styles per component; global design tokens via custom properties |
| **State** | Zustand | Lightweight, hook-based store; no boilerplate like Redux |
| **Visualization** | D3.js v7 | Scatter plots, contour rendering, vector fields — used imperatively via `useRef` |
| **Charts** | Recharts | React-native charting library built on D3; great for timeline panels |
| **Math Rendering** | KaTeX + react-katex | Fast LaTeX rendering in React components |
| **Heavy Compute** | Web Workers | Run SVM solver and gradient ascent off the main thread |
| **Linear Algebra** | Custom `linalg.js` | Small, focused matrix operations for 2D problems |
| **Fonts** | next/font (Inter, JetBrains Mono) | Optimized font loading via Next.js |
| **Package Manager** | npm | Standard Node.js package management |

### Key Dependencies

```json
{
  "dependencies": {
    "next": "^15",
    "react": "^19",
    "react-dom": "^19",
    "zustand": "^5",
    "d3": "^7",
    "recharts": "^2",
    "katex": "^0.16",
    "react-katex": "^3"
  }
}
```

### No Backend Required

All computation is client-side. The SVM solver and poisoning attack run in pure JavaScript inside a Web Worker. Next.js is used purely for its frontend capabilities:

- Optimized builds and code splitting
- CSS Modules for scoped styling
- Font optimization
- Easy deployment to Vercel / GitHub Pages
- Hot module replacement during development
