# Design System — ML Security Attack Visualizer

## 1. Color & Theme

### Design Philosophy

The design draws inspiration from **cybersecurity dashboards** and **scientific visualization tools**. The aesthetic is **dark, precise, and data-dense** — like a control room for analyzing adversarial attacks. Colors are chosen for:

1. **High contrast** on dark backgrounds (accessibility)
2. **Semantic meaning** (red = danger/attack, green = safe/clean, blue = neutral/data)
3. **Scientific credibility** (muted, professional palette — not "gamer" neon)

### Color Palette

```css
:root {
  /* ── Background Layers ── */
  --bg-primary:       #0a0e17;      /* Deep navy-black — main background */
  --bg-secondary:     #111827;      /* Slightly lighter — panels, cards */
  --bg-tertiary:      #1a2332;      /* Elevated surfaces — dropdowns, modals */
  --bg-hover:         #1f2d3d;      /* Hover state for interactive surfaces */
  --bg-active:        #263445;      /* Active/selected state */

  /* ── Accent Colors ── */
  --accent-primary:   #6366f1;      /* Indigo — primary actions, links */
  --accent-hover:     #818cf8;      /* Lighter indigo — hover state */
  --accent-glow:      rgba(99, 102, 241, 0.15); /* Glow for focused elements */

  /* ── Semantic Colors ── */
  --color-clean:      #10b981;      /* Emerald — clean model, safe state */
  --color-clean-dim:  rgba(16, 185, 129, 0.2);
  --color-attack:     #ef4444;      /* Red — attack, poisoned, danger */
  --color-attack-dim: rgba(239, 68, 68, 0.2);
  --color-warning:    #f59e0b;      /* Amber — warnings, caution */
  --color-info:       #3b82f6;      /* Blue — informational */

  /* ── Data Visualization ── */
  --data-class-a:     #6366f1;      /* Indigo — class +1 */
  --data-class-b:     #f97316;      /* Orange — class -1 */
  --data-poison:      #ef4444;      /* Red — poisoned points */
  --data-support:     #fbbf24;      /* Yellow ring — support vectors */
  --data-boundary:    #e2e8f0;      /* Light gray — decision boundary */
  --data-gradient:    #8b5cf6;      /* Purple — gradient arrows */
  --data-contour-low: #1e3a5f;      /* Dark blue — low objective value */
  --data-contour-high:#7f1d1d;      /* Dark red — high objective value */

  /* ── Text ── */
  --text-primary:     #f1f5f9;      /* Near-white — headings, key values */
  --text-secondary:   #94a3b8;      /* Muted gray — labels, descriptions */
  --text-tertiary:    #64748b;      /* Dim gray — placeholders, disabled */
  --text-accent:      #a5b4fc;      /* Light indigo — links, highlights */
  --text-code:        #c4b5fd;      /* Light purple — code, math values */

  /* ── Borders & Dividers ── */
  --border-subtle:    rgba(148, 163, 184, 0.1);
  --border-default:   rgba(148, 163, 184, 0.2);
  --border-strong:    rgba(148, 163, 184, 0.3);

  /* ── Shadows ── */
  --shadow-sm:        0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md:        0 4px 12px rgba(0, 0, 0, 0.4);
  --shadow-lg:        0 8px 24px rgba(0, 0, 0, 0.5);
  --shadow-glow:      0 0 20px var(--accent-glow);

  /* ── Spacing Scale ── */
  --space-1:  4px;
  --space-2:  8px;
  --space-3:  12px;
  --space-4:  16px;
  --space-5:  20px;
  --space-6:  24px;
  --space-8:  32px;
  --space-10: 40px;
  --space-12: 48px;
  --space-16: 64px;

  /* ── Border Radius ── */
  --radius-sm:  4px;
  --radius-md:  8px;
  --radius-lg:  12px;
  --radius-xl:  16px;
  --radius-full: 9999px;

  /* ── Transitions ── */
  --transition-fast:   150ms ease;
  --transition-base:   250ms ease;
  --transition-slow:   400ms ease;
  --transition-spring: 300ms cubic-bezier(0.34, 1.56, 0.64, 1);
}
```

### Glassmorphism Effects

```css
.glass-panel {
  background: rgba(17, 24, 39, 0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border-subtle);
}
```

---

## 2. Fonts

### Primary: Inter

- **Use for**: All body text, labels, descriptions, buttons
- **Weights**: 400 (Regular), 500 (Medium), 600 (SemiBold), 700 (Bold)
- **Source**: Google Fonts — `https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap`

### Monospace: JetBrains Mono

- **Use for**: Mathematical values, code, metrics, data readouts
- **Weights**: 400 (Regular), 500 (Medium), 700 (Bold)
- **Source**: Google Fonts — `https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap`

### Math: KaTeX Default (Latin Modern)

- **Use for**: All LaTeX-rendered equations in the math inspector
- **Loaded by**: KaTeX library CSS

### Fallback Stack

```css
--font-sans:  'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
--font-mono:  'JetBrains Mono', 'Fira Code', 'SF Mono', 'Cascadia Code', monospace;
```

---

## 3. Typography

### Type Scale

| Token | Size | Line Height | Weight | Use |
|---|---|---|---|---|
| `--text-xs` | 11px | 1.4 | 400 | Tooltips, footnotes |
| `--text-sm` | 13px | 1.5 | 400 | Labels, axis ticks, captions |
| `--text-base` | 15px | 1.6 | 400 | Body text, descriptions |
| `--text-md` | 16px | 1.5 | 500 | Panel headers, control labels |
| `--text-lg` | 18px | 1.4 | 600 | Section titles |
| `--text-xl` | 22px | 1.3 | 700 | Page heading |
| `--text-2xl` | 28px | 1.2 | 700 | Hero title |
| `--text-data` | 14px | 1 | 500 | Monospace data readouts |
| `--text-metric` | 20px | 1 | 700 | Large metric values (e.g., "MSE: 0.342") |

### CSS Implementation

```css
:root {
  --text-xs:     11px;
  --text-sm:     13px;
  --text-base:   15px;
  --text-md:     16px;
  --text-lg:     18px;
  --text-xl:     22px;
  --text-2xl:    28px;
  --text-data:   14px;
  --text-metric: 20px;
}

body {
  font-family: var(--font-sans);
  font-size: var(--text-base);
  line-height: 1.6;
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1 { font-size: var(--text-2xl); font-weight: 700; letter-spacing: -0.02em; }
h2 { font-size: var(--text-xl);  font-weight: 700; letter-spacing: -0.01em; }
h3 { font-size: var(--text-lg);  font-weight: 600; }
h4 { font-size: var(--text-md);  font-weight: 500; }

.data-value {
  font-family: var(--font-mono);
  font-size: var(--text-data);
  font-weight: 500;
  color: var(--text-code);
}

.metric-value {
  font-family: var(--font-mono);
  font-size: var(--text-metric);
  font-weight: 700;
  color: var(--text-primary);
}
```

### Mathematical Notation Conventions

- All vectors in **bold**: **w**, **x**, **α**
- Scalars in italic: *b*, *C*, *η*
- Matrices in bold uppercase: **K**, **H**
- Set notation: 𝒟 (dataset), 𝒮 (support vectors)
- Subscripts for iteration: w_t, w_{t+1}
- Color-coding: clean values in `--color-clean`, poisoned values in `--color-attack`

### Micro-Interactions

| Element | Animation |
|---|---|
| Value change in math panel | 300ms color flash (green ↑ / red ↓) + 200ms scale bounce |
| Decision boundary shift | 400ms smooth morph transition |
| Poison point injection | 500ms "pulse" animation (scale 0 → 1.2 → 1) with red glow |
| Support vector highlight | 300ms golden ring appear + gentle pulse |
| Button hover | 150ms background lightening + subtle lift shadow |
| Panel expand/collapse | 300ms slide + opacity transition |
| Chart data point add | 200ms fade-in from right |
| Gradient arrow | Gentle 2s "breathing" opacity animation |
