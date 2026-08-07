# Design System — ML Security Attack Visualizer

*Tailwind CSS v4 · dark-only interface · tokens live in CSS, not JavaScript*

## 1. Design Philosophy

The interface reads as a **signal-analysis instrument**, not a marketing dashboard — something a researcher would trust to tell them the truth about how an SVM's decision boundary bends when a handful of poisoned points get slipped into training data. Three ideas carry that through everything below:

1. **High contrast on dark.** The workspace stays dark so gradient arrows, contour fills, and injected points read as *the signal*, not the chrome around it.
2. **Color carries meaning, and only meaning.** Red is reserved for attack/danger states, emerald for clean/verified states, amber for caution, blue for neutral information. The brand accent never borrows a semantic hue, so color alone never misleads about what's happening to the model.
3. **Scientific credibility over spectacle.** Muted saturation, no neon, nothing that would look out of place in a paper's supplementary figures.

The system now speaks Tailwind natively: every token below is a CSS custom property inside a v4 `@theme` block, so the full palette, scale, and motion system ships as real utility classes (`bg-attack`, `text-clean`, `rounded-lg`, `duration-300`) with zero JavaScript config file. A `tailwind.config.js` fallback for teams still on v3.4 is at the bottom of this document.

**Signature motif — Contour Field & Reticle.** Two recurring marks tie the chrome back to the actual subject instead of decorating it. A near-invisible topographic contour texture — built from the same hues as the objective-value heatmap — sits behind large panels, standing in for the loss landscape the tool exists to chart. A crosshair reticle marks anything being targeted: an injected poison point, or the cursor itself in "click to attack" mode — because that's literally what an attack on a model is, aiming at a boundary. Full detail in §7.

---

## 2. Tailwind Architecture

Tailwind v4 configures entirely in CSS — no `tailwind.config.js` required for this system.

```css
/* app.css */
@import "tailwindcss";
@custom-variant dark (&:is(.dark *));

:root {
  /* design tokens — §3.4, §5, §6 */
}

@theme inline {
  /* maps the tokens above into real Tailwind utilities */
}

@layer components {
  /* .glass-panel, .data-value, .metric-value, .eyebrow — §8 */
}

@media (prefers-reduced-motion: reduce) {
  /* §6.1 */
}
```

Every color, radius, shadow, and font declared below becomes a real utility class the moment it exists — `--color-attack` in `@theme` gives you `bg-attack`, `text-attack`, `border-attack`, `ring-attack`, and `fill-attack` for free. This system is dark-only today (no `.light` variant exists), but the `:root` / `@theme inline` split leaves room to add one later without restructuring anything.

---

## 3. Color System

### 3.1 Interface tokens

These follow the `background` / `foreground` / `card` / `primary` naming that shadcn/ui and most Tailwind v4 component kits already expect, so a drop-in component library works without renaming a single class.

| Token | Value | Use |
|---|---|---|
| `--background` | `#0a0e17` | App canvas |
| `--foreground` | `#f1f5f9` | Default text |
| `--card` | `#111827` | Panel / card surfaces |
| `--card-foreground` | `#f1f5f9` | Text on cards |
| `--popover` | `#1a2332` | Dropdowns, modals, tooltips |
| `--popover-foreground` | `#f1f5f9` | Text on popovers |
| `--primary` | `#0ea5e9` | Buttons, links, focus, brand |
| `--primary-foreground` | `#f8fafc` | Text on primary fill |
| `--secondary` | `#1f2d3d` | Secondary buttons, chips |
| `--secondary-foreground` | `#f1f5f9` | Text on secondary |
| `--muted` | `#263445` | Disabled fields, subtle fills |
| `--muted-foreground` | `#94a3b8` | Labels, captions, placeholders |
| `--accent` | `#38bdf8` | Hover/highlight tint, links |
| `--accent-foreground` | `#f1f5f9` | Text on accent tint |
| `--border` | `rgba(148,163,184,0.2)` | Default border/divider |
| `--border-subtle` | `rgba(148,163,184,0.1)` | Faint dividers |
| `--border-strong` | `rgba(148,163,184,0.3)` | Emphasized borders |
| `--input` | `var(--border)` | Form field borders |
| `--ring` | `var(--primary)` | Focus ring |
| `--radius` | `0.75rem` | Base corner radius — see §5.2 |

**Why the accent moved off indigo.** `#6366f1` indigo-on-near-black is the default a huge share of AI and dev tools reach for right now — it doesn't say anything specific about *this* tool. The new primary, `#0ea5e9`, sits in the sky/cyan family: a cooler, more electric blue closer to an oscilloscope trace or a radar sweep than a SaaS button, which fits a tool whose whole job is reading a signal for tampering. It's still restrained — moderate lightness, no neon — and it sits a clean 20°+ of hue away from every semantic color below, so it's never mistaken for a status.

### 3.2 Status tokens — domain semantics

The functional vocabulary of the tool. These never shift for branding reasons; an attack is always red.

| Token | Value | Meaning | shadcn-compatible alias |
|---|---|---|---|
| `--clean` | `#10b981` | Clean / verified model state | `--success` |
| `--attack` | `#ef4444` | Attack, poisoned, danger | `--destructive` |
| `--warning` | `#f59e0b` | Caution, needs review | `--warning` |
| `--info` | `#3b82f6` | Informational, neutral | — |

Each has a matching `-foreground` token for text set directly on a solid fill.

### 3.3 Data-visualization tokens

Unique to the plots and math inspector — never used for UI chrome, so they can't collide with the meanings above.

| Token | Value | Use |
|---|---|---|
| `--data-class-a` | `#6366f1` | Scatter plot class +1 |
| `--data-class-b` | `#f97316` | Scatter plot class −1 |
| `--data-poison` | `var(--attack)` | Poisoned / injected points |
| `--data-support` | `#fbbf24` | Support-vector ring |
| `--data-boundary` | `#e2e8f0` | Decision boundary line |
| `--data-gradient` | `#8b5cf6` | Gradient / attack-direction arrows |
| `--data-contour-low` | `#1e3a5f` | Objective value — low |
| `--data-contour-high` | `#7f1d1d` | Objective value — high |
| `--text-code` | `#c4b5fd` | Inline math/code values |

### 3.4 The full token block

```css
:root {
  --background: #0a0e17;
  --foreground: #f1f5f9;
  --card: #111827;
  --card-foreground: #f1f5f9;
  --popover: #1a2332;
  --popover-foreground: #f1f5f9;
  --primary: #0ea5e9;
  --primary-foreground: #f8fafc;
  --secondary: #1f2d3d;
  --secondary-foreground: #f1f5f9;
  --muted: #263445;
  --muted-foreground: #94a3b8;
  --accent: #38bdf8;
  --accent-foreground: #f1f5f9;
  --border: rgba(148, 163, 184, 0.2);
  --border-subtle: rgba(148, 163, 184, 0.1);
  --border-strong: rgba(148, 163, 184, 0.3);
  --input: var(--border);
  --ring: var(--primary);
  --radius: 0.75rem;

  --clean: #10b981;
  --clean-foreground: #052e16;
  --attack: #ef4444;
  --attack-foreground: #fef2f2;
  --warning: #f59e0b;
  --warning-foreground: #451a03;
  --info: #3b82f6;
  --info-foreground: #eff6ff;
  --success: var(--clean);
  --success-foreground: var(--clean-foreground);
  --destructive: var(--attack);
  --destructive-foreground: var(--attack-foreground);

  --data-class-a: #6366f1;
  --data-class-b: #f97316;
  --data-poison: var(--attack);
  --data-support: #fbbf24;
  --data-boundary: #e2e8f0;
  --data-gradient: #8b5cf6;
  --data-contour-low: #1e3a5f;
  --data-contour-high: #7f1d1d;
  --text-code: #c4b5fd;
}

@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --color-card: var(--card);
  --color-card-foreground: var(--card-foreground);
  --color-popover: var(--popover);
  --color-popover-foreground: var(--popover-foreground);
  --color-primary: var(--primary);
  --color-primary-foreground: var(--primary-foreground);
  --color-secondary: var(--secondary);
  --color-secondary-foreground: var(--secondary-foreground);
  --color-muted: var(--muted);
  --color-muted-foreground: var(--muted-foreground);
  --color-accent: var(--accent);
  --color-accent-foreground: var(--accent-foreground);
  --color-border: var(--border);
  --color-border-subtle: var(--border-subtle);
  --color-border-strong: var(--border-strong);
  --color-input: var(--input);
  --color-ring: var(--ring);

  --color-clean: var(--clean);
  --color-clean-foreground: var(--clean-foreground);
  --color-attack: var(--attack);
  --color-attack-foreground: var(--attack-foreground);
  --color-warning: var(--warning);
  --color-warning-foreground: var(--warning-foreground);
  --color-info: var(--info);
  --color-info-foreground: var(--info-foreground);
  --color-success: var(--success);
  --color-success-foreground: var(--success-foreground);
  --color-destructive: var(--destructive);
  --color-destructive-foreground: var(--destructive-foreground);

  --color-data-class-a: var(--data-class-a);
  --color-data-class-b: var(--data-class-b);
  --color-data-poison: var(--data-poison);
  --color-data-support: var(--data-support);
  --color-data-boundary: var(--data-boundary);
  --color-data-gradient: var(--data-gradient);
  --color-data-contour-low: var(--data-contour-low);
  --color-data-contour-high: var(--data-contour-high);
  --color-text-code: var(--text-code);
}
```

That's the entire palette wired up — `bg-card`, `text-muted-foreground`, `border-attack/40`, `fill-data-class-a` are all real, generated classes. No config file involved.

### 3.5 Glassmorphism, updated

```css
@layer components {
  .glass-panel {
    background: color-mix(in oklab, var(--color-card) 70%, transparent);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--color-border-subtle);
  }
}
```

Same effect as before, but `color-mix()` reads the panel color from `--color-card` instead of a hardcoded `rgba(17, 24, 39, 0.7)` — change the card token once and every glass panel updates with it.

---

## 4. Typography

| Role | Family | Notes |
|---|---|---|
| Body & UI | **Inter** (variable) | Unchanged — tall x-height and screen hinting at small sizes are exactly what a dense dashboard needs |
| Headings | **Inter Tight** (variable) | Same foundry, tighter default tracking — gives large text more presence without loading a second typographic voice |
| Data, code, labels | **JetBrains Mono** | Unchanged for math/metrics/code, *and* now also used for eyebrows and panel headers (uppercase, tracked out) — mono marks anything measured or machine-generated, Inter marks anything written |
| Math | **KaTeX** (Latin Modern) | Unchanged |

Weights in use: Inter 400/500/600/700 and JetBrains Mono 400/500/700 — all map directly to Tailwind's default `font-normal`/`medium`/`semibold`/`bold` utilities, no config needed.

```css
@theme {
  --font-sans: "Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  --font-display: "Inter Tight", "Inter", sans-serif;
  --font-mono: "JetBrains Mono", "Fira Code", "SF Mono", monospace;
}
```

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Inter+Tight:wght@600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
```

### 4.1 Type scale

```css
@theme {
  --text-xs: 0.6875rem;      /* 11px */
  --text-xs--line-height: 1.4;
  --text-sm: 0.8125rem;      /* 13px */
  --text-sm--line-height: 1.5;
  --text-base: 0.9375rem;    /* 15px — see override note below */
  --text-base--line-height: 1.6;
  --text-md: 1rem;           /* 16px */
  --text-md--line-height: 1.5;
  --text-lg: 1.125rem;       /* 18px */
  --text-lg--line-height: 1.4;
  --text-xl: 1.375rem;       /* 22px */
  --text-xl--line-height: 1.3;
  --text-2xl: 1.75rem;       /* 28px */
  --text-2xl--line-height: 1.2;
  --text-data: 0.875rem;     /* 14px */
  --text-data--line-height: 1;
  --text-metric: 1.25rem;    /* 20px */
  --text-metric--line-height: 1;
}
```

> Redeclaring `--text-base`, `--text-lg`, `--text-xl`, and `--text-2xl` inside `@theme` **overrides Tailwind's built-in scale project-wide.** That's intentional here — 15px reads better than 16px at this data density — but it's worth flagging so `text-base` doesn't surprise anyone expecting the framework default.

Reserve `font-display` (Inter Tight) for `text-xl`/`text-2xl` headings only; body copy and panel labels stay on `font-sans`.

### 4.2 Mathematical notation conventions

Unchanged from the original system — still the right conventions:

- Vectors in **bold**: **w**, **x**, **α**
- Scalars in italic: *b*, *C*, *η*
- Matrices bold uppercase: **K**, **H**
- Set notation: 𝒟 (dataset), 𝒮 (support vectors)
- Iteration subscripts: w_t, w_{t+1}
- Color-code clean values with `text-clean`, poisoned values with `text-attack`

---

## 5. Spacing, Radius & Shadow

### 5.1 Spacing

The original 4/8/12/16/20/24/32/40/48/64px scale is, pixel for pixel, Tailwind's default spacing scale. No custom spacing config is needed at all:

| px | Tailwind | px | Tailwind |
|---|---|---|---|
| 4 | `1` | 24 | `6` |
| 8 | `2` | 32 | `8` |
| 12 | `3` | 40 | `10` |
| 16 | `4` | 48 | `12` |
| 20 | `5` | 64 | `16` |

`p-4`, `gap-6`, `space-y-8` are already correct, out of the box.

### 5.2 Radius

```css
:root {
  --radius: 0.75rem; /* 12px — equal to --radius-lg */
}
@theme inline {
  --radius-sm: calc(var(--radius) - 8px);  /* 4px */
  --radius-md: calc(var(--radius) - 4px);  /* 8px */
  --radius-lg: var(--radius);              /* 12px */
  --radius-xl: calc(var(--radius) + 4px);  /* 16px */
  --radius-full: 9999px;
}
```

Same 4/8/12/16 values as the original system, but now driven from one variable — turn the `--radius` knob and the whole scale retunes together.

### 5.3 Shadows

```css
@theme {
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
  --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.5);
  --shadow-glow: 0 0 20px rgba(14, 165, 233, 0.18);
}
```

`shadow-sm`, `shadow-md`, `shadow-lg`, `shadow-glow` are generated automatically; `shadow-glow` now keys off the new primary hue.

---

## 6. Motion

```css
@theme {
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}
```

Durations don't need theme tokens in v4 — `duration-150`, `duration-250`, `duration-400`, or any arbitrary `duration-[420ms]` all work directly as utilities without configuration.

| Element | Implementation |
|---|---|
| Value change in math panel | `transition-colors duration-300` + `animate-value-flash` (scale bounce) |
| Decision boundary shift | `transition-all duration-400 ease-in-out` on the path, or a morph library for the SVG `d` attribute |
| Poison point injection | `animate-poison-pulse` (scale 0 → 1.2 → 1, red glow) |
| Support vector highlight | `animate-ring-appear` with `ease-spring` |
| Button hover | `transition-colors duration-150 hover:bg-primary/90 hover:shadow-md` |
| Panel expand/collapse | `transition-all duration-300` on `grid-template-rows` (avoids animating `height: auto`) |
| Chart data point add | `animate-in fade-in slide-in-from-right-2 duration-200` via `tw-animate-css`, the v4-native successor to the old `tailwindcss-animate` plugin |
| Gradient arrow | `animate-breathe` — 2s infinite, opacity 0.4 ↔ 1 |

```css
@theme {
  --animate-poison-pulse: poison-pulse 500ms ease-out;
  --animate-ring-appear: ring-appear 300ms var(--ease-spring);
  --animate-value-flash: value-flash 300ms ease-out;
  --animate-breathe: breathe 2s ease-in-out infinite;
}

@keyframes poison-pulse {
  0% { transform: scale(0); opacity: 0; }
  60% { transform: scale(1.2); opacity: 1; }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes ring-appear {
  0% { transform: scale(0.5); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes value-flash {
  0% { transform: scale(1); }
  40% { transform: scale(1.08); }
  100% { transform: scale(1); }
}
@keyframes breathe {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}
```

### 6.1 Reduced motion

New in this revision — the original system had no `prefers-reduced-motion` handling at all:

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

Every animation above should still communicate its end state instantly — a poisoned point still turns red, it just stops pulsing to get there.

---

## 7. Signature Motif — Contour Field & Reticle

Two marks, both pulled directly from the math this tool visualizes rather than invented as decoration.

**Contour field.** A near-invisible topographic texture — concentric rings — sits behind large panels at roughly 5% opacity, standing in for the objective-function landscape the tool exists to chart. It reuses the contour heatmap hues, so it reads as part of the same visual language rather than a separate decoration.

```html
<svg class="pointer-events-none absolute inset-0 h-full w-full text-data-contour-low/[0.05]" aria-hidden="true">
  <defs>
    <pattern id="contour-field" width="140" height="140" patternUnits="userSpaceOnUse">
      <circle cx="70" cy="70" r="18" fill="none" stroke="currentColor" stroke-width="1" />
      <circle cx="70" cy="70" r="38" fill="none" stroke="currentColor" stroke-width="1" />
      <circle cx="70" cy="70" r="58" fill="none" stroke="currentColor" stroke-width="1" />
      <circle cx="70" cy="70" r="78" fill="none" stroke="currentColor" stroke-width="1" />
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#contour-field)" />
</svg>
```

**Reticle.** A crosshair marks anything targeted — an injected poison point on a scatter plot, or the cursor itself while the person is in "click to inject" mode. Render it in `text-attack` for an actual attack marker, `text-primary` for a neutral targeting cursor.

```html
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="text-attack">
  <circle cx="12" cy="12" r="6" stroke="currentColor" stroke-width="1.5" />
  <circle cx="12" cy="12" r="1.5" fill="currentColor" />
  <line x1="12" y1="1" x2="12" y2="4" stroke="currentColor" stroke-width="1.5" />
  <line x1="12" y1="20" x2="12" y2="23" stroke="currentColor" stroke-width="1.5" />
  <line x1="1" y1="12" x2="4" y2="12" stroke="currentColor" stroke-width="1.5" />
  <line x1="20" y1="12" x2="23" y2="12" stroke="currentColor" stroke-width="1.5" />
</svg>
```

Use the reticle sparingly, on the two or three moments that are genuinely about targeting something. Everywhere else stays quiet so those moments land.

---

## 8. Component Recipes

```css
@layer components {
  .glass-panel {
    background: color-mix(in oklab, var(--color-card) 70%, transparent);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--color-border-subtle);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
  }

  .data-value {
    font-family: var(--font-mono);
    font-size: var(--text-data);
    line-height: 1;
    font-weight: 500;
    color: var(--color-text-code);
  }

  .metric-value {
    font-family: var(--font-mono);
    font-size: var(--text-metric);
    line-height: 1;
    font-weight: 700;
    color: var(--color-foreground);
  }

  .eyebrow {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--color-muted-foreground);
  }
}
```

Direct utility recipes, for teams that prefer composing classes in JSX over authoring new ones:

| Component | Classes |
|---|---|
| Primary button | `inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50` |
| Secondary button | `inline-flex items-center justify-center rounded-md bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground transition-colors duration-150 hover:bg-secondary/80` |
| Status pill — clean | `inline-flex items-center gap-1.5 rounded-full bg-clean/10 px-2.5 py-1 text-xs font-medium text-clean ring-1 ring-inset ring-clean/20` |
| Status pill — attack | `inline-flex items-center gap-1.5 rounded-full bg-attack/10 px-2.5 py-1 text-xs font-medium text-attack ring-1 ring-inset ring-attack/20` |
| Panel header | `eyebrow flex items-center justify-between border-b border-border-subtle px-4 py-3` |

---

## 9. Appendix: Tailwind v3 fallback

Tailwind 3.4 stays supported until February 2027 — for teams not yet on v4, the same tokens as a `tailwind.config.js` extension:

```js
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  theme: {
    extend: {
      colors: {
        background: "#0a0e17",
        foreground: "#f1f5f9",
        card: { DEFAULT: "#111827", foreground: "#f1f5f9" },
        popover: { DEFAULT: "#1a2332", foreground: "#f1f5f9" },
        primary: { DEFAULT: "#0ea5e9", foreground: "#f8fafc" },
        secondary: { DEFAULT: "#1f2d3d", foreground: "#f1f5f9" },
        muted: { DEFAULT: "#263445", foreground: "#94a3b8" },
        accent: { DEFAULT: "#38bdf8", foreground: "#f1f5f9" },
        clean: { DEFAULT: "#10b981", foreground: "#052e16" },
        attack: { DEFAULT: "#ef4444", foreground: "#fef2f2" },
        warning: { DEFAULT: "#f59e0b", foreground: "#451a03" },
        info: { DEFAULT: "#3b82f6", foreground: "#eff6ff" },
        "data-class-a": "#6366f1",
        "data-class-b": "#f97316",
        "data-support": "#fbbf24",
        "data-boundary": "#e2e8f0",
        "data-gradient": "#8b5cf6",
        "data-contour-low": "#1e3a5f",
        "data-contour-high": "#7f1d1d",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Inter Tight", "Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        sm: "4px",
        md: "8px",
        lg: "12px",
        xl: "16px",
      },
      boxShadow: {
        glow: "0 0 20px rgba(14, 165, 233, 0.18)",
      },
    },
  },
};
```
