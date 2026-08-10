# Design System — ML Security Attack Visualizer

*Tailwind CSS v4 · light + dark · tokens live in CSS, not JavaScript*

Implemented in [`src/app/globals.css`](../ml-security-viz/src/app/globals.css). If this document and that file disagree, the file wins — update the doc.

## 1. Design Philosophy

The interface reads as a **signal-analysis instrument**, not a marketing dashboard — something a researcher would trust to tell them the truth about how a decision boundary bends when a handful of poisoned points get slipped into training data. Four ideas carry that through everything below:

1. **The data is the only loud thing.** Chrome is near-neutral in both themes — greys with barely any hue — so gradient arrows, contour fills, and injected points are the only saturated marks on screen. Anything that isn't measurement stays quiet.
2. **Color carries meaning, and only meaning.** Red is attack/danger, emerald is clean/verified, amber is caution, blue is neutral information. The brand accent never borrows a semantic hue, so color alone never misleads about what's happening to the model.
3. **Both themes are first-class.** Dark suits a long session staring at a loss landscape; light suits a projector, a lab bench under fluorescents, and figures pasted into a paper. Neither is a tinted afterthought of the other — each has its own tuned values.
4. **Scientific credibility over spectacle.** Muted saturation, no neon, no gratuitous depth. Nothing that would look out of place in a paper's supplementary figures.

**Signature motif — Contour Field & Reticle.** Two recurring marks tie the chrome back to the actual subject instead of decorating it. A near-invisible topographic contour texture stands in for the loss landscape the tool exists to chart. A crosshair reticle marks anything being targeted — an injected poison point, or the cursor in "click to attack" mode — because that's literally what an attack on a model is: aiming at a boundary. Full detail in §8.

---

## 2. Theme Architecture

Tailwind v4 configures entirely in CSS. There is no `tailwind.config.js`.

```css
@import "tailwindcss";
@custom-variant dark (&:where(.dark, .dark *));

:root  { /* light tokens — the default */ }
.dark  { /* dark tokens — overrides */ }

@theme inline { /* maps both onto real utility classes */ }
@layer components { /* .glass-panel, .data-value, .metric-value, .eyebrow */ }
```

### 2.1 Why `@theme inline` matters here

`@theme inline` makes utilities resolve to the *underlying* variable rather than a snapshot of its value. So `bg-card` compiles to something that reads `--card` at runtime, and swapping the class on `<html>` re-resolves every token on the page at once. **This is the whole theming mechanism** — there are no `dark:` variants on components, and there shouldn't be. Write `bg-card`, not `bg-white dark:bg-slate-900`.

### 2.2 How the theme is chosen

| Step | Where |
|---|---|
| Resolve before first paint (no flash) | inline script in [`layout.tsx`](../ml-security-viz/src/app/layout.tsx) |
| Precedence | saved choice (`localStorage["ml-sec-theme"]`) → system `prefers-color-scheme` → **dark** |
| Runtime state | `theme` in the Zustand store |
| DOM + persistence owner | [`ThemeToggle.tsx`](../ml-security-viz/src/components/ThemeToggle.tsx) |

`ThemeToggle` *adopts* whatever the pre-paint script decided on mount rather than overwriting it, which is what keeps the server render, the first paint, and the store in agreement. The root element also carries `color-scheme`, so native scrollbars, form controls, and the browser's own canvas match the theme.

### 2.3 The escape hatch for JS-computed color

Recharts inline styles and a few SVG attributes are computed in JavaScript and can't read a CSS class. Those use [`useThemeTokens()`](../ml-security-viz/src/hooks/useThemeTokens.ts), which returns resolved token values and recomputes when the theme changes. **It is a last resort, not a convenience** — if a Tailwind class can express the color, use the class.

---

## 3. Color System

Every token below is defined for both themes. Light values are tuned for contrast on white; dark values are lightened so they stay readable on near-black without going neon.

### 3.1 Interface tokens

Naming follows the `background` / `foreground` / `card` / `primary` convention that shadcn/ui and most Tailwind v4 kits expect, so a drop-in component library works without renaming a class.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--background` | `#ffffff` | `#0a0d12` | App canvas, plot surface |
| `--foreground` | `#0d1117` | `#e8edf5` | Default text |
| `--card` | `#f8fafc` | `#11161e` | Panel / chrome surfaces |
| `--popover` | `#ffffff` | `#161c25` | Dropdowns, modals, tooltips |
| `--primary` | `#0369a1` | `#38bdf8` | Buttons, focus, brand |
| `--primary-foreground` | `#ffffff` | `#041520` | Text on primary fill |
| `--secondary` | `#f1f5f9` | `#171d27` | Secondary buttons, chips, strips |
| `--muted` | `#e8edf3` | `#1e2530` | Disabled fields, subtle fills |
| `--muted-foreground` | `#5b6775` | `#8c98a9` | Labels, captions, placeholders |
| `--accent` | `#075985` | `#7dd3fc` | Links |
| `--border` | `rgba(15,23,42,.12)` | `rgba(148,163,184,.16)` | Default divider |
| `--border-subtle` | `rgba(15,23,42,.07)` | `rgba(148,163,184,.08)` | Faint dividers |
| `--border-strong` | `rgba(15,23,42,.22)` | `rgba(148,163,184,.3)` | Emphasized borders |
| `--ring` | `var(--primary)` | `var(--primary)` | Focus ring |
| `--radius` | `0.625rem` | — | Base corner radius (§6.2) |

Note the inversion: in light, `--background` is the *brightest* surface and chrome sits slightly grey on top of it; in dark, `--background` is the *darkest* and chrome lifts above it. In both cases the plot area is the extreme end of the scale, so data always sits on the cleanest possible surface.

**On the primary hue.** Indigo-on-near-black is the default a huge share of AI and dev tools reach for — it says nothing specific about *this* tool. The primary here sits in the sky family: closer to an oscilloscope trace or a radar sweep than a SaaS button, which fits a tool whose whole job is reading a signal for tampering. Light uses sky-700 and dark uses sky-400 — the same hue, each side tuned to clear 4.5:1 on its own background.

### 3.2 Status tokens — domain semantics

The functional vocabulary of the tool. These never shift for branding reasons; an attack is always red.

| Token | Light | Dark | Meaning | Alias |
|---|---|---|---|---|
| `--clean` | `#047857` | `#34d399` | Clean / verified model state | `--success` |
| `--attack` | `#b91c1c` | `#f87171` | Attack, poisoned, danger | `--destructive` |
| `--warning` | `#b45309` | `#fbbf24` | Caution, needs review | — |
| `--info` | `#2563eb` | `#60a5fa` | Informational, neutral | — |

Each has a matching `-foreground` for text set directly on a solid fill.

### 3.3 Data-visualization tokens

Unique to the plots and math inspector — never used for UI chrome, so they can't collide with the meanings above. They are deliberately **decoupled from the status tokens**: chart marks want saturation, small text wants contrast, and those two goals pull in opposite directions. That's why `--data-poison` is not an alias of `--attack`.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--data-class-a` | `#4f46e5` | `#818cf8` | Scatter class +1 |
| `--data-class-b` | `#ea580c` | `#fb923c` | Scatter class −1 |
| `--data-poison` | `#dc2626` | `#ef4444` | Poisoned / injected points |
| `--data-support` | `#b45309` | `#fbbf24` | Support-vector ring, selection ring |
| `--data-boundary` | `#0f172a` | `#e2e8f0` | Decision boundary line |
| `--data-gradient` | `#7c3aed` | `#a78bfa` | Gradient / attack-direction arrows |
| `--data-grid` | `rgba(15,23,42,.06)` | `rgba(148,163,184,.08)` | Plot gridlines |
| `--data-contour-low` | `#bfdbfe` | `#1e3a5f` | Objective value — low |
| `--data-contour-high` | `#fecaca` | `#7f1d1d` | Objective value — high |
| `--data-image-bg` | `#0b0f14` | `#000000` | MNIST/CIFAR raster backdrop |
| `--text-code` | `#6d28d9` | `#c4b5fd` | Inline math/code values |
| `--glow-clean` / `--glow-attack` | 28% alpha | 35% alpha | Boundary drop-shadow halos |

Two conventions worth knowing:

- **Scatter points are stroked with `--background`, not white.** A halo the color of the canvas separates overlapping dots in both themes; a hardcoded white halo vanishes on a light background.
- **`--data-image-bg` stays near-black in both themes.** MNIST digits are white-on-black rasters — that's the data, not chrome, so inverting it would misrepresent the sample.

### 3.4 Elevation

Shadows are theme-specific: soft neutral drops on light, deeper black on dark. Raw `--elevation-1/2/3` are mapped to `shadow-sm` / `shadow-md` / `shadow-lg`, and `--elevation-glow` to `shadow-glow` (a focus halo on light, an actual glow on dark).

### 3.5 Glass panels

```css
.glass-panel {
  background: color-mix(in oklab, var(--color-card) var(--glass-tint), transparent);
  backdrop-filter: blur(10px);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
}
```

`--glass-tint` is 82% on light and 72% on dark — light backgrounds need more opacity to stay legible over a busy plot. The panel color comes from `--card`, so retuning the card token updates every glass surface.

---

## 4. Accessibility

Measured on the live page, WCAG 2.1 contrast ratios, lowest value across both themes:

| Pair | Light | Dark |
|---|---|---|
| `foreground` on `background` | 18.9 | 16.6 |
| `muted-foreground` on `card` | 5.5 | 6.2 |
| `primary` on `background` | 5.9 | 9.1 |
| `clean` on `secondary` | 5.0 | 8.8 |
| `attack` on `secondary` | 5.9 | 6.1 |
| `warning` on `background` | 5.0 | 11.7 |
| `text-code` on `card` | 6.8 | 9.8 |
| `primary-foreground` on `primary` | 5.9 | 8.7 |

Every pair clears **4.5:1** (AA for normal text) in both themes. When adding or retuning a token, re-check it — the status colors in light mode have the least headroom, and it is easy to reach for a `-500` Tailwind hue that fails on white.

Beyond contrast: focus is never conveyed by color alone (`:focus-visible` draws a 2px `--ring` outline with offset), status pills pair color with a text label, and `prefers-reduced-motion` collapses every animation to its end state.

---

## 5. Typography

| Role | Family | Notes |
|---|---|---|
| Body & UI | **Inter** (variable) | Tall x-height and screen hinting at small sizes — what a dense dashboard needs |
| Headings | **Inter Tight** (variable) | Same foundry, tighter tracking; more presence without a second typographic voice |
| Data, code, labels | **JetBrains Mono** (variable) | Math, metrics, code, *and* eyebrows/panel headers — mono marks anything **measured or machine-generated**, Inter marks anything **written** |
| Math | **KaTeX** (Latin Modern) | Rendered by KaTeX itself |

All three load through `next/font/google` as **variable** fonts — one file per family, every weight available, self-hosted with no network request to Google at runtime.

### 5.1 Wiring (this part is easy to get wrong)

`next/font` generates an obfuscated family name and exposes it as a CSS variable. The theme must point at that variable:

```ts
// layout.tsx
const inter = Inter({ variable: "--font-inter", subsets: ["latin"], display: "swap" });
```

```css
/* globals.css */
@theme inline {
  --font-sans: var(--font-inter), ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-display: var(--font-inter-tight), var(--font-inter), ui-sans-serif, system-ui, sans-serif;
  --font-mono: var(--font-jetbrains-mono), ui-monospace, "SF Mono", "Cascadia Mono", monospace;
}
```

Naming the literal family (`--font-sans: "Inter", …`) instead **silently falls back to the system font**, because `"Inter"` never matches the hashed family the loader actually registered. Same trap inside SVG: use `className="font-sans"` / `font-mono` on `<text>`, never `fontFamily="Inter"`.

### 5.2 OpenType features

Two settings that matter for a numeric UI:

```css
body       { font-feature-settings: "cv02" 1, "cv03" 1, "cv04" 1, "cv11" 1; }  /* open digits, single-storey a */
.data-value, .metric-value {
  font-variant-numeric: tabular-nums slashed-zero;
}
```

Tabular figures are the important one: metrics update every iteration, and proportional digits make the whole panel twitch as values change. Slashed zero keeps `0` and `O` apart in weight vectors.

### 5.3 Type scale

```css
--text-xs: 0.6875rem;   /* 11px */   --text-lg: 1.125rem;   /* 18px */
--text-sm: 0.8125rem;   /* 13px */   --text-xl: 1.375rem;   /* 22px */
--text-base: 0.9375rem; /* 15px */   --text-2xl: 1.75rem;   /* 28px */
--text-md: 1rem;        /* 16px */   --text-data: 0.875rem; /* 14px */
                                     --text-metric: 1.25rem;/* 20px */
```

> Redeclaring `--text-base`/`lg`/`xl`/`2xl` **overrides Tailwind's built-in scale project-wide.** Intentional — 15px reads better than 16px at this density — but worth flagging so `text-base` doesn't surprise anyone expecting the framework default.

Reserve `font-display` (Inter Tight) for `text-xl`/`text-2xl` headings; body copy and panel labels stay on `font-sans`. Headings carry negative tracking (−0.01 to −0.022em) that scales with size — large type needs it, small type doesn't.

### 5.4 Mathematical notation conventions

- Vectors in **bold**: **w**, **x**, **α** · Scalars in italic: *b*, *C*, *η* · Matrices bold uppercase: **K**, **H**
- Set notation: 𝒟 (dataset), 𝒮 (support vectors) · Iteration subscripts: w_t, w_{t+1}
- Color-code clean values with `text-clean`, poisoned values with `text-attack`
- Every formula matches its paper's notation, with the equation reference in a comment or tooltip

---

## 6. Spacing, Radius & Shadow

### 6.1 Spacing

The 4/8/12/16/20/24/32/40/48/64px scale is, pixel for pixel, Tailwind's default. No custom config: `p-4`, `gap-6`, `space-y-8` are already correct.

### 6.2 Radius

```css
:root { --radius: 0.625rem; }        /* 10px */
--radius-sm: calc(var(--radius) - 6px);  /*  4px */
--radius-md: calc(var(--radius) - 3px);  /*  7px */
--radius-lg: var(--radius);              /* 10px */
--radius-xl: calc(var(--radius) + 5px);  /* 15px */
```

Driven from one variable — turn the `--radius` knob and the whole scale retunes together. 10px rather than the old 12px: slightly tighter reads as instrument rather than app.

### 6.3 Shadows

Elevation is used sparingly — panels are separated by borders and surface color first, shadow only where something genuinely floats (popovers, modals, the canvas legend). On light, shadows are low-alpha neutral; on dark, they're deep black, because a shadow on a near-black surface has to work harder to register.

---

## 7. Motion

| Element | Implementation |
|---|---|
| Value change in math panel | `transition-colors duration-300` + `animate-value-flash` |
| Decision boundary shift | `transition-all duration-400 ease-in-out` on the path |
| Poison point injection | `animate-poison-pulse` (scale 0 → 1.2 → 1) |
| Support vector highlight | `animate-ring-appear` with `ease-spring` |
| Button hover | `transition-colors duration-150 hover:bg-primary/90` |
| Panel expand/collapse | `transition-all duration-300` on `grid-template-columns` |
| Gradient arrow | `animate-breathe` — 2s infinite, opacity 0.4 ↔ 1 |
| Theme switch | 180ms `background-color`/`color` fade on `body` |

Durations don't need theme tokens in v4 — `duration-150`, `duration-400`, or arbitrary `duration-[420ms]` all work directly.

### 7.1 Reduced motion

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

Every animation must still communicate its end state instantly — a poisoned point still turns red, it just stops pulsing to get there.

---

## 8. Signature Motif — Contour Field & Reticle

Both marks are pulled from the math this tool visualizes rather than invented as decoration.

**Contour field.** A near-invisible topographic texture behind large panels, standing in for the objective-function landscape. It reuses the contour heatmap hues, so it reads as part of the same visual language. Keep it at ~5% opacity on dark and ~7% on light — a light surface swallows a faint line faster than a dark one.

```html
<svg class="pointer-events-none absolute inset-0 h-full w-full text-data-contour-low/[0.06]" aria-hidden="true">
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

**Reticle.** A crosshair marks anything targeted — an injected poison point, or the cursor in "click to inject" mode. `text-attack` for a real attack marker, `text-primary` for a neutral targeting cursor.

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

Use the reticle sparingly, on the two or three moments genuinely about targeting something. Everywhere else stays quiet so those moments land.

---

## 9. Component Recipes

```css
@layer components {
  .data-value   { font-family: var(--font-mono); font-size: var(--text-data);
                  font-weight: 500; font-variant-numeric: tabular-nums slashed-zero;
                  color: var(--color-text-code); }

  .metric-value { font-family: var(--font-mono); font-size: var(--text-metric);
                  font-weight: 700; font-variant-numeric: tabular-nums slashed-zero;
                  letter-spacing: -0.01em; color: var(--color-foreground); }

  .eyebrow      { font-family: var(--font-mono); font-size: var(--text-xs);
                  font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
                  color: var(--color-muted-foreground); }
}
```

| Component | Classes |
|---|---|
| Primary button | `inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors duration-150 hover:bg-primary/90 focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50` |
| Secondary button | `inline-flex items-center justify-center rounded-md border border-border bg-secondary px-4 py-2 text-sm font-medium text-secondary-foreground transition-colors duration-150 hover:bg-secondary/80` |
| Icon button | `flex items-center justify-center w-8 h-8 rounded-md border border-border bg-secondary text-secondary-foreground transition-colors duration-150 hover:bg-secondary/80 hover:text-foreground` |
| Status pill — clean | `inline-flex items-center gap-1.5 rounded-full bg-clean/10 px-2.5 py-1 text-xs font-medium text-clean ring-1 ring-inset ring-clean/20` |
| Status pill — attack | `inline-flex items-center gap-1.5 rounded-full bg-attack/10 px-2.5 py-1 text-xs font-medium text-attack ring-1 ring-inset ring-attack/20` |
| Panel header | `eyebrow flex items-center justify-between border-b border-border-subtle px-4 py-3` |
| Modal surface | `bg-popover text-popover-foreground border border-border rounded-xl shadow-lg` |
| Modal scrim | `fixed inset-0 bg-foreground/25 backdrop-blur-sm` |

Note the scrim: `bg-foreground/25` dims correctly in **both** themes, whereas `bg-black/60` is invisible on dark and crushing on light.

---

## 10. Rules for Contributors

**Never hardcode a color.** Not `bg-white`, not `text-gray-400`, not `bg-[#059669]`, not `stroke="rgba(255,255,255,0.2)"`, not a hex literal in a Recharts prop. Every one of those is a light-mode or dark-mode bug waiting to happen. Reach for the semantic token; if none fits, add one to *both* theme blocks.

Also:

- **Don't write `dark:` variants** on components. The token layer already handles the theme; a `dark:` override is a second source of truth that will drift.
- **`--data-*` tokens are for plots only.** Don't dress up UI chrome in them, or the color vocabulary stops being trustworthy.
- **Use `useThemeTokens()` only where CSS genuinely can't reach** — Recharts style objects and JS-computed SVG attributes. Everything else takes a class.
- **Re-check contrast when retuning a status color**, especially in light mode (§4).
- **Sanity-check both themes before opening a PR.** The toggle is in the header; the fastest check is running an attack and scrubbing the timeline in each.

### Appendix: adding a token

1. Add the raw variable to **both** `:root` and `.dark` in `globals.css`.
2. Map it in `@theme inline` (`--color-<name>: var(--<name>)`) to get `bg-`/`text-`/`border-`/`fill-`/`stroke-` utilities for free.
3. If JS needs it, add it to `VAR_NAMES` **and** the dark-value `FALLBACK` in `useThemeTokens.ts` — the fallback covers the pre-hydration render.
4. Document it in §3 with both values.
