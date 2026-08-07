const fs = require('fs');

const path = 'c:/Users/soham/Desktop/ML for Security/ml-security-viz/src/components/MathPanel.tsx';
let content = fs.readFileSync(path, 'utf8');

const replacements = [
  [/bg-\[var\(--bg-secondary\)\]/g, 'bg-card'],
  [/bg-\[var\(--bg-tertiary\)\]/g, 'bg-secondary'],
  [/bg-\[var\(--bg-primary\)\]/g, 'bg-background'],
  [/border-\[var\(--border-subtle\)\]/g, 'border-border-subtle'],
  [/border-\[var\(--border-default\)\]/g, 'border-border'],
  [/text-\[var\(--text-primary\)\]/g, 'text-foreground'],
  [/text-\[var\(--text-secondary\)\]/g, 'text-muted-foreground'],
  [/text-\[var\(--text-tertiary\)\]/g, 'text-muted-foreground/70'],
  [/text-\[var\(--text-code\)\]/g, 'text-text-code'],
  [/text-\[var\(--text-accent\)\]/g, 'text-accent'],
  [/bg-\[var\(--color-clean\)\]/g, 'bg-clean'],
  [/bg-\[var\(--color-attack\)\]/g, 'bg-attack'],
  [/text-\[var\(--color-clean\)\]/g, 'text-clean'],
  [/text-\[var\(--color-attack\)\]/g, 'text-attack'],
  [/shadow-\[0_0_4px_var\(--color-attack-dim\)\]/g, 'shadow-sm shadow-attack/50'],
  [/<aside className="bg-card border-l border-border-subtle overflow-y-auto p-3 flex flex-col gap-3 flex-1 min-h-0">/g, '<aside className="glass-panel overflow-y-auto p-4 flex flex-col gap-4 flex-1 min-h-0">'],
  [/className="text-sm font-semibold text-foreground flex items-center gap-2/g, 'className="eyebrow flex items-center gap-2'],
  [/text-sm font-semibold text-foreground px-4 py-3/g, 'eyebrow px-4 py-3'],
  [/font-mono text-xs font-medium text-text-code/g, 'data-value']
];

for (const [regex, replacement] of replacements) {
  content = content.replace(regex, replacement);
}

fs.writeFileSync(path, content, 'utf8');
console.log('MathPanel updated.');
