const fs = require('fs');

const path = 'c:/Users/soham/Desktop/ML for Security/ml-security-viz/src/components/PointInspector.tsx';
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
  [/bg-\[var\(--data-class-a\)\]/g, 'bg-data-class-a'],
  [/bg-\[var\(--data-class-b\)\]/g, 'bg-data-class-b'],
  [/text-\[var\(--color-clean\)\]/g, 'text-clean'],
  [/text-\[var\(--color-attack\)\]/g, 'text-attack'],
  [/text-\[var\(--data-support\)\]/g, 'text-data-support'],
  [/bg-\[rgba\(239,68,68,0\.1\)\]/g, 'bg-attack/10'],
  [/className="absolute bottom-6 left-6 z-30 bg-card border border-border shadow-2xl rounded-lg w-\[320px\] overflow-hidden animate-\[fade-in_0\.2s_ease-out\]"/g, 'className="glass-panel absolute bottom-6 left-6 z-30 w-[320px] overflow-hidden animate-[fade-in_0.2s_ease-out]"']
];

for (const [regex, replacement] of replacements) {
  content = content.replace(regex, replacement);
}

fs.writeFileSync(path, content, 'utf8');
console.log('PointInspector updated.');
