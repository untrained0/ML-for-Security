const fs = require('fs');
const path = require('path');

const dir = 'c:/Users/soham/Desktop/ML for Security/ml-security-viz/src/components';
const files = fs.readdirSync(dir);

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
  [/bg-\[var\(--accent-primary\)\]/g, 'bg-primary'],
  [/text-\[var\(--accent-primary\)\]/g, 'text-primary'],
  [/border-\[var\(--accent-primary\)\]/g, 'border-primary'],
  [/bg-\[var\(--accent-hover\)\]/g, 'bg-primary/90'],
  [/bg-\[var\(--accent-muted\)\]/g, 'bg-primary/10'],
  [/bg-\[var\(--bg-hover\)\]/g, 'bg-secondary/50'],
  [/bg-\[var\(--color-clean-dim\)\]/g, 'bg-clean/10'],
  [/bg-\[var\(--color-attack-dim\)\]/g, 'bg-attack/10'],
  [/bg-\[var\(--data-class-a\)\]/g, 'bg-data-class-a'],
  [/bg-\[var\(--data-class-b\)\]/g, 'bg-data-class-b'],
  [/text-\[var\(--data-support\)\]/g, 'text-data-support'],
  [/bg-\[rgba\(239,68,68,0\.1\)\]/g, 'bg-attack/10']
];

for (const file of files) {
  if (file.endsWith('.tsx') || file.endsWith('.ts')) {
    const fullPath = path.join(dir, file);
    let content = fs.readFileSync(fullPath, 'utf8');
    let original = content;
    for (const [regex, replacement] of replacements) {
      content = content.replace(regex, replacement);
    }
    if (content !== original) {
      fs.writeFileSync(fullPath, content, 'utf8');
      console.log(`Updated ${file}`);
    }
  }
}
