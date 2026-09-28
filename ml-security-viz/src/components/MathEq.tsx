'use client';
import React, { useMemo, useState } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface MathEqProps {
  math: string;
  inline?: boolean;
  tooltip?: React.ReactNode;
}

export default function MathEq({ math, inline = true, tooltip }: MathEqProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        throwOnError: false,
        displayMode: !inline,
      });
    } catch (e) {
      return math;
    }
  }, [math, inline]);

  const wrapper = (
    <span
      dangerouslySetInnerHTML={{ __html: html }}
      className={`inline-block ${tooltip ? 'cursor-help border-b border-dashed border-muted-foreground' : ''}`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    />
  );

  if (!tooltip) return wrapper;

  return (
    <div className="relative inline-block">
      {wrapper}
      {showTooltip && (
        <div className="absolute z-50 w-48 p-2 mt-1 text-xs text-foreground bg-background border border-border rounded shadow-lg left-1/2 -translate-x-1/2 before:content-[''] before:absolute before:-top-1 before:left-1/2 before:-translate-x-1/2 before:border-4 before:border-transparent before:border-b-[var(--border-strong)]">
          {tooltip}
        </div>
      )}
    </div>
  );
}
