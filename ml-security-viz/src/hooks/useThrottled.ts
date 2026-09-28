import { useEffect, useRef, useState } from 'react';

/**
 * `value`, updated at most once per `ms` (always the latest value once the interval allows).
 * ms ≤ 0 passes the value straight through.
 */
export function useThrottled<T>(value: T, ms: number): T {
  const [throttled, setThrottled] = useState(value);
  const last = useRef(0);
  useEffect(() => {
    if (ms <= 0) return;
    const wait = Math.max(0, last.current + ms - Date.now());
    const id = setTimeout(() => { last.current = Date.now(); setThrottled(value); }, wait);
    return () => clearTimeout(id);
  }, [value, ms]);
  return ms <= 0 ? value : throttled;
}
