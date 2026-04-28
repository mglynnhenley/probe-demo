import type { CSSProperties } from "react";

// Strong red when score is above threshold; nothing otherwise.
// Reused by both the per-token bubble renderer and the markdown walker.
export function fireStyle(
  score: number,
  threshold: number,
): CSSProperties | undefined {
  if (score < threshold) return undefined;
  const headroom = Math.max(0.0001, 1 - threshold);
  const intensity = Math.min(1, (score - threshold) / headroom);
  // Min 0.45 so a token just barely above threshold is still clearly lit.
  const alpha = 0.45 + intensity * 0.4;
  return {
    backgroundColor: `rgba(220, 38, 38, ${alpha.toFixed(3)})`,
    color: "#fff",
  };
}
