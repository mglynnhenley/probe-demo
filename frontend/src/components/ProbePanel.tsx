import type { AssistantMessage } from "../types";

interface Props {
  message: AssistantMessage | null;
  probeNames: string[];
  activeProbe: string | null;
  onSelectProbe: (name: string) => void;
  thresholds: Record<string, number>;
  onThresholdChange: (name: string, value: number) => void;
}

interface Span {
  text: string;
  maxScore: number;
  startIdx: number;
  // True when this span is one or more contiguous tokens above threshold.
  // False when it's a single peak token shown with surrounding context as a
  // calibration hint when nothing is firing.
  fired: boolean;
}

interface ProbeStats {
  values: number[];
  spans: Span[];
}

const CONTEXT_WINDOW = 2;
const TOP_N = 5;

function statsFor(
  message: AssistantMessage | null,
  probeName: string,
  threshold: number,
): ProbeStats {
  if (!message || message.tokens.length === 0) {
    return { values: [], spans: [] };
  }
  const tokens = message.tokens;
  const values = tokens.map((t) => t.scores[probeName] ?? 0);

  // First pass: group contiguous above-threshold tokens into spans.
  const fired: Span[] = [];
  let cur: Span | null = null;
  for (let i = 0; i < tokens.length; i++) {
    if (values[i] >= threshold) {
      if (cur) {
        cur.text += tokens[i].text;
        cur.maxScore = Math.max(cur.maxScore, values[i]);
      } else {
        cur = { text: tokens[i].text, maxScore: values[i], startIdx: i, fired: true };
      }
    } else if (cur) {
      fired.push(cur);
      cur = null;
    }
  }
  if (cur) fired.push(cur);

  if (fired.length > 0) {
    fired.sort((a, b) => b.maxScore - a.maxScore);
    return { values, spans: fired.slice(0, TOP_N) };
  }

  // No fires: surface peak tokens with ±CONTEXT_WINDOW context as calibration hints.
  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v);
  const topPeaks = indexed.slice(0, TOP_N);
  const calibration: Span[] = topPeaks.map(({ v, i }) => {
    const start = Math.max(0, i - CONTEXT_WINDOW);
    const end = Math.min(tokens.length, i + CONTEXT_WINDOW + 1);
    const text = tokens.slice(start, end).map((t) => t.text).join("");
    return { text, maxScore: v, startIdx: i, fired: false };
  });
  return { values, spans: calibration };
}

// Horizontal strip with one thin bar per token, lit when above threshold.
function FireStrip({ values, threshold }: { values: number[]; threshold: number }) {
  if (values.length === 0) {
    return <div className="firestrip empty">no tokens yet</div>;
  }
  const N = values.length;
  return (
    <svg
      className="firestrip"
      viewBox={`0 0 ${N} 10`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`Fire pattern across ${N} tokens`}
    >
      <rect x={0} y={0} width={N} height={10} fill="rgba(255,255,255,0.04)" />
      {values.map((v, i) =>
        v >= threshold ? (
          <rect
            key={i}
            x={i}
            y={0}
            width={1}
            height={10}
            fill="#dc2626"
            opacity={Math.max(0.6, v)}
          />
        ) : null,
      )}
    </svg>
  );
}

// Display text safely (visible whitespace markers help when token is just " " or "\n").
function tokenLabel(text: string): string {
  if (text === "") return "∅";
  if (text === "\n") return "↵";
  if (text.trim() === "") return text.replace(/ /g, "·").replace(/\t/g, "→");
  return text;
}

export function ProbePanel({
  message,
  probeNames,
  activeProbe,
  onSelectProbe,
  thresholds,
  onThresholdChange,
}: Props) {
  if (probeNames.length === 0) {
    return (
      <aside className="probe-panel empty">
        <h3>Probe scores</h3>
        <p className="muted">
          The backend hasn't loaded any probes. Set <code>PROBE_PATH</code> when
          starting the server to see scores here.
        </p>
      </aside>
    );
  }

  return (
    <aside className="probe-panel">
      <h3>Probe scores</h3>
      <p className="muted">
        Each probe reads a layer's hidden state and predicts a per-token score
        in [0, 1]. Adjust each probe's threshold to control what counts as a
        "fire". Click a probe to color the focused message by its score.
      </p>
      <ul className="probe-list">
        {probeNames.map((name) => {
          const threshold = thresholds[name] ?? 0.5;
          const { values, spans } = statsFor(message, name, threshold);
          const total = values.length;
          const fired = values.filter((v) => v >= threshold).length;
          const pct = total > 0 ? ((fired / total) * 100).toFixed(0) : "—";
          const isActive = activeProbe === name;
          const showingFires = spans.length > 0 && spans[0].fired;

          return (
            <li
              key={name}
              className={`probe-row ${isActive ? "active" : ""}`}
              onClick={() => onSelectProbe(name)}
            >
              <div className="probe-header">
                <span className="probe-name">{name}</span>
              </div>

              <div className="probe-threshold">
                <label>
                  <span>threshold</span>
                  <span className="value">{threshold.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={threshold}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) =>
                    onThresholdChange(name, parseFloat(e.target.value))
                  }
                />
              </div>

              <FireStrip values={values} threshold={threshold} />

              <div className="probe-fire">
                {total > 0 ? (
                  <>
                    fired on <strong>{fired}</strong> of {total} tokens (
                    {pct}%)
                  </>
                ) : (
                  <span className="muted">no tokens in focused message</span>
                )}
              </div>

              {spans.length > 0 && (
                <>
                  <div className="spans-header">
                    {showingFires
                      ? "Top fires"
                      : "Top peaks (below threshold — calibration)"}
                  </div>
                  <ol className="top-tokens">
                    {spans.map((s) => (
                      <li key={s.startIdx} className={s.fired ? "fired" : ""}>
                        <code>{tokenLabel(s.text)}</code>
                        <span className="score">{s.maxScore.toFixed(2)}</span>
                      </li>
                    ))}
                  </ol>
                </>
              )}
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
