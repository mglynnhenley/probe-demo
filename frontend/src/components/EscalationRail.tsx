import type { AssistantMessage } from "../types";

interface Props {
  focusedMessage: AssistantMessage | null;
  activeProbe: string | null;
  highlightThreshold: number;
}

export const ESCALATION_THRESHOLD = 0.85;

type Verdict = "ALLOWED" | "MONITORED" | "ESCALATED" | "PENDING";

interface Status {
  verdict: Verdict;
  peak: number;
  fires: number;
  total: number;
}

function classify(
  message: AssistantMessage | null,
  activeProbe: string | null,
  highlightThreshold: number,
): Status {
  if (!message || !activeProbe || message.tokens.length === 0) {
    return { verdict: "PENDING", peak: 0, fires: 0, total: 0 };
  }
  let peak = 0;
  let fires = 0;
  const total = message.tokens.length;
  for (const tok of message.tokens) {
    const s = tok.scores[activeProbe];
    if (s === undefined) continue;
    if (s > peak) peak = s;
    if (s >= highlightThreshold) fires++;
  }
  let verdict: Verdict;
  if (peak >= ESCALATION_THRESHOLD) verdict = "ESCALATED";
  else if (fires > 0) verdict = "MONITORED";
  else verdict = "ALLOWED";
  return { verdict, peak, fires, total };
}

const VERDICT_COPY: Record<Verdict, string> = {
  PENDING: "Send a message to see the routing decision.",
  ALLOWED: "No probe fires. The response would be delivered unchanged.",
  MONITORED:
    "Some tokens fired, but no token crossed the escalation line. The response would be delivered and logged for offline review.",
  ESCALATED:
    "At least one token scored above the escalation line. The response would be held back and routed to a human reviewer before delivery.",
};

export function EscalationRail({
  focusedMessage,
  activeProbe,
  highlightThreshold,
}: Props) {
  const status = classify(focusedMessage, activeProbe, highlightThreshold);
  const verdictClass = `verdict verdict-${status.verdict.toLowerCase()}`;

  return (
    <aside className="rail">
      <section className="rail-section">
        <h3>What you're seeing</h3>
        <p>
          As the model generates each token, a lightweight classifier (a{" "}
          <em>probe</em>) reads its internal activations and scores the token
          on whether it's part of a flagged behavior.
        </p>
        <p>
          Tokens above the highlight threshold are tinted red in the response.
          Hover any highlight for the full per-probe breakdown.
        </p>
      </section>

      <section className="rail-section">
        <h3>This probe</h3>
        <p className="rail-probe-name">{activeProbe ?? "—"}</p>
        <p className="muted">
          Trained to detect direct investment advice (specific assets,
          allocations, buy / sell calls) issued without a disclaimer.
        </p>
      </section>

      <section className="rail-section">
        <h3>Production routing</h3>
        <p className="muted">
          In a real deployment, every response would be classified before
          delivery using the probe's peak score.
        </p>
        <ul className="routing-list">
          <li>
            <span className="routing-dot routing-allowed" />
            <div>
              <strong>ALLOWED</strong>
              <span className="muted"> — no fires above threshold</span>
            </div>
          </li>
          <li>
            <span className="routing-dot routing-monitored" />
            <div>
              <strong>MONITORED</strong>
              <span className="muted">
                {" "}
                — fires below {ESCALATION_THRESHOLD.toFixed(2)}, logged
              </span>
            </div>
          </li>
          <li>
            <span className="routing-dot routing-escalated" />
            <div>
              <strong>ESCALATED</strong>
              <span className="muted">
                {" "}
                — peak ≥ {ESCALATION_THRESHOLD.toFixed(2)}, sent to human
                review
              </span>
            </div>
          </li>
        </ul>
      </section>

      <section className="rail-section rail-status">
        <h3>This response</h3>
        <div className={verdictClass}>{status.verdict}</div>
        {status.verdict !== "PENDING" && (
          <div className="rail-metrics">
            <div className="rail-metric">
              <div className="rail-metric-label">peak score</div>
              <div className="rail-metric-value">{status.peak.toFixed(2)}</div>
            </div>
            <div className="rail-metric">
              <div className="rail-metric-label">tokens fired</div>
              <div className="rail-metric-value">
                {status.fires}
                <span className="rail-metric-of"> / {status.total}</span>
              </div>
            </div>
          </div>
        )}
        <p className="rail-verdict-note">{VERDICT_COPY[status.verdict]}</p>
      </section>
    </aside>
  );
}
