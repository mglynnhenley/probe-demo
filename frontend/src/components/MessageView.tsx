import type { DisplayMessage } from "../types";
import { MessageMarkdown } from "./MessageMarkdown";

interface Props {
  message: DisplayMessage;
  activeProbe: string | null;
  threshold: number;
  focused: boolean;
  onFocus?: () => void;
}

export function MessageView({ message, activeProbe, threshold, focused, onFocus }: Props) {
  if (message.role === "user") {
    return (
      <div className="message user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const className = [
    "message",
    "assistant",
    focused ? "focused" : "",
    onFocus ? "clickable" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={className}>
      <div
        className="bubble"
        role={onFocus ? "button" : undefined}
        tabIndex={onFocus ? 0 : undefined}
        aria-pressed={onFocus ? focused : undefined}
        onClick={onFocus}
        onKeyDown={(e) => {
          if (!onFocus) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onFocus();
          }
        }}
      >
        {message.tokens.length === 0 && !message.done ? (
          <span className="cursor">▍</span>
        ) : (
          <MessageMarkdown
            tokens={message.tokens}
            activeProbe={activeProbe}
            threshold={threshold}
          />
        )}
        {!message.done && message.tokens.length > 0 && (
          <span className="cursor">▍</span>
        )}
      </div>
    </div>
  );
}
