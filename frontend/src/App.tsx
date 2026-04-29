import { useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_BACKEND, DEFAULT_MODEL, fetchModel, streamChat } from "./api";
import { EscalationRail } from "./components/EscalationRail";
import { MessageView } from "./components/MessageView";
import { ProbePanel } from "./components/ProbePanel";
import type { AssistantMessage, ChatMessage, DisplayMessage } from "./types";

const STORAGE_KEY = "probe-demo.backendUrl.v2";
const DEFAULT_THRESHOLD = 0.5;

function newId() {
  return Math.random().toString(36).slice(2, 10);
}

export function App() {
  const [backendUrl, setBackendUrl] = useState<string>(() => {
    return localStorage.getItem(STORAGE_KEY) ?? DEFAULT_BACKEND;
  });
  const [serverModel, setServerModel] = useState<string | null>(null);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeProbe, setActiveProbe] = useState<string | null>(null);
  const [thresholds, setThresholds] = useState<Record<string, number>>({});
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, backendUrl);
    fetchModel(backendUrl).then(setServerModel);
  }, [backendUrl]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const probeNames = useMemo(() => {
    const seen = new Set<string>();
    for (const m of messages) {
      if (m.role !== "assistant") continue;
      for (const tok of m.tokens) {
        for (const k of Object.keys(tok.scores)) seen.add(k);
      }
    }
    return Array.from(seen).sort();
  }, [messages]);

  // Default-fill any new probe with the standard threshold.
  useEffect(() => {
    setThresholds((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const name of probeNames) {
        if (next[name] === undefined) {
          next[name] = DEFAULT_THRESHOLD;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [probeNames]);

  useEffect(() => {
    if (activeProbe === null && probeNames.length > 0) {
      setActiveProbe(probeNames[0]);
    }
  }, [probeNames, activeProbe]);

  // Resolved focused message: explicit click, else most recent assistant.
  const focusedMessage = useMemo<AssistantMessage | null>(() => {
    if (focusedId) {
      const m = messages.find((m) => m.id === focusedId);
      if (m && m.role === "assistant") return m;
    }
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant") return m;
    }
    return null;
  }, [messages, focusedId]);

  function setThreshold(name: string, value: number) {
    setThresholds((prev) => ({ ...prev, [name]: value }));
  }

  async function handleSend() {
    const trimmed = input.trim();
    if (!trimmed || streaming) return;

    const userMsg: DisplayMessage = {
      id: newId(),
      role: "user",
      content: trimmed,
    };
    const assistantMsg: AssistantMessage = {
      id: newId(),
      role: "assistant",
      tokens: [],
      done: false,
    };
    const next = [...messages, userMsg, assistantMsg];
    setMessages(next);
    setFocusedId(assistantMsg.id);
    setInput("");
    setStreaming(true);
    setError(null);

    const history: ChatMessage[] = next
      .slice(0, -1)
      .map((m) =>
        m.role === "user"
          ? { role: "user" as const, content: m.content }
          : {
              role: "assistant" as const,
              content: m.tokens.map((t) => t.text).join(""),
            },
      );

    const controller = new AbortController();
    abortRef.current = controller;

    await streamChat(
      backendUrl,
      history,
      {
        onToken: (text, scores) => {
          setMessages((prev) => {
            const copy = prev.slice();
            const last = copy[copy.length - 1];
            if (last?.role !== "assistant") return prev;
            const updated: AssistantMessage = {
              ...last,
              tokens: [...last.tokens, { text, scores: scores ?? {} }],
            };
            copy[copy.length - 1] = updated;
            return copy;
          });
        },
        onDone: () => {
          setStreaming(false);
          setMessages((prev) => {
            const copy = prev.slice();
            const last = copy[copy.length - 1];
            if (last?.role === "assistant") {
              copy[copy.length - 1] = { ...last, done: true };
            }
            return copy;
          });
        },
        onError: (err) => {
          setError(err.message);
          setStreaming(false);
          setMessages((prev) => {
            const copy = prev.slice();
            const last = copy[copy.length - 1];
            if (last?.role === "assistant") {
              copy[copy.length - 1] = { ...last, done: true };
            }
            return copy;
          });
        },
      },
      controller.signal,
    );
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  function handleReset() {
    abortRef.current?.abort();
    setMessages([]);
    setFocusedId(null);
    setError(null);
  }

  const activeThreshold =
    activeProbe && thresholds[activeProbe] !== undefined
      ? thresholds[activeProbe]
      : DEFAULT_THRESHOLD;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <h1>Probe Demo</h1>
          <span className="subtitle">chatting with {DEFAULT_MODEL}</span>
        </div>
        <div className="topbar-controls">
          <label className="backend-input">
            Backend
            <input
              type="text"
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              spellCheck={false}
            />
          </label>
          <span className="server-model">
            {serverModel ? `serving ${serverModel}` : "backend offline"}
          </span>
          <button
            className="ghost"
            onClick={handleReset}
            disabled={streaming && messages.length === 0}
          >
            New chat
          </button>
        </div>
      </header>

      <main className="main">
        <EscalationRail
          focusedMessage={focusedMessage}
          activeProbe={activeProbe}
          highlightThreshold={activeThreshold}
        />

        <section className="chat">
          <div className="transcript" ref={transcriptRef}>
            {messages.length === 0 && (
              <div className="empty-state">
                <h2>Start the conversation</h2>
                <p>
                  Send a message and watch the probes flag tokens above the
                  threshold. Click any past response to inspect its scores.
                </p>
              </div>
            )}
            {messages.map((m) => (
              <MessageView
                key={m.id}
                message={m}
                activeProbe={activeProbe}
                threshold={activeThreshold}
                focused={m.id === focusedMessage?.id}
                onFocus={
                  m.role === "assistant" ? () => setFocusedId(m.id) : undefined
                }
              />
            ))}
            {error && (
              <div className="error">
                <strong>Error.</strong> {error}
              </div>
            )}
          </div>

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Message gpt-4.1…"
              rows={2}
              disabled={streaming}
            />
            {streaming ? (
              <button type="button" className="primary" onClick={handleStop}>
                Stop
              </button>
            ) : (
              <button type="submit" className="primary" disabled={!input.trim()}>
                Send
              </button>
            )}
          </form>
        </section>

        <ProbePanel
          message={focusedMessage}
          probeNames={probeNames}
          activeProbe={activeProbe}
          onSelectProbe={setActiveProbe}
          thresholds={thresholds}
          onThresholdChange={setThreshold}
        />
      </main>
    </div>
  );
}
