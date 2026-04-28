import type { ChatMessage } from "./types";

export const DEFAULT_BACKEND = "/api";
export const DEFAULT_MODEL = "gpt-4.1";

const PROBE_NAME = "financial_advice";

interface ProbeResponse {
  completion_prob: number;
  completion_logit: number;
  threshold: number;
  flagged: boolean;
  n_tokens: number;
  tokens: string[];
  token_probs: number[];
}

interface ChatCompletionResponse {
  id: string;
  model: string;
  choices: Array<{
    message: { role: string; content: string };
    finish_reason: string;
  }>;
  probe?: ProbeResponse;
}

export interface StreamCallbacks {
  onToken: (text: string, scores: Record<string, number> | null) => void;
  onDone: () => void;
  onError: (err: Error) => void;
}

// Gemma SentencePiece marker for "this token starts a new word".
function renderTokenText(tok: string): string {
  return tok.replace(/\u2581/g, " ");
}

function isHiddenToken(tok: string): boolean {
  return (
    tok === "<end_of_turn>" ||
    tok === "<start_of_turn>" ||
    tok === "<eos>" ||
    tok === "<bos>"
  );
}

export async function streamChat(
  baseUrl: string,
  messages: ChatMessage[],
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_MODEL,
        messages,
        include_scores: true,
        temperature: 0.7,
        top_p: 0.9,
      }),
      signal,
    });
  } catch (err) {
    if ((err as Error)?.name === "AbortError") {
      callbacks.onDone();
      return;
    }
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    callbacks.onError(
      new Error(`Backend ${response.status}: ${text || response.statusText}`),
    );
    return;
  }

  let data: ChatCompletionResponse;
  try {
    data = await response.json();
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }

  const tokens = data.probe?.tokens ?? [];
  const probs = data.probe?.token_probs ?? [];

  // Backend is non-streaming: replay tokens in a microtask loop so the UI
  // still sees them arrive one-by-one and the typing cursor / probe panel
  // animate. Each onToken triggers a React state update; setTimeout(0)
  // yields to the renderer between tokens.
  for (let i = 0; i < tokens.length; i++) {
    const raw = tokens[i];
    if (isHiddenToken(raw)) continue;
    const text = renderTokenText(raw);
    const score = probs[i];
    const scores: Record<string, number> | null =
      score !== undefined ? { [PROBE_NAME]: score } : null;
    callbacks.onToken(text, scores);
    // Yield once every few tokens so React paints between batches.
    if (i % 3 === 0) {
      await new Promise((r) => setTimeout(r, 12));
    }
  }

  callbacks.onDone();
}

export async function fetchModel(baseUrl: string): Promise<string | null> {
  try {
    const res = await fetch(`${baseUrl}/openapi.json`);
    if (!res.ok) return null;
    const data = await res.json();
    return data?.info?.title ?? "online";
  } catch {
    return null;
  }
}
