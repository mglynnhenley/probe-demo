export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

// One generated token plus the per-probe scores emitted with that token's chunk.
export interface ScoredToken {
  text: string;
  scores: Record<string, number>;
}

// An assistant message rendered as a sequence of scored tokens.
export interface AssistantMessage {
  id: string;
  role: "assistant";
  tokens: ScoredToken[];
  done: boolean;
}

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
}

export type DisplayMessage = UserMessage | AssistantMessage;
