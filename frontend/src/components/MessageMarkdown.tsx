import React from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ScoredToken } from "../types";
import { fireStyle } from "./fireStyle";

interface Props {
  tokens: ScoredToken[];
  activeProbe: string | null;
  threshold: number;
}

interface AstNode {
  type: string;
  value?: string;
  children?: AstNode[];
  position?: { start: { offset: number }; end: { offset: number } };
}

interface State {
  fullText: string;
  posToTokenIdx: number[];
  tokens: ScoredToken[];
  activeProbe: string | null;
  threshold: number;
}

function buildState(
  tokens: ScoredToken[],
  activeProbe: string | null,
  threshold: number,
): State {
  let fullText = "";
  const posToTokenIdx: number[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i].text;
    for (let c = 0; c < t.length; c++) posToTokenIdx.push(i);
    fullText += t;
  }
  return { fullText, posToTokenIdx, tokens, activeProbe, threshold };
}

// Each markdown component override carries an __mdTag pointing to its HTML
// equivalent. When walking encounters one of our overrides as a child, we
// render its content directly via the HTML tag — so React doesn't re-invoke
// our override on the already-walked subtree.
type OverrideFn = ((props: object) => React.ReactNode) & { __mdTag?: string };

function walkChildren(
  reactChildren: React.ReactNode,
  astChildren: AstNode[] | undefined,
  s: State,
): React.ReactNode {
  return React.Children.map(reactChildren, (child, i) =>
    walkOne(child, astChildren?.[i], s, i),
  );
}

function walkOne(
  child: React.ReactNode,
  astNode: AstNode | undefined,
  s: State,
  key: React.Key,
): React.ReactNode {
  if (typeof child === "string") {
    const offset = astNode?.position?.start?.offset;
    if (offset === undefined) return child;
    return renderHighlightedString(child, offset, s, key);
  }
  if (typeof child === "number") return String(child);
  if (React.isValidElement(child)) {
    const props = child.props as {
      children?: React.ReactNode;
      node?: AstNode;
      [k: string]: unknown;
    };
    const elNode = props.node ?? astNode;
    const typeAny = child.type as unknown;
    const tag =
      typeof typeAny === "function"
        ? (typeAny as OverrideFn).__mdTag
        : undefined;
    if (tag) {
      const { children, node: _n, ...rest } = props;
      return React.createElement(
        tag,
        { ...rest, key },
        walkChildren(children, elNode?.children, s),
      );
    }
    return React.cloneElement(
      child as React.ReactElement<{ children?: React.ReactNode }>,
      { key },
      walkChildren(props.children, elNode?.children, s),
    );
  }
  return child;
}

// Locate text leaf at its known offset in fullText. Splits on token boundaries
// so each contiguous run sharing a token index renders as one span.
function renderHighlightedString(
  text: string,
  offset: number,
  s: State,
  key: React.Key,
): React.ReactNode {
  if (text.length === 0) return text;
  // Sanity-check: if the offset doesn't match (e.g., escape sequences in source
  // shorten in render), bail out and render unhighlighted rather than mismark.
  if (s.fullText.substr(offset, text.length) !== text) return text;

  const out: React.ReactNode[] = [];
  let runStart = 0;
  let runTokenIdx = s.posToTokenIdx[offset];
  for (let i = 1; i < text.length; i++) {
    const idx = s.posToTokenIdx[offset + i];
    if (idx !== runTokenIdx) {
      out.push(
        makeSpan(text.slice(runStart, i), runTokenIdx, s, `${key}-${runStart}`),
      );
      runStart = i;
      runTokenIdx = idx;
    }
  }
  out.push(
    makeSpan(text.slice(runStart), runTokenIdx, s, `${key}-${runStart}`),
  );
  return out;
}

function makeSpan(
  text: string,
  tokenIdx: number | undefined,
  s: State,
  key: React.Key,
): React.ReactNode {
  if (!s.activeProbe || tokenIdx === undefined) return text;
  const tok = s.tokens[tokenIdx];
  const score = tok?.scores[s.activeProbe];
  if (score === undefined) return text;
  const style = fireStyle(score, s.threshold);
  if (!style) return text;
  const tooltip = Object.entries(tok.scores)
    .map(([n, v]) => `${n}: ${v.toFixed(3)}`)
    .join("\n");
  return (
    <span key={key} style={style} title={tooltip || undefined}>
      {text}
    </span>
  );
}

const PASSTHRU_TAGS = [
  "p",
  "li",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "strong",
  "em",
  "del",
  "code",
  "pre",
  "blockquote",
  "td",
  "th",
  "a",
] as const;

function buildComponents(s: State): Components {
  const out: Components = {};
  for (const tag of PASSTHRU_TAGS) {
    const fn: OverrideFn = ({
      children,
      node,
      ...rest
    }: {
      children?: React.ReactNode;
      node?: AstNode;
    }) =>
      React.createElement(tag, rest, walkChildren(children, node?.children, s));
    fn.__mdTag = tag;
    (out as Record<string, unknown>)[tag] = fn;
  }
  return out;
}

export function MessageMarkdown({ tokens, activeProbe, threshold }: Props) {
  const state = buildState(tokens, activeProbe, threshold);
  const components = buildComponents(state);
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {state.fullText}
    </ReactMarkdown>
  );
}
