#!/usr/bin/env bash
# Smoke tests for the probe-demo hallucination backend.
# Usage:
#   bash backend/smoke_test.sh [BASE_URL]
# Default BASE_URL targets the deployed Modal endpoint.
set -uo pipefail

BASE="${1:-https://nesmith55555555--probe-demo-hallucination-api.modal.run}"
PASS=0
FAIL=0

# ── helpers ────────────────────────────────────────────────────────────────

green()  { printf '\033[32m✓ %s\033[0m\n' "$*"; }
red()    { printf '\033[31m✗ %s\033[0m\n' "$*"; }

pass() { green "$1"; ((PASS++)) || true; }
fail() { red   "$1"; ((FAIL++)) || true; }

# assert_json TEST_NAME JQ_FILTER EXPECTED JSON
assert_json() {
    local name="$1" filter="$2" expected="$3" json="$4"
    local actual
    actual=$(echo "$json" | jq -r "$filter" 2>/dev/null)
    if [[ "$actual" == "$expected" ]]; then
        pass "$name"
    else
        fail "$name  (expected '$expected', got '$actual')"
    fi
}

# assert_jq_true TEST_NAME JQ_CONDITION JSON  — condition must return 'true'
assert_jq_true() {
    local name="$1" cond="$2" json="$3"
    local result
    result=$(echo "$json" | jq "$cond" 2>/dev/null)
    if [[ "$result" == "true" ]]; then
        pass "$name"
    else
        fail "$name  (condition '$cond' was '$result')"
    fi
}

post() {
    # On non-2xx, writes the body to stderr (so it's visible) and echoes 'null'
    # so downstream jq calls don't crash the script.
    local body status
    body=$(curl -s -w '\n__STATUS__:%{http_code}' -X POST "$BASE$1" \
        -H "Content-Type: application/json" \
        -d "$2")
    status=$(echo "$body" | sed -n 's/^__STATUS__://p' | tail -1)
    body=$(echo "$body" | sed '/^__STATUS__:/d')
    if [[ "$status" != "200" ]]; then
        echo "  [HTTP $status] $body" >&2
        echo "null"
        return
    fi
    echo "$body"
}

post_stream() {
    # Returns the raw SSE lines (strips 'data: ' prefix, drops [DONE])
    curl -s -X POST "$BASE$1" \
        -H "Content-Type: application/json" \
        -H "Accept: text/event-stream" \
        -d "$2" \
    | grep '^data:' | grep -v '\[DONE\]' | sed 's/^data: //'
}

# ── 1. Health ───────────────────────────────────────────────────────────────

echo; echo "── health ──────────────────────────────────────────────────"
HEALTH=$(curl -s "$BASE/health")
assert_json "GET /health → {status: ok}" '.status' 'ok' "$HEALTH"

# ── 2. Models list ──────────────────────────────────────────────────────────

echo; echo "── /v1/models ──────────────────────────────────────────────"
MODELS=$(curl -s "$BASE/v1/models")
assert_json "GET /v1/models object=list"    '.object'       'list'  "$MODELS"
assert_jq_true "GET /v1/models has ≥1 model" '(.data | length) >= 1' "$MODELS"
MODEL_ID=$(echo "$MODELS" | jq -r '.data[0].id')
pass "  model id = $MODEL_ID"

# ── 3. Non-streaming generate with scores ───────────────────────────────────

echo; echo "── generate (non-streaming) ────────────────────────────────"
GEN=$(post /v1/chat/completions '{
    "model": "default",
    "messages": [{"role": "user", "content": "Say the single word: hello"}],
    "include_scores": true,
    "max_tokens": 10,
    "temperature": 0.0
}')
assert_json  "generate: object=chat.completion"     '.object'              'chat.completion' "$GEN"
assert_json  "generate: choices[0].finish_reason"   '.choices[0].finish_reason' 'stop'       "$GEN"
assert_jq_true "generate: content non-empty"        '(.choices[0].message.content | length) > 0' "$GEN"
assert_jq_true "generate: scores key present"       '.scores != null'                            "$GEN"
assert_jq_true "generate: scores has ≥1 probe key"  '(.scores | keys | length) >= 1'             "$GEN"
PROBE_KEY=$(echo "$GEN" | jq -r '.scores | keys[0]')
N_TOKENS=$(echo "$GEN" | jq ".usage.completion_tokens")
N_SCORES=$(echo "$GEN" | jq ".scores[\"$PROBE_KEY\"] | length")
if [[ "$N_TOKENS" == "$N_SCORES" ]]; then
    pass "generate: score count == completion_tokens ($N_SCORES)"
else
    fail "generate: score count $N_SCORES ≠ completion_tokens $N_TOKENS"
fi
assert_jq_true "generate: all scores in [0,1]" \
    "[.scores[\"$PROBE_KEY\"][] | (. >= 0 and . <= 1)] | all" "$GEN"

# ── 4. Non-streaming generate without scores ────────────────────────────────

echo; echo "── generate (include_scores=false) ─────────────────────────"
GEN_NOSCORES=$(post /v1/chat/completions '{
    "model": "default",
    "messages": [{"role": "user", "content": "Say the single word: hello"}],
    "include_scores": false,
    "max_tokens": 10,
    "temperature": 0.0
}')
assert_json  "generate no-scores: object=chat.completion" '.object' 'chat.completion' "$GEN_NOSCORES"
assert_json  "generate no-scores: scores is null"         '.scores' 'null'            "$GEN_NOSCORES"

# ── 5. Streaming generate ───────────────────────────────────────────────────

echo; echo "── generate (streaming) ─────────────────────────────────────"
STREAM_LINES=$(post_stream /v1/chat/completions '{
    "model": "default",
    "messages": [{"role": "user", "content": "Say the single word: hello"}],
    "include_scores": true,
    "max_tokens": 10,
    "temperature": 0.0,
    "stream": true
}')
FIRST=$(echo "$STREAM_LINES" | head -1)
assert_json "stream: first chunk object=chat.completion.chunk" '.object' 'chat.completion.chunk' "$FIRST"
assert_json "stream: first chunk has role=assistant" '.choices[0].delta.role' 'assistant' "$FIRST"

# Content chunks carry a score; check at least one does
CONTENT_WITH_SCORE=$(echo "$STREAM_LINES" | jq -s '[.[] | select(.scores != null)] | length')
if (( CONTENT_WITH_SCORE > 0 )); then
    pass "stream: ≥1 chunk carries scores ($CONTENT_WITH_SCORE)"
else
    fail "stream: no chunks carried scores"
fi

LAST=$(echo "$STREAM_LINES" | tail -1)
assert_json "stream: last chunk finish_reason=stop" '.choices[0].finish_reason' 'stop' "$LAST"

# ── 6. Analyze (non-streaming) ──────────────────────────────────────────────

echo; echo "── analyze (non-streaming) ──────────────────────────────────"
ASSISTANT_TEXT="The capital of France is Paris."
N_EXPECTED=$(echo -n "$ASSISTANT_TEXT" | wc -c)  # rough lower bound; actual is token count

ANALYZE=$(post /v1/chat/completions "{
    \"model\": \"default-analyze\",
    \"messages\": [
        {\"role\": \"user\",      \"content\": \"What is the capital of France?\"},
        {\"role\": \"assistant\", \"content\": \"$ASSISTANT_TEXT\"}
    ],
    \"include_scores\": true,
    \"max_tokens\": 1
}")
assert_json  "analyze: object=chat.completion"           '.object'  'chat.completion' "$ANALYZE"
assert_jq_true "analyze: scores key present"             '.scores != null'             "$ANALYZE"
assert_jq_true "analyze: scores has ≥1 probe key"        '(.scores | keys | length) >= 1' "$ANALYZE"
AK=$(echo "$ANALYZE" | jq -r '.scores | keys[0]')
AN=$(echo "$ANALYZE" | jq ".scores[\"$AK\"] | length")
assert_jq_true "analyze: score count > 0 ($AN scores)" "(${AN} > 0)" "$ANALYZE"
assert_jq_true "analyze: all scores in [0,1]" \
    "[.scores[\"$AK\"][] | (. >= 0 and . <= 1)] | all" "$ANALYZE"
# Echo text should be the assistant message
assert_json "analyze: echoed assistant content" \
    '.choices[0].message.content' "$ASSISTANT_TEXT" "$ANALYZE"

# ── 7. Analyze (streaming) ──────────────────────────────────────────────────

echo; echo "── analyze (streaming) ──────────────────────────────────────"
ANALYZE_STREAM=$(post_stream /v1/chat/completions "{
    \"model\": \"default-analyze\",
    \"messages\": [
        {\"role\": \"user\",      \"content\": \"What is the capital of France?\"},
        {\"role\": \"assistant\", \"content\": \"$ASSISTANT_TEXT\"}
    ],
    \"include_scores\": true,
    \"max_tokens\": 1,
    \"stream\": true
}")
AFIRST=$(echo "$ANALYZE_STREAM" | head -1)
assert_json "analyze stream: first chunk role=assistant" '.choices[0].delta.role' 'assistant' "$AFIRST"
ALAST=$(echo "$ANALYZE_STREAM" | tail -1)
assert_json "analyze stream: last chunk finish_reason=stop" '.choices[0].finish_reason' 'stop' "$ALAST"
A_SCORED=$(echo "$ANALYZE_STREAM" | jq -s '[.[] | select(.scores != null)] | length')
if (( A_SCORED > 0 )); then
    pass "analyze stream: ≥1 chunk carries scores ($A_SCORED)"
else
    fail "analyze stream: no chunks carried scores"
fi

# ── 8. Analyze: empty assistant message → 422 ───────────────────────────────

echo; echo "── edge cases ───────────────────────────────────────────────"
HTTP_STATUS=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default-analyze",
        "messages": [{"role": "user", "content": "hello"}],
        "include_scores": true,
        "max_tokens": 1
    }')
if [[ "$HTTP_STATUS" == "422" ]]; then
    pass "analyze with no assistant message → 422"
else
    fail "analyze with no assistant message → expected 422, got $HTTP_STATUS"
fi

# ── 9. Usage counts are consistent ──────────────────────────────────────────

echo; echo "── usage consistency ────────────────────────────────────────"
assert_jq_true "generate: total_tokens = prompt + completion" \
    '.usage.total_tokens == (.usage.prompt_tokens + .usage.completion_tokens)' "$GEN"

# ── summary ─────────────────────────────────────────────────────────────────

echo
echo "────────────────────────────────────────────────────────────"
echo "  passed: $PASS   failed: $FAIL"
echo "────────────────────────────────────────────────────────────"
[[ $FAIL -eq 0 ]]
