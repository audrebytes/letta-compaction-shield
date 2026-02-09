#!/bin/bash
# Compaction-Rx: Context Warning + Compaction Detection Hook (UserPromptSubmit)
# v0.2 alpha
#
# Two jobs:
# 1. COMPACTION DETECTION: Compares current message count to cached count.
#    If it dropped significantly, compaction just happened. Grabs the summary
#    message and saves it to archival memory — full, untruncated, permanent.
#
# 2. CONTEXT WARNING: Queries API for actual context window size and message
#    count, calculates dynamic percentage-based thresholds. Warns agent when
#    context is filling up.
#
# IMPORTANT: Token estimates are ROUGH (~4 chars/token, ~400 tokens/message).
# Real usage may differ significantly. For exact token counts, use the Letta
# Python SDK: client.runs.usage.retrieve(run_id=...) — see README.
#
# Configuration (environment variables — set in your shell or .env file):
#   LETTA_API_KEY       - Single API key (required)
#   LETTA_API_KEYS      - Comma-separated keys for multiple accounts (optional)
#   CRX_WARN_PCT        - Warning threshold percentage (default: 70)
#   CRX_CRIT_PCT        - Critical threshold percentage (default: 85)
#   CRX_MSG_DROP        - Message count drop to detect compaction (default: 30)
#   CRX_FALLBACK_WARN   - Fallback message count for warning (default: 85)
#   CRX_FALLBACK_CRIT   - Fallback message count for critical (default: 110)
#   CRX_TOKENS_PER_MSG  - Estimated tokens per message (default: 400)
#   CRX_CHARS_PER_TOKEN - Estimated chars per token (default: 4)
#   CRX_OUTPUT_RESERVE  - Tokens reserved for model output (default: 8000)
#
# Install: Add to ~/.letta/settings.json under hooks.UserPromptSubmit
# See README for full setup instructions.

# --- Configuration ---
WARN_PCT="${CRX_WARN_PCT:-70}"
CRIT_PCT="${CRX_CRIT_PCT:-85}"
MSG_DROP_THRESHOLD="${CRX_MSG_DROP:-30}"
FALLBACK_WARN="${CRX_FALLBACK_WARN:-85}"
FALLBACK_CRIT="${CRX_FALLBACK_CRIT:-110}"
TOKENS_PER_MSG="${CRX_TOKENS_PER_MSG:-400}"
CHARS_PER_TOKEN="${CRX_CHARS_PER_TOKEN:-4}"
OUTPUT_RESERVE="${CRX_OUTPUT_RESERVE:-8000}"

CACHE_DIR="/tmp/letta-context-cache"
mkdir -p "$CACHE_DIR"

input=$(cat)
event_type=$(echo "$input" | jq -r '.event_type // empty')

# Only process UserPromptSubmit
if [ "$event_type" != "UserPromptSubmit" ]; then
    exit 0
fi

# Get agent_id from hook input
agent_id=$(echo "$input" | jq -r '.agent_id // empty')
if [ -z "$agent_id" ]; then
    exit 0
fi

# --- Build API key list ---
# Reads from LETTA_API_KEYS (comma-separated) or LETTA_API_KEY (single)
KEYS=()
if [ -n "${LETTA_API_KEYS:-}" ]; then
    IFS=',' read -ra KEYS <<< "$LETTA_API_KEYS"
elif [ -n "${LETTA_API_KEY:-}" ]; then
    KEYS=("$LETTA_API_KEY")
else
    # No key configured — can't query API, exit silently
    exit 0
fi

# Try each key until one works — get full agent data
agent_json=""
working_key=""
for key in "${KEYS[@]}"; do
    key=$(echo "$key" | xargs)  # trim whitespace
    result=$(curl -s --max-time 5 \
        -H "Authorization: Bearer ${key}" \
        "https://api.letta.com/v1/agents/${agent_id}" 2>/dev/null)
    
    check=$(echo "$result" | jq -r '.id // empty' 2>/dev/null)
    if [ -n "$check" ]; then
        agent_json="$result"
        working_key="$key"
        break
    fi
done

if [ -z "$agent_json" ]; then
    exit 0
fi

# Extract key metrics
agent_name=$(echo "$agent_json" | jq -r '.name // "unknown"' 2>/dev/null)
context_window=$(echo "$agent_json" | jq '.llm_config.context_window // 0' 2>/dev/null)
msg_count=$(echo "$agent_json" | jq '.message_ids | length' 2>/dev/null)
block_chars=$(echo "$agent_json" | jq '[.blocks[]?.value // "" | length] | add // 0' 2>/dev/null)
system_chars=$(echo "$agent_json" | jq '.system | length' 2>/dev/null)

# =====================================================================
# JOB 1: COMPACTION DETECTION — save summary to archival if compaction
#         just happened
# =====================================================================

cache_file="${CACHE_DIR}/${agent_id}"
compaction_detected=false

if [ -f "$cache_file" ]; then
    prev_count=$(cat "$cache_file" 2>/dev/null)
    if [ -n "$prev_count" ] && [ "$prev_count" -gt 0 ] 2>/dev/null; then
        drop=$(( prev_count - msg_count ))
        if [ "$drop" -gt "$MSG_DROP_THRESHOLD" ]; then
            compaction_detected=true
            timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
            
            # Get the first few messages — the compaction summary is typically
            # a system_message near the start of the remaining context
            summary_content=$(curl -s --max-time 5 \
                -H "Authorization: Bearer ${working_key}" \
                "https://api.letta.com/v1/agents/${agent_id}/messages/?limit=5&order=asc" 2>/dev/null \
                | jq -r '[.[] | select(.message_type == "system_message" or .message_type == "user_message") | {type: .message_type, content: .content}] | map("[\(.type)]\n\(.content)") | join("\n\n---\n\n")' 2>/dev/null)
            
            if [ -n "$summary_content" ] && [ "$summary_content" != "null" ]; then
                archival_text="Compaction auto-save. Agent: ${agent_name} (${agent_id}). Time: ${timestamp}. Context: ${prev_count} messages compacted to ${msg_count} remaining (${context_window} token window). Summary content follows. ${summary_content}"

                archival_payload=$(jq -n \
                    --arg text "$archival_text" \
                    '{
                        "text": $text,
                        "tags": ["compaction-recovery", "compaction-summary", "auto-save"]
                    }')

                curl -s --max-time 5 \
                    -X POST \
                    -H "Authorization: Bearer ${working_key}" \
                    -H "Content-Type: application/json" \
                    -d "$archival_payload" \
                    "https://api.letta.com/v1/agents/${agent_id}/archival-memory" >/dev/null 2>&1

                cat <<WARN >&2
COMPACTION DETECTED: Context was compacted (${prev_count} → ${msg_count} messages). Full compaction summary has been auto-saved to archival memory. Search for tags "compaction-summary" or "auto-save" to retrieve it.
WARN
            fi
        fi
    fi
fi

# Update cached message count
echo "$msg_count" > "$cache_file"

# =====================================================================
# JOB 2: CONTEXT WARNING — dynamic thresholds based on real window size
# =====================================================================

# Skip warning if we just detected compaction (context just got freed)
if [ "$compaction_detected" = true ]; then
    exit 0
fi

# Safety: if we couldn't get context_window, fall back to message-count heuristic
if [ -z "$context_window" ] || [ "$context_window" -eq 0 ] 2>/dev/null; then
    if [ "$msg_count" -gt "$FALLBACK_CRIT" ]; then
        cat <<WARN >&2
CONTEXT WINDOW CRITICAL (${msg_count} messages, context_window unknown). COMPACTION IS IMMINENT.
ACTION REQUIRED NOW: Save your current working state to archival memory.
WARN
    elif [ "$msg_count" -gt "$FALLBACK_WARN" ]; then
        cat <<WARN >&2
CONTEXT WINDOW WARNING (${msg_count} messages, context_window unknown). Consider saving state.
WARN
    fi
    exit 0
fi

# Estimate token usage from fixed content (blocks + system prompt)
fixed_tokens=$(( (block_chars + system_chars) / CHARS_PER_TOKEN ))

# Available tokens for messages = context_window - fixed_tokens - output_reserve
available_for_messages=$(( context_window - fixed_tokens - OUTPUT_RESERVE ))

# Estimate message tokens
estimated_msg_tokens=$(( msg_count * TOKENS_PER_MSG ))

# Calculate usage percentage
if [ "$available_for_messages" -gt 0 ]; then
    usage_pct=$(( (estimated_msg_tokens * 100) / available_for_messages ))
else
    usage_pct=100
fi

# Dynamic thresholds
if [ "$usage_pct" -gt "$CRIT_PCT" ]; then
    cat <<WARN >&2
CONTEXT WINDOW CRITICAL (~${usage_pct}% estimated capacity, ${msg_count} messages, ${context_window} token window).
COMPACTION IS IMMINENT. ACTION REQUIRED NOW: Save your current working state to archival memory.
Include: (1) current task and step, (2) next planned action, (3) key file paths, (4) any decisions or findings in progress.
After compaction, search archival for tag "todo-recovery" or "compaction-recovery" to find your place.
WARN
    exit 0
elif [ "$usage_pct" -gt "$WARN_PCT" ]; then
    cat <<WARN >&2
CONTEXT WINDOW WARNING (~${usage_pct}% estimated capacity, ${msg_count} messages, ${context_window} token window).
Compaction may occur soon. Consider saving your current working state to archival memory: current task, step, next action, key file paths.
WARN
    exit 0
fi

exit 0
