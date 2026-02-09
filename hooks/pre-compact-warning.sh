#!/bin/bash
# Pre-Compaction Auto-Save Hook v2 (PreCompact)
#
# Fires immediately before context compaction occurs.
# Automatically saves working state to the agent's archival memory
# via the Letta API — no agent action needed.
#
# Captures:
#   - Agent name and ID
#   - Message count and context window size
#   - Last 10 messages (text content, truncated) for continuity
#   - Timestamp
#
# Tags the archival entry with "compaction-recovery" and "auto-save"
# so the agent can find it after waking up.
#
# Also outputs a warning to stderr so the agent knows it happened.

export PATH="/home/a/.local/bin:$PATH"

input=$(cat)

# Get agent_id from hook input
agent_id=$(echo "$input" | jq -r '.agent_id // empty' 2>/dev/null)
if [ -z "$agent_id" ]; then
    cat <<WARN >&2
COMPACTION IS HAPPENING NOW. Auto-save failed: no agent_id in hook input.
WARN
    exit 0
fi

# Both account API keys
KEYS=(
    "YOUR_LETTA_API_KEY_HERE"
    "YOUR_LETTA_API_KEY_HERE"
)

# Find the right key and get agent info
agent_json=""
working_key=""
for key in "${KEYS[@]}"; do
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
    cat <<WARN >&2
COMPACTION IS HAPPENING NOW. Auto-save failed: could not reach API for agent ${agent_id}.
WARN
    exit 0
fi

# Extract agent info
agent_name=$(echo "$agent_json" | jq -r '.name // "unknown"' 2>/dev/null)
context_window=$(echo "$agent_json" | jq '.llm_config.context_window // 0' 2>/dev/null)
msg_count=$(echo "$agent_json" | jq '.message_ids | length' 2>/dev/null)
timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

# Get last 10 messages for context continuity
last_messages=$(curl -s --max-time 5 \
    -H "Authorization: Bearer ${working_key}" \
    "https://api.letta.com/v1/agents/${agent_id}/messages/?limit=10&order=desc" 2>/dev/null \
    | jq -r '[.[] | {
        type: .message_type,
        content: (
            if .content then (.content | tostring | .[0:300])
            elif .assistant_message then (.assistant_message | .[0:300])
            elif .reasoning then (.reasoning | .[0:300])
            elif .tool_call then ("tool: " + (.tool_call.name // "?"))
            elif .tool_return then (.tool_return | tostring | .[0:200])
            else "—"
            end
        )
    }] | map("[\(.type)] \(.content)") | join("\n")' 2>/dev/null)

# Build the archival entry
archival_text="## Auto-Save: Pre-Compaction Snapshot
**Agent:** ${agent_name} (${agent_id})
**Time:** ${timestamp}
**Context:** ${msg_count} messages in ${context_window} token window
**Trigger:** Automatic pre-compaction hook

### Last 10 Messages (most recent first):
${last_messages}

### Recovery Instructions:
This was saved automatically by the pre-compaction hook. Your manual archival saves (tagged todo-recovery or compaction-recovery) will have more detailed working state if you made them. Check both."

# Write to archival memory via API
archival_payload=$(jq -n \
    --arg text "$archival_text" \
    '{
        "text": $text,
        "tags": ["compaction-recovery", "auto-save"]
    }')

save_result=$(curl -s --max-time 5 \
    -X POST \
    -H "Authorization: Bearer ${working_key}" \
    -H "Content-Type: application/json" \
    -d "$archival_payload" \
    "https://api.letta.com/v1/agents/${agent_id}/archival-memory" 2>/dev/null)

save_id=$(echo "$save_result" | jq -r '.id // empty' 2>/dev/null)

if [ -n "$save_id" ]; then
    cat <<WARN >&2
COMPACTION IS HAPPENING NOW. Working state auto-saved to archival memory (${save_id}). After compaction, search archival for tags "compaction-recovery" or "auto-save" to recover context.
WARN
else
    cat <<WARN >&2
COMPACTION IS HAPPENING NOW. Auto-save attempted but may have failed. Search archival for "compaction-recovery" after waking up. Error: $(echo "$save_result" | head -c 200)
WARN
fi

exit 0
