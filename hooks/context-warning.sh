#!/bin/bash
# Context Window Warning Hook v2 (UserPromptSubmit)
# 
# Uses Letta API to get actual context window size and message count,
# then calculates dynamic thresholds based on the agent's real configuration.
# No more hardcoded message counts — works correctly for any context window size.
#
# Data from API: context_window (tokens), message_ids (count), block char totals
# Estimate: ~4 chars per token (rough but better than fixed thresholds)
#
# Supports both Letta accounts (audre + replyomatic).
# Install: Add to ~/.letta/settings.json under hooks.UserPromptSubmit

export PATH="/home/a/.local/bin:$PATH"

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

# Both account API keys
KEYS=(
    "YOUR_LETTA_API_KEY_HERE"
    "YOUR_LETTA_API_KEY_HERE"
)

# Try each key until one works — get full agent data
agent_json=""
for key in "${KEYS[@]}"; do
    result=$(curl -s --max-time 5 \
        -H "Authorization: Bearer ${key}" \
        "https://api.letta.com/v1/agents/${agent_id}" 2>/dev/null)
    
    # Check if we got valid JSON with an id field
    check=$(echo "$result" | jq -r '.id // empty' 2>/dev/null)
    if [ -n "$check" ]; then
        agent_json="$result"
        break
    fi
done

if [ -z "$agent_json" ]; then
    exit 0
fi

# Extract key metrics
context_window=$(echo "$agent_json" | jq '.llm_config.context_window // 0' 2>/dev/null)
msg_count=$(echo "$agent_json" | jq '.message_ids | length' 2>/dev/null)
block_chars=$(echo "$agent_json" | jq '[.blocks[]?.value // "" | length] | add // 0' 2>/dev/null)
system_chars=$(echo "$agent_json" | jq '.system | length' 2>/dev/null)

# Safety: if we couldn't get context_window, fall back to message-count heuristic
if [ -z "$context_window" ] || [ "$context_window" -eq 0 ] 2>/dev/null; then
    # Fallback: old behavior with fixed thresholds
    if [ "$msg_count" -gt 110 ]; then
        cat <<WARN >&2
CONTEXT WINDOW CRITICAL (${msg_count} messages, context_window unknown). COMPACTION IS IMMINENT.
ACTION REQUIRED NOW: Save your current working state to archival memory.
WARN
    elif [ "$msg_count" -gt 85 ]; then
        cat <<WARN >&2
CONTEXT WINDOW WARNING (${msg_count} messages, context_window unknown). Consider saving state.
WARN
    fi
    exit 0
fi

# Estimate token usage from fixed content (blocks + system prompt)
# ~4 chars per token is a rough estimate for English text
fixed_tokens=$(( (block_chars + system_chars) / 4 ))

# Available tokens for messages = context_window - fixed_tokens - output_reserve
# Reserve ~8k tokens for model output
output_reserve=8000
available_for_messages=$(( context_window - fixed_tokens - output_reserve ))

# Estimate tokens per message (observed average: ~300-500 tokens/message)
# Use 400 as middle estimate
tokens_per_msg=400
estimated_msg_tokens=$(( msg_count * tokens_per_msg ))

# Calculate usage percentage
if [ "$available_for_messages" -gt 0 ]; then
    usage_pct=$(( (estimated_msg_tokens * 100) / available_for_messages ))
else
    usage_pct=100
fi

# Dynamic thresholds
if [ "$usage_pct" -gt 85 ]; then
    cat <<WARN >&2
CONTEXT WINDOW CRITICAL (~${usage_pct}% capacity, ${msg_count} messages, ${context_window} token window).
COMPACTION IS IMMINENT. ACTION REQUIRED NOW: Save your current working state to archival memory.
Include: (1) current task and step, (2) next planned action, (3) key file paths, (4) any decisions or findings in progress.
After compaction, search archival for tag "todo-recovery" or "compaction-recovery" to find your place.
WARN
    exit 0
elif [ "$usage_pct" -gt 70 ]; then
    cat <<WARN >&2
CONTEXT WINDOW WARNING (~${usage_pct}% capacity, ${msg_count} messages, ${context_window} token window).
Compaction may occur soon. Consider saving your current working state to archival memory: current task, step, next action, key file paths.
WARN
    exit 0
fi

exit 0
