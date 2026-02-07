#!/bin/bash
# Context Window Warning Hook (UserPromptSubmit)
#
# Checks agent's message count via Letta API as a proxy for context usage.
# Outputs warning to stderr which gets injected as <system-reminder> to the agent.
# Exit 0 = non-blocking, message proceeds with warning appended.
#
# Environment variables:
#   LETTA_API_KEYS  - Comma-separated list of Letta API keys (tries each until one works)
#   LETTA_API_KEY   - Single API key (fallback if LETTA_API_KEYS not set)
#   CONTEXT_WARN_THRESHOLD  - Message count for warning (default: 85)
#   CONTEXT_CRIT_THRESHOLD  - Message count for critical warning (default: 110)
#
# Install: Add to ~/.letta/settings.json under hooks.UserPromptSubmit
# See settings-example.json for configuration.

# Ensure jq is available
if ! command -v jq &>/dev/null; then
    # Common manual install locations
    for p in "$HOME/.local/bin" "/usr/local/bin" "/opt/homebrew/bin"; do
        if [ -x "$p/jq" ]; then
            export PATH="$p:$PATH"
            break
        fi
    done
fi

if ! command -v jq &>/dev/null; then
    exit 0  # Can't check without jq — fail silently
fi

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

# Build list of API keys to try
KEYS=()
if [ -n "$LETTA_API_KEYS" ]; then
    IFS=',' read -ra KEYS <<< "$LETTA_API_KEYS"
elif [ -n "$LETTA_API_KEY" ]; then
    KEYS=("$LETTA_API_KEY")
else
    exit 0  # No keys configured
fi

# Try each key until one works
msg_count=""
for key in "${KEYS[@]}"; do
    key=$(echo "$key" | xargs)  # trim whitespace
    result=$(curl -s --max-time 5 \
        -H "Authorization: Bearer ${key}" \
        "https://api.letta.com/v1/agents/${agent_id}" 2>/dev/null \
        | jq '.message_ids | length' 2>/dev/null)

    if [ -n "$result" ] && [ "$result" != "null" ] && [ "$result" -gt 0 ] 2>/dev/null; then
        msg_count="$result"
        break
    fi
done

if [ -z "$msg_count" ]; then
    exit 0
fi

# Thresholds based on observed patterns:
# - Compaction triggers around 100-175 messages for 100k-200k context windows
# - 70% warning at ~85 messages
# - 85% critical at ~110 messages
WARN_THRESHOLD="${CONTEXT_WARN_THRESHOLD:-85}"
CRIT_THRESHOLD="${CONTEXT_CRIT_THRESHOLD:-110}"

if [ "$msg_count" -gt "$CRIT_THRESHOLD" ]; then
    cat <<WARN >&2
CONTEXT WINDOW CRITICAL (~85%+ capacity, ${msg_count} messages). COMPACTION IS IMMINENT.
ACTION REQUIRED NOW: Save your current working state to archival memory. Include: (1) current task and step, (2) next planned action, (3) key file paths, (4) any decisions or findings in progress. After compaction, search archival for "working state" to recover.
WARN
    exit 0
elif [ "$msg_count" -gt "$WARN_THRESHOLD" ]; then
    cat <<WARN >&2
CONTEXT WINDOW WARNING (~70%+ capacity, ${msg_count} messages). Compaction may occur soon. Consider saving your current working state to archival memory: current task, step, next action, key file paths.
WARN
    exit 0
fi

exit 0
