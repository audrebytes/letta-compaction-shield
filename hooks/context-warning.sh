#!/bin/bash
# Context Window Warning Hook v4 (UserPromptSubmit)
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
# Debug logging: set CRX_DEBUG=1 to enable (logs to /tmp/letta-hook-debug.log)

export PATH="/home/a/.local/bin:$PATH"

# --- Debug logging ---
CRX_DEBUG="${CRX_DEBUG:-1}"
LOG_FILE="/tmp/letta-hook-debug.log"

log() {
    if [ "$CRX_DEBUG" = "1" ]; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] [context-warning] $*" >> "$LOG_FILE"
    fi
}

log "=== Hook fired ==="

CACHE_DIR="/tmp/letta-context-cache"
mkdir -p "$CACHE_DIR"

input=$(cat)
event_type=$(echo "$input" | jq -r '.event_type // empty')

log "event_type=$event_type"

# Only process UserPromptSubmit
if [ "$event_type" != "UserPromptSubmit" ]; then
    log "Wrong event type, exiting"
    exit 0
fi

# Get agent_id from hook input
agent_id=$(echo "$input" | jq -r '.agent_id // empty')
if [ -z "$agent_id" ]; then
    log "No agent_id, exiting"
    exit 0
fi

log "agent_id=$agent_id"

# Both account API keys
KEYS=(
    "sk-let-OWQyNTI0YjEtZDc0NS00MzczLWIxMjctZjdlZjAzYTg1MzFmOjk3NDA2YWZhLWI0MzgtNGViMi1hYmE2LWQ5YjMyYzJkYWVkMg=="
    "sk-let-MGY0YTNkODctOTMyMi00MTIzLTkzNjktYWU4MWMxMDYxZGM0OmIyMWRhNDY1LTlhOGQtNDA5ZC05YjkyLTY0ZTU3OThlY2FiNQ=="
)

# Try each key until one works — get full agent data
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
        log "API key matched (key ending ...${key: -8})"
        break
    fi
done

if [ -z "$agent_json" ]; then
    log "No API key worked, exiting"
    exit 0
fi

# Extract key metrics
agent_name=$(echo "$agent_json" | jq -r '.name // "unknown"' 2>/dev/null)
context_window=$(echo "$agent_json" | jq '.llm_config.context_window // 0' 2>/dev/null)
msg_count=$(echo "$agent_json" | jq '.message_ids | length' 2>/dev/null)
block_chars=$(echo "$agent_json" | jq '[.blocks[]?.value // "" | length] | add // 0' 2>/dev/null)
system_chars=$(echo "$agent_json" | jq '.system | length' 2>/dev/null)

log "agent=$agent_name msgs=$msg_count window=$context_window blocks=${block_chars}chars system=${system_chars}chars"

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
        log "prev=$prev_count current=$msg_count drop=$drop"
        # If message count dropped by 30+, compaction happened
        if [ "$drop" -gt 30 ]; then
            compaction_detected=true
            timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
            log "COMPACTION DETECTED! Saving summary to archival..."
            
            # Get the first few messages — the compaction summary is typically
            # a system_message near the start of the remaining context
            summary_content=$(curl -s --max-time 5 \
                -H "Authorization: Bearer ${working_key}" \
                "https://api.letta.com/v1/agents/${agent_id}/messages/?limit=5&order=asc" 2>/dev/null \
                | jq -r '[.[] | select(.message_type == "system_message" or .message_type == "user_message") | {type: .message_type, content: .content}] | map("[\(.type)]\n\(.content)") | join("\n\n---\n\n")' 2>/dev/null)
            
            if [ -n "$summary_content" ] && [ "$summary_content" != "null" ]; then
                archival_text="## Compaction Summary (Auto-Saved)
**Agent:** ${agent_name} (${agent_id})
**Time:** ${timestamp}
**Context:** ${prev_count} messages compacted → ${msg_count} remaining (${context_window} token window)

### Summary Content:
${summary_content}"

                archival_payload=$(jq -n \
                    --arg text "$archival_text" \
                    '{
                        "text": $text,
                        "tags": ["compaction-recovery", "compaction-summary", "auto-save"]
                    }')

                save_result=$(curl -s --max-time 5 \
                    -X POST \
                    -H "Authorization: Bearer ${working_key}" \
                    -H "Content-Type: application/json" \
                    -d "$archival_payload" \
                    "https://api.letta.com/v1/agents/${agent_id}/archival-memory" 2>/dev/null)

                save_id=$(echo "$save_result" | jq -r 'if type == "array" then .[0].id else .id end // empty' 2>/dev/null)
                log "Archival save result: id=$save_id"
                # Clear autosave cache so next cycle can save again
                rm -f "${CACHE_DIR}/${agent_id}_autosaved"

                cat <<WARN >&2
COMPACTION DETECTED: Context was compacted (${prev_count} → ${msg_count} messages). Full compaction summary has been auto-saved to archival memory. Search for tags "compaction-summary" or "auto-save" to retrieve it.
WARN
            else
                log "Summary content was empty or null"
            fi
        fi
    fi
else
    log "No cache file yet (first run for this agent)"
fi

# Update cached message count
echo "$msg_count" > "$cache_file"
log "Cache updated: $msg_count"

# =====================================================================
# JOB 2: CONTEXT WARNING — dynamic thresholds based on real window size
# =====================================================================

# Skip warning if we just detected compaction (context just got freed)
if [ "$compaction_detected" = true ]; then
    log "Compaction just detected, skipping warning"
    exit 0
fi

# Safety: if we couldn't get context_window, fall back to message-count heuristic
if [ -z "$context_window" ] || [ "$context_window" -eq 0 ] 2>/dev/null; then
    log "No context_window, using fallback heuristic"
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

log "fixed=${fixed_tokens}tok avail=${available_for_messages}tok est_used=${estimated_msg_tokens}tok pct=${usage_pct}%"

# Dynamic thresholds
if [ "$usage_pct" -gt 85 ]; then
    log "CRITICAL WARNING fired"

    # Auto-save: grab last 10 messages and save to archival as pre-compaction snapshot
    # This compensates for PreCompact hook not being wired in Letta Code (as of v0.13)
    autosave_cache="${CACHE_DIR}/${agent_id}_autosaved"
    if [ ! -f "$autosave_cache" ]; then
        log "Auto-saving pre-compaction snapshot..."
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

        timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
        archival_text="## Auto-Save: Pre-Compaction Snapshot (Context Warning)
**Agent:** ${agent_name} (${agent_id})
**Time:** ${timestamp}
**Context:** ${msg_count} messages in ${context_window} token window (~${usage_pct}% full)
**Trigger:** Context warning hook at critical threshold (PreCompact hook workaround)

### Last 10 Messages (most recent first):
${last_messages}

### Recovery Instructions:
This was saved automatically when context hit ~${usage_pct}%. Your manual archival saves (tagged todo-recovery or compaction-recovery) will have more detailed working state if you made them. Check both."

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

        save_id=$(echo "$save_result" | jq -r 'if type == "array" then .[0].id else .id end // empty' 2>/dev/null)
        if [ -n "$save_id" ]; then
            log "AUTO-SAVE SUCCESS: id=$save_id"
            # Mark as saved so we don't spam archival every message at 85%+
            echo "$timestamp" > "$autosave_cache"
        else
            log "AUTO-SAVE FAILED: $(echo "$save_result" | head -c 200)"
        fi
    else
        log "Already auto-saved this cycle ($(cat "$autosave_cache"))"
    fi

    cat <<WARN >&2
CONTEXT WINDOW CRITICAL (~${usage_pct}% capacity, ${msg_count} messages, ${context_window} token window).
COMPACTION IS IMMINENT. A basic auto-save has been made, but YOU can do better.

**LOAD THE HOLD-MY-BEER SKILL NOW:** Use Skill("hold-my-beer") to save a structured recovery snapshot.
The skill knows what to save and how. Load it, follow the instructions, unload it. Do this BEFORE your next action.

If the skill is unavailable, manually save to archival with tags ["compaction-recovery", "hold-my-beer"]:
your current task, current step, next action, key file paths, and any decisions made this session.
WARN
    exit 0
elif [ "$usage_pct" -gt 70 ]; then
    log "WARNING fired"
    cat <<WARN >&2
CONTEXT WINDOW WARNING (~${usage_pct}% capacity, ${msg_count} messages, ${context_window} token window).
Compaction may occur soon. Consider saving your current working state to archival memory: current task, step, next action, key file paths.
WARN
    exit 0
fi

log "No warning needed (${usage_pct}%)"
log "=== Hook complete ==="
exit 0
