#!/bin/bash
# Apply Compaction Protection Settings to Letta Agents
#
# Discovers all agents on your account(s) and applies:
# - Custom compaction prompt that preserves working state
# - Sonnet 4.5 as the summarizer model (cheaper than Opus, good quality)
# - Sliding window mode keeping 80% recent context
# - 5000 char clip for compaction summaries
#
# Environment variables:
#   LETTA_API_KEYS  - Comma-separated list of Letta API keys
#   LETTA_API_KEY   - Single API key (fallback)
#
# Usage:
#   ./apply-compaction-settings.sh              # Apply to all agents
#   ./apply-compaction-settings.sh --dry-run    # Preview without changing
#   ./apply-compaction-settings.sh --prompt custom-prompt.txt  # Use custom prompt file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
FORCE=false
PROMPT_FILE="$SCRIPT_DIR/compaction-prompt.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --force) FORCE=true; shift ;;
        --prompt) PROMPT_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check dependencies
if ! command -v jq &>/dev/null; then
    echo "Error: jq is required. Install with: apt install jq / brew install jq"
    exit 1
fi

if ! command -v curl &>/dev/null; then
    echo "Error: curl is required."
    exit 1
fi

# Load compaction prompt
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Compaction prompt file not found: $PROMPT_FILE"
    echo "Create one or use --prompt to specify a custom path."
    exit 1
fi

COMPACTION_PROMPT=$(cat "$PROMPT_FILE")

# Build list of API keys
KEYS=()
if [ -n "${LETTA_API_KEYS:-}" ]; then
    IFS=',' read -ra KEYS <<< "$LETTA_API_KEYS"
elif [ -n "${LETTA_API_KEY:-}" ]; then
    KEYS=("$LETTA_API_KEY")
else
    echo "Error: Set LETTA_API_KEY or LETTA_API_KEYS environment variable."
    exit 1
fi

echo "=== Letta Compaction Shield — Apply Settings ==="
echo ""
if $DRY_RUN; then
    echo "  *** DRY RUN — no changes will be made ***"
    echo ""
fi

total_agents=0
total_updated=0
total_skipped=0

for key in "${KEYS[@]}"; do
    key=$(echo "$key" | xargs)  # trim whitespace
    
    # Get all agents for this key
    agents_json=$(curl -s --max-time 15 \
        -H "Authorization: Bearer ${key}" \
        "https://api.letta.com/v1/agents/?limit=100" 2>/dev/null)
    
    if [ -z "$agents_json" ] || echo "$agents_json" | jq -e '.detail' &>/dev/null; then
        echo "Warning: Could not fetch agents for key ${key:0:20}..."
        continue
    fi
    
    agent_count=$(echo "$agents_json" | jq 'length')
    echo "Found ${agent_count} agents for key ${key:0:20}..."
    echo ""
    
    for i in $(seq 0 $((agent_count - 1))); do
        agent_id=$(echo "$agents_json" | jq -r ".[$i].id")
        agent_name=$(echo "$agents_json" | jq -r ".[$i].name")
        
        # Check current settings
        current=$(curl -s --max-time 10 \
            -H "Authorization: Bearer ${key}" \
            "https://api.letta.com/v1/agents/${agent_id}" 2>/dev/null \
            | jq '.compaction_settings')
        
        current_prompt=$(echo "$current" | jq -r '.prompt // "null"')
        
        total_agents=$((total_agents + 1))
        
        if [ "$current_prompt" != "null" ] && [ "$current_prompt" != "" ] && ! $FORCE; then
            echo "  [$agent_name] Already has custom compaction prompt — skipping (use --force to overwrite)"
            total_skipped=$((total_skipped + 1))
            continue
        fi
        
        if $DRY_RUN; then
            echo "  [$agent_name] Would apply compaction settings"
            total_updated=$((total_updated + 1))
            continue
        fi
        
        # Build payload
        payload=$(python3 -c "
import json, sys
prompt = open('$PROMPT_FILE').read()
print(json.dumps({
    'compaction_settings': {
        'prompt': prompt,
        'model': 'anthropic/claude-sonnet-4-5-20250929',
        'mode': 'sliding_window',
        'sliding_window_percentage': 0.2,
        'clip_chars': 5000
    }
}))
")
        
        result=$(curl -s --max-time 15 -X PATCH \
            -H "Authorization: Bearer ${key}" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "https://api.letta.com/v1/agents/${agent_id}" 2>/dev/null)
        
        if echo "$result" | jq -e '.compaction_settings.prompt' &>/dev/null; then
            echo "  [$agent_name] ✓ Settings applied"
            total_updated=$((total_updated + 1))
        else
            echo "  [$agent_name] ✗ Failed to apply"
        fi
    done
    echo ""
done

echo "=== Summary ==="
echo "  Total agents:  $total_agents"
echo "  Updated:       $total_updated"
echo "  Skipped:       $total_skipped"
if $DRY_RUN; then
    echo ""
    echo "  This was a dry run. Run without --dry-run to apply changes."
fi
