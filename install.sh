#!/bin/bash
# Letta Compaction Shield — Installer
#
# Sets up the three-layer compaction protection system:
#   1. Copies hook scripts
#   2. Configures ~/.letta/settings.json
#   3. Optionally applies compaction settings to all agents
#
# Usage:
#   ./install.sh                    # Interactive install
#   ./install.sh --hooks-only       # Just install hooks, skip agent settings
#   ./install.sh --skip-hooks       # Just apply agent settings, skip hooks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$HOME/.letta/hooks"
SETTINGS_FILE="$HOME/.letta/settings.json"
HOOKS_ONLY=false
SKIP_HOOKS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hooks-only) HOOKS_ONLY=true; shift ;;
        --skip-hooks) SKIP_HOOKS=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Letta Compaction Shield — Installer ==="
echo ""

# Check for jq
if ! command -v jq &>/dev/null; then
    echo "⚠ jq not found. The context-warning hook requires jq."
    echo "  Install with: apt install jq / brew install jq"
    echo "  Or download from: https://jqlang.github.io/jq/download/"
    echo ""
    read -p "Continue without jq? (hooks will fail silently) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if ! $SKIP_HOOKS; then
    echo "[1/3] Installing hook scripts..."
    
    mkdir -p "$HOOKS_DIR"
    cp "$SCRIPT_DIR/hooks/context-warning.sh" "$HOOKS_DIR/"
    cp "$SCRIPT_DIR/hooks/pre-compact-warning.sh" "$HOOKS_DIR/"
    chmod +x "$HOOKS_DIR/context-warning.sh" "$HOOKS_DIR/pre-compact-warning.sh"
    echo "  ✓ Hooks installed to $HOOKS_DIR"
    echo ""
    
    echo "[2/3] Configuring settings.json..."
    
    if [ ! -f "$SETTINGS_FILE" ]; then
        echo "  ⚠ No settings.json found at $SETTINGS_FILE"
        echo "  Creating minimal settings with hooks..."
        cat > "$SETTINGS_FILE" << SETTINGS
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$HOOKS_DIR/context-warning.sh",
            "timeout": 10000
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$HOOKS_DIR/pre-compact-warning.sh",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
SETTINGS
        echo "  ✓ Created $SETTINGS_FILE with hook configuration"
    else
        # Check if hooks already configured
        if jq -e '.hooks.UserPromptSubmit' "$SETTINGS_FILE" &>/dev/null; then
            echo "  ⚠ Hooks already configured in settings.json"
            echo "  Check that paths point to: $HOOKS_DIR/"
            echo "  See settings-example.json for reference."
        else
            echo "  ⚠ settings.json exists but has no hooks configured."
            echo "  Back up: cp $SETTINGS_FILE ${SETTINGS_FILE}.bak"
            echo "  Then merge hooks from settings-example.json manually."
            echo "  (Automated merge risks overwriting your other settings.)"
        fi
    fi
    echo ""
else
    echo "[1/3] Skipping hooks (--skip-hooks)"
    echo "[2/3] Skipping settings (--skip-hooks)"
    echo ""
fi

if ! $HOOKS_ONLY; then
    echo "[3/3] Applying compaction settings to agents..."
    echo ""
    
    if [ -z "${LETTA_API_KEY:-}" ] && [ -z "${LETTA_API_KEYS:-}" ]; then
        echo "  ⚠ No LETTA_API_KEY or LETTA_API_KEYS set."
        echo "  Set your API key and run apply-compaction-settings.sh manually:"
        echo ""
        echo "    export LETTA_API_KEY='your-key-here'"
        echo "    $SCRIPT_DIR/apply-compaction-settings.sh"
        echo ""
    else
        "$SCRIPT_DIR/apply-compaction-settings.sh"
    fi
else
    echo "[3/3] Skipping agent settings (--hooks-only)"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "What was installed:"
echo "  Layer 1: Custom compaction prompt (preserves working state during compaction)"
echo "  Layer 2: Context warning hook (warns agent as context fills up)"
echo "  Layer 3: Pre-compaction hook (last-chance alert before compaction fires)"
echo ""
echo "The hooks are global — every agent you talk to via CLI gets warnings."
echo "Compaction settings are per-agent — new agents need settings applied manually."
echo "Run apply-compaction-settings.sh periodically to catch new agents."
