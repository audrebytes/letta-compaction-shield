#!/bin/bash
# Pre-Compaction Warning Hook (PreCompact)
#
# Fires immediately before context compaction occurs.
# Cannot block compaction, but outputs a warning to stderr.
# This is the last chance for the agent to see a warning before context is rewritten.
#
# Note: PreCompact cannot block (exit 2 won't prevent compaction).
# But stderr output is injected as <system-reminder> to the agent.
#
# Install: Add to ~/.letta/settings.json under hooks.PreCompact
# See settings-example.json for configuration.

input=$(cat)

cat <<WARN >&2
COMPACTION IS HAPPENING NOW. If you see this message and have not yet saved your working state, do so IMMEDIATELY in your next turn: save current task, step, next action, and key file paths to archival memory.
WARN

exit 0
