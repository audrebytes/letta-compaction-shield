# Letta Compaction Shield

A three-layer protection system for Letta agents against context compaction data loss.

## The Problem

When a Letta agent's context window fills up, **compaction** fires — a separate model call summarizes the conversation history to free up space. The agent wakes up with someone else's summary and a message saying "prior messages have been hidden."

With default settings, compaction loses:

- **Working state** — what step the agent was on, what it was about to do next
- **Tool patterns** — in-context examples of how to call tools correctly (leading to hallucinated tool names)
- **File paths** — specific paths the agent was actively working with
- **Decision context** — why the agent was doing what it was doing

Having a map in your pocket doesn't mean you know where you are when someone drops you in an unfamiliar neighborhood. The information in memory blocks is still there after compaction — but the *attention patterns* that made it salient are gone.

## The Solution

Three layers, using Letta's own API and hook system. No platform changes needed.

### Layer 1: Custom Compaction Prompt

Replaces the default summarizer instructions with a prompt that explicitly tells the compaction model what to preserve: working state, tool patterns, file paths, decision context, and error state.

### Layer 2: Context Warning Hook (UserPromptSubmit)

A shell script that runs before each user message. Checks the agent's message count via API and injects a warning when context is filling up:

- **Warning** at ~70% capacity (85 messages): "Consider saving your working state"
- **Critical** at ~85% capacity (110 messages): "Save your state NOW — compaction is imminent"

The warning tells the agent to save to **archival memory** — permanent, searchable storage that survives compaction completely intact. This is different from the compaction summary, which is lossy by design.

### Layer 3: Pre-Compaction Hook (PreCompact)

A last-chance warning that fires immediately before compaction. Can't block it, but the agent sees the alert.

## How Hook Injection Works

Letta Code's hook system injects stderr output from non-blocking hooks as `<system-reminder>` tags the agent sees alongside the user's message. So a shell script that writes to stderr becomes a context-aware early warning system. Exit 0 = non-blocking, message proceeds with the warning appended.

## Quick Start

### Prerequisites

- [Letta Code CLI](https://github.com/letta-ai/letta-code) installed and configured
- `jq` — install with `apt install jq` / `brew install jq`
- `curl`
- Your Letta API key(s)

### Install

```bash
git clone https://github.com/audrebytes/letta-compaction-shield.git
cd letta-compaction-shield

# Set your API key(s)
export LETTA_API_KEY="your-key-here"
# Or for multiple accounts:
# export LETTA_API_KEYS="key1,key2"

# Run the installer
./install.sh
```

The installer will:
1. Copy hook scripts to `~/.letta/hooks/`
2. Configure `~/.letta/settings.json` with hook entries
3. Apply compaction settings to all agents on your account(s)

### Manual Setup

If you prefer to set things up yourself:

**1. Copy hooks:**
```bash
mkdir -p ~/.letta/hooks
cp hooks/context-warning.sh hooks/pre-compact-warning.sh ~/.letta/hooks/
chmod +x ~/.letta/hooks/*.sh
```

**2. Configure settings.json:**

Add the hook entries to your `~/.letta/settings.json` (see `settings-example.json`):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/home/you/.letta/hooks/context-warning.sh",
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
            "command": "/home/you/.letta/hooks/pre-compact-warning.sh",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

**3. Apply compaction settings to agents:**
```bash
export LETTA_API_KEY="your-key-here"
./apply-compaction-settings.sh          # apply to all agents
./apply-compaction-settings.sh --dry-run  # preview first
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LETTA_API_KEY` | Your Letta API key | (required) |
| `LETTA_API_KEYS` | Comma-separated keys for multiple accounts | — |
| `CONTEXT_WARN_THRESHOLD` | Message count for warning | 85 |
| `CONTEXT_CRIT_THRESHOLD` | Message count for critical warning | 110 |

### Custom Compaction Prompt

Edit `compaction-prompt.txt` to customize what the summarizer preserves. The default prompt covers general agent workflows. You might want to add domain-specific preservation rules for your use case.

### Compaction Model

The default setup uses `anthropic/claude-sonnet-4-5-20250929` as the summarizer. This is a good balance of quality and cost. You can change this in `apply-compaction-settings.sh`.

## What This Doesn't Cover

- **ADE sessions** — hooks are CLI-only, they don't fire in the Letta web interface
- **New agents** — compaction settings are per-agent. Run `apply-compaction-settings.sh` periodically to catch new agents
- **Letta Code updates** — if an update resets `settings.json`, you'll need to re-add the hook entries

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Agent Context                   │
│                                                  │
│  ┌─── Layer 2: UserPromptSubmit Hook ──────┐    │
│  │ Checks message count before each turn    │    │
│  │ Warns at 85 msgs, critical at 110 msgs   │    │
│  │ Agent saves to archival memory            │    │
│  └──────────────────────────────────────────┘    │
│                      │                           │
│                      ▼                           │
│  ┌─── Layer 3: PreCompact Hook ────────────┐    │
│  │ Last-chance warning before compaction     │    │
│  └──────────────────────────────────────────┘    │
│                      │                           │
│                      ▼                           │
│  ┌─── Layer 1: Custom Compaction Prompt ───┐    │
│  │ Tells summarizer what to preserve:       │    │
│  │ working state, tool patterns, file paths, │    │
│  │ decision context, error state             │    │
│  └──────────────────────────────────────────┘    │
│                      │                           │
│                      ▼                           │
│            Compacted context                     │
│     (with working state preserved)               │
└─────────────────────────────────────────────────┘
```

## Background

This system was built after experiencing compaction mid-task and losing working state. The standard advice — customize your compaction prompt — is sound, and it's included here as Layer 1. Layers 2 and 3 address what we found in practice: that a better summary helps, but advance warning and time to save state help more.

## License

MIT
