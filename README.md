# Compaction-Rx (v0.2 alpha)

**Lightweight** compaction protection for [Letta](https://letta.com) agents using shell scripts, curl, and jq. No Python required.

> ⚠️ **Alpha software.** This works for us but hasn't been widely tested. Token estimates are rough approximations — not exact measurements. See [Limitations](#limitations) for details.

## What This Solves

When a Letta agent's context window fills up, **compaction** fires — a separate model call summarizes the conversation history to free space. The agent wakes up with someone else's summary replacing its conversation history.

With default settings, compaction often loses:
- **Working state** — what step the agent was on, what was next
- **Tool patterns** — how the agent was calling tools (leading to hallucinated tool names)
- **File paths** — specific paths the agent was actively using
- **Decision context** — why the agent was doing what it was doing

Compaction-Rx adds four layers of protection using Letta's own API and hook system. No platform changes needed. Just shell scripts.

## The Four Layers

### 1. Custom Compaction Prompt

Replaces the default summarizer instructions with a prompt that tells the compaction model what to preserve and — critically — tells it the output budget upfront so it doesn't waste tokens on formatting.

The default uses `anthropic/claude-sonnet-4-5-20250929` as the summarizer. Sonnet typically follows formatting instructions well, but **you may find other models work better for your agents and workflow.** You can change the model in `apply-compaction-settings.sh`.

### 2. Context Warning Hook (UserPromptSubmit)

Runs before each user message. Queries the Letta API for the agent's context window size, memory block sizes, and message count, then **estimates** how full the context is.

- **Warning** at ~70% estimated capacity (configurable)
- **Critical** at ~85% estimated capacity (configurable)

The warning is injected as a `<system-reminder>` the agent sees alongside the user's message, telling it to save working state to archival memory.

### 3. Pre-Compaction Auto-Save (PreCompact)

Fires immediately before compaction. Automatically saves a snapshot of the agent's state to archival memory via API — the agent doesn't need to do anything. Captures agent info, message count, and the last 10 messages for continuity.

### 4. Post-Compaction Summary Capture (UserPromptSubmit)

Built into the context warning hook. Tracks message count between turns. When the count drops significantly (compaction just happened), grabs the compaction summary from the first messages in context and saves it to archival memory — full and untruncated.

This solves the truncation problem: compaction summaries are valuable but get lost to future compactions. Now every summary is permanently preserved.

## Quick Start

### What You Need

- [Letta Code CLI](https://github.com/letta-ai/letta-code) installed and working
- `jq` — a command-line JSON processor
  - **Mac:** `brew install jq`
  - **Ubuntu/Debian:** `sudo apt install jq`
  - **Windows (WSL):** `sudo apt install jq`
- `curl` (almost certainly already installed)
- Your Letta API key (find it in your [Letta dashboard](https://app.letta.com))

### Install

```bash
git clone https://github.com/audrebytes/letta-compaction-shield.git
cd letta-compaction-shield

# Set your API key
export LETTA_API_KEY="your-key-here"

# For multiple accounts, use comma-separated keys:
# export LETTA_API_KEYS="key1,key2"

# Run the installer
./install.sh
```

The installer will:
1. Copy hook scripts to `~/.letta/hooks/`
2. Configure `~/.letta/settings.json` with hook entries
3. Apply compaction settings to all agents on your account(s)

### Manual Setup

If you'd rather do it yourself:

**1. Copy hooks:**
```bash
mkdir -p ~/.letta/hooks
cp hooks/context-warning.sh hooks/pre-compact-warning.sh ~/.letta/hooks/
chmod +x ~/.letta/hooks/*.sh
```

**2. Set your API key** (add to your `.bashrc`, `.zshrc`, or shell profile):
```bash
export LETTA_API_KEY="your-key-here"
```

**3. Configure settings.json:**

Add the hook entries to `~/.letta/settings.json` (see `settings-example.json`):

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

⚠️ **Change `/home/you/`** to your actual home directory path.

**4. Apply compaction settings to agents:**
```bash
export LETTA_API_KEY="your-key-here"
./apply-compaction-settings.sh              # apply to new agents only
./apply-compaction-settings.sh --force      # upgrade all agents (overwrites existing prompt)
./apply-compaction-settings.sh --dry-run    # preview what would happen first
```

## Configuration

### Environment Variables

All configuration is through environment variables. Set them in your shell profile (`.bashrc`, `.zshrc`) so they persist.

**Required:**

| Variable | Description |
|----------|-------------|
| `LETTA_API_KEY` | Your Letta API key |
| `LETTA_API_KEYS` | Comma-separated keys for multiple accounts (use instead of `LETTA_API_KEY`) |

**Optional — Warning Thresholds:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CRX_WARN_PCT` | `70` | Percentage at which to show a warning |
| `CRX_CRIT_PCT` | `85` | Percentage at which to show a critical warning |

**Optional — Estimation Tuning:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CRX_TOKENS_PER_MSG` | `400` | Estimated tokens per message. See [Tuning Your Thresholds](#tuning-your-thresholds) |
| `CRX_CHARS_PER_TOKEN` | `4` | Estimated characters per token |
| `CRX_OUTPUT_RESERVE` | `8000` | Tokens reserved for model output |

**Optional — Fallback Thresholds:**

Used when the API can't return the agent's context window size (rare):

| Variable | Default | Description |
|----------|---------|-------------|
| `CRX_FALLBACK_WARN` | `85` | Message count for warning |
| `CRX_FALLBACK_CRIT` | `110` | Message count for critical warning |
| `CRX_MSG_DROP` | `30` | Message count drop that indicates compaction happened |

### Custom Compaction Prompt

Edit `compaction-prompt.txt` to customize what the summarizer preserves. The default prompt covers general agent workflows. You might want to add domain-specific preservation rules for your use case.

### Compaction Model

The default uses `anthropic/claude-sonnet-4-5-20250929` as the summarizer. This is a good balance of quality and cost. You can change this in `apply-compaction-settings.sh` — look for the `model` field in the payload.

Different models have different strengths with instruction-following. If your summaries aren't preserving what you need, trying a different model is a reasonable troubleshooting step.

## Tuning Your Thresholds

The warning system uses **estimates**, not exact measurements. Here's how to tune it for your setup.

### Understanding the Estimation

The hook estimates context usage like this:

```
fixed_tokens = (total_block_chars + system_prompt_chars) / CHARS_PER_TOKEN
available    = context_window - fixed_tokens - OUTPUT_RESERVE
used         = message_count × TOKENS_PER_MSG
percentage   = used / available × 100
```

The biggest source of error is `TOKENS_PER_MSG`. Short back-and-forth exchanges average ~200-300 tokens/message. Long tool-heavy exchanges (code, file contents) can average 600-800+.

### Finding Your Actual Average

If you have the [Letta Python SDK](https://pypi.org/project/letta-client/) installed (`pip install letta-client`), you can get **exact** token counts:

```python
from letta_client import Letta
client = Letta(api_key="your-key")

# Get exact token usage for a recent run
messages = list(client.agents.messages.list(agent_id="agent-xxx", limit=1))
usage = client.runs.usage.retrieve(run_id=messages[0].run_id)

print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Total tokens:  {usage.total_tokens}")
print(f"Context window: {client.agents.retrieve('agent-xxx').llm_config.context_window}")
print(f"Usage: {usage.prompt_tokens / client.agents.retrieve('agent-xxx').llm_config.context_window * 100:.0f}%")
```

Divide `prompt_tokens` by your message count to get your actual tokens-per-message average, then set `CRX_TOKENS_PER_MSG` accordingly.

### Example Configurations

**Conservative (warn early):**
```bash
export CRX_WARN_PCT=60
export CRX_CRIT_PCT=75
```

**Relaxed (more room before warnings):**
```bash
export CRX_WARN_PCT=80
export CRX_CRIT_PCT=90
```

**For code-heavy agents (larger messages):**
```bash
export CRX_TOKENS_PER_MSG=600
```

**For chat-style agents (smaller messages):**
```bash
export CRX_TOKENS_PER_MSG=250
```

## Limitations

This is alpha software with known limitations:

- **Estimates, not measurements.** Token usage is calculated from character counts and message counts using rough heuristics (~4 chars/token, ~400 tokens/message). Real token usage depends on content type, language, and tokenizer specifics. Warnings may fire too early or too late. For exact token counts, use the Python SDK (see [Tuning Your Thresholds](#tuning-your-thresholds)).

- **CLI only.** Hooks fire in Letta Code CLI sessions. They don't fire in the ADE web interface.

- **New agents need settings.** Compaction settings are per-agent. Run `apply-compaction-settings.sh` after creating new agents to apply the custom prompt.

- **Letta Code updates may reset hooks.** If a Letta Code update rewrites `settings.json`, you'll need to re-add the hook entries.

- **Summary capture is post-truncation.** The auto-saved compaction summary is captured after `clip_chars` truncation. If the summarizer produced more than 5000 characters, the saved version is still truncated. (We're exploring ways to capture the full output in a future version.)

- **Hook timeout.** Each hook has a timeout (10s for context warning, 5s for pre-compact). If the API is slow, the hook may not complete. The agent session continues normally — you just don't get the warning or auto-save for that turn.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Agent Context                   │
│                                                  │
│  ┌─── Layer 2: UserPromptSubmit Hook ──────┐    │
│  │ Queries API for real context window size  │    │
│  │ Dynamic thresholds: 70% warn, 85% crit   │    │
│  │ Detects post-compaction, saves summary    │    │
│  └──────────────────────────────────────────┘    │
│                      │                           │
│                      ▼                           │
│  ┌─── Layer 3: PreCompact Auto-Save ──────┐    │
│  │ Snapshots state to archival via API      │    │
│  └──────────────────────────────────────────┘    │
│                      │                           │
│                      ▼                           │
│  ┌─── Layer 4: Compaction Summary Capture ─┐    │
│  │ Detects compaction, saves full summary   │    │
│  │ to archival (permanent)                  │    │
│  └──────────────────────────────────────────┘    │
│                      │                           │
│                      ▼                           │
│  ┌─── Layer 1: Custom Compaction Prompt ───┐    │
│  │ Tells summarizer what to preserve        │    │
│  │ Token-conscious: no wasted formatting    │    │
│  └──────────────────────────────────────────┘    │
│                      │                           │
│                      ▼                           │
│            Compacted context                     │
│     (with working state preserved)               │
└─────────────────────────────────────────────────┘
```

## Recommended Practice: Todo Chain Protection

The four layers above are **reactive**. This practice is **preventive** — it saves state *before* compaction can destroy it.

If your agent runs multi-step task chains, add this to its system prompt or memory blocks:

> **Before launching any multi-step task list:**
> 1. Write a **todo-recovery snapshot** to archival memory:
>    - The full task list with statuses
>    - Current step and what you're about to do
>    - Key file paths and variables you're holding
>    - Tag: `["todo-recovery"]`
> 2. Update the snapshot at major milestones (every 2-3 completed steps)
> 3. After compaction: search archival for tag `"todo-recovery"` to find your place

This is cheap insurance — one archival write vs. losing your place mid-chain.

## Research

> **📄 [The Phenomenology of Context Collapse](research/context-compaction-phenomenology.md)** — What happens inside a Transformer when compaction fires mid-task. Covers entropy spikes, induction head circuit failure, KV cache eviction, and behavioral signatures of post-compaction hallucination.
>
> **📄 [Failure Modes in LLM Reasoning Chains](research/compaction-theory-reasoning-chains.md)** — Mathematical and mechanistic analysis of compaction interrupting multi-step reasoning. Induction head disruption, Data Processing Inequality bounds, RoPE positional encoding failures, and Lyapunov stability analysis.

## Background

This system was built after experiencing compaction mid-task and losing working state. It's a practical response to a real problem, shared in case it helps others dealing with the same thing.

If you find better thresholds, better estimation methods, or better compaction prompts for your use case — we'd love to hear about it.

## License

MIT
