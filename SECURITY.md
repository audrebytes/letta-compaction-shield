# Security

## !! DO NOT HARDCODE API KEYS IN HOOK SCRIPTS !!

The hook scripts in `hooks/` are committed to a **public git repository**.

If you paste an API key directly into a hook script and commit it, that key
is now public. Rotating it is a pain. Don't do it.

**Keys go in environment variables. Full stop.**

```bash
# In ~/.bashrc, ~/.zshrc, or your shell profile:
export LETTA_API_KEY="your-key-here"

# For multiple Letta projects (keys are project-scoped):
export LETTA_API_KEYS="key-project-1,key-project-2"
```

Or in `~/.letta/settings.json` under the `env` block:

```json
{
  "env": {
    "LETTA_API_KEY": "your-key-here"
  }
}
```

`~/.letta/settings.json` is local and not committed to this repo.
The `hooks/` scripts are. Never the twain shall meet.

---

## Reporting a Security Issue

If you find a security issue in this project, please open a GitHub issue.
This is a small open-source tool — no formal disclosure process, just tell us.
