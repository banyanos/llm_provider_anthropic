# llm_provider_anthropic

Anthropic Messages provider plugin for BanyanOS llm_engine.

Part of the BanyanOS LLM provider plugin architecture — see
`agent_engine/docs/LLM_PROVIDER_PLUGIN_DEVELOPMENT_PLAN.md` (D7, D8, D9).
Ported from the SilvaEngine `anthropic_agent_handler` package; the SilvaEngine package is
unchanged and still serves `ai_agent_core_engine`.

## Plugin registration

Discovered by `llm_engine` through the `banyan.llm_providers` entry point:

| Entry point | Class |
|---|---|
| `anthropic` | `llm_provider_anthropic.AnthropicProvider` |

The handler performs exactly one model call per request (single-turn) on
the pooled `httpx.Client` that `llm_engine` injects through
`ProviderContext`; credentials come from `ctx.credentials`.

## Tests

```bash
pytest
ruff check .
```
