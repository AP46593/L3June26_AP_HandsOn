"""
config.py - Shared configuration for all agents and orchestrators.
Contains model assignments, feature flags, and default parameters.
Safe to publish to GitHub (no secrets — those go in .env).

SETUP: Copy .env.example to .env and fill in your API keys.
"""

# =============================================================================
# OLLAMA
# =============================================================================
OLLAMA_BASE_URL = "http://localhost:11434"

# =============================================================================
# MODELS (format: "provider:model_name")
# Swap any model by changing the string. Examples:
#   "ollama:llama3"              - Local Llama 3 (8B)
#   "ollama:phi4"                - Local Phi-4 (14B)
#   "ollama:gpt-oss:120b-cloud"  - Cloud GPT-OSS (120B)
#   "openai:gpt-4o-mini"         - OpenAI (requires OPENAI_API_KEY in .env)
#   "anthropic:claude-3.5-sonnet" - Anthropic (requires ANTHROPIC_API_KEY in .env)
# =============================================================================
ORCHESTRATOR_MODEL = "ollama:gpt-oss:120b-cloud"
WEATHER_MODEL = "ollama:gpt-oss:120b-cloud"
CALC_MODEL = "ollama:gpt-oss:120b-cloud"
STOCK_MODEL = "ollama:gpt-oss:120b-cloud"
WEB_SEARCH_MODEL = "ollama:gpt-oss:120b-cloud"
TAVILY_SEARCH_MODEL = "ollama:gpt-oss:120b-cloud"

# =============================================================================
# LLM DEFAULTS
# =============================================================================
MAX_TOKENS = 500
TEMPERATURE = 0.7

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Observability — set to True to enable Opik tracing (requires OPIK_API_KEY in .env)
ENABLE_OPIK_TRACING = False

# Web Search — choose which search provider to use
# Options: "tavily" (requires TAVILY_API_KEY in .env) or "duckduckgo" (free, no key)
WEB_SEARCH_PROVIDER = "duckduckgo"

# Session Logging — set to True to write Message/agentLog files to User-Chat/
ENABLE_SESSION_LOGGING = True

# =============================================================================
# LOGGING
# =============================================================================
LOG_DIR = "User-Chat"
