##
## @file api_config.py
## @brief Lista dostawców, modele per dostawca, cennik (input/output za 1M tokenów) i nazwy do wyświetlania.

from typing import Dict, List

PROVIDERS = ["openai", "anthropic", "deepseek"]

MODELS: Dict[str, List[str]] = {
    "openai": ["gpt-5.2", "gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"]
}

## @brief Opłata za sesję kontenera OpenAI Code Interpreter (20 min) wg cennika built-in tools.
CODE_INTERPRETER_CONTAINER_SESSION_USD_OPENAI: Dict[str, float] = {
    "1g": 0.03,
    "4g": 0.12,
    "16g": 0.48,
    "64g": 1.92,
}

## @brief Opłata za kontener Anthropic Code execution.
ANTHROPIC_CODE_EXECUTION_CONTAINER_USD_PER_HOUR = 0.05
ANTHROPIC_CODE_EXECUTION_MIN_SECONDS = 300

PRICING: Dict[str, Dict[str, Dict[str, float]]] = {
    "openai": {
        "gpt-5.2": {"input": 1.75, "output": 14.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60}
    },
    "anthropic": {
        "claude-opus-4.5": {"input": 5.00, "output": 25.00},
        "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
        "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    },
    "deepseek": {
        "deepseek-chat": {"input": 0.028, "output": 0.28},
        "deepseek-reasoner": {"input": 0.55, "output": 2.19}
    }
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "gpt-5.2": "GPT-5.2 (OpenAI)",
    "gpt-4o": "GPT-4o (OpenAI)",
    "gpt-4o-mini": "GPT-4o mini (OpenAI)",
    "claude-opus-4.5": "Claude Opus 4.5 (Anthropic)",
    "claude-sonnet-4.5": "Claude Sonnet 4.5 (Anthropic)",
    "claude-haiku-4.5": "Claude Haiku 4.5 (Anthropic)",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner"
}

def get_models_for_provider(provider: str) -> List[str]:
    """!
    @brief Zwraca listę nazw modeli dla podanego dostawcy.

    @param provider Nazwa dostawcy (openai, anthropic, deepseek).
    @return Lista nazw modeli.
    """
    return MODELS.get(provider, [])


def get_model_price(provider: str, model: str) -> Dict[str, float]:
    """!
    @brief Zwraca cennik za 1M tokenów (input, output) dla modelu, (0,0) przy braku.

    @param provider Nazwa dostawcy.
    @param model Nazwa modelu.
    @return Słownik {"input": float, "output": float}.
    """
    return PRICING.get(provider, {}).get(model, {"input": 0, "output": 0})


def get_display_name(model: str) -> str:
    """!
    @brief Zwraca czytelną nazwę modelu do UI, przy braku mapowania zwraca model.

    @param model Identyfikator modelu.
    @return Łańcuch do wyświetlenia.
    """
    return MODEL_DISPLAY_NAMES.get(model, model)