##
## @file model_params.py
## @brief Domyślne parametry modeli, limity tokenów per provider/model oraz reguły kompatybilności (np. GPT-5 sampling tylko przy reasoning_effort==none).

from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "temperature": 0.0,
    "max_output_tokens": 1500,
    "streaming": True,
    "top_p": 1.0,
    "top_k": None,
    "top_logprobs": 0,
    "reasoning_effort": "medium",
    "output_verbosity": "medium",
    "truncation": "disabled",
    "service_tier": "auto",
    "store": False,
    "thinking_budget": 4096,
}

MIN_OUTPUT_TOKENS = 16
MAX_OUTPUT_TOKENS_FALLBACK = 8000


def get_temperature_max(provider: str) -> float:
    """!
    @brief Górny dopuszczalny zakres parametru temperature dla danego dostawcy API.

    @param provider Identyfikator dostawcy (openai, anthropic, deepseek).
    @return 2.0 dla OpenAI i DeepSeek, 1.0 dla Anthropic i nieznanych dostawców.
    """
    if provider in ("openai", "deepseek"):
        return 2.0
    return 1.0


def normalize_temperature(provider: str, value: Any) -> float:
    """!
    @brief Ogranicza temperature do [0, get_temperature_max(provider)].

    @param provider Identyfikator dostawcy.
    @param value Wartość do skorygowania.
    @return Wartość zmiennoprzecinkowa w dozwolonym zakresie.
    """
    hi = get_temperature_max(provider)
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.0
    return max(0.0, min(hi, x))


MAX_OUTPUT_TOKENS_BY_MODEL: Dict[str, Dict[str, int]] = {
    "openai": {
        "gpt-5.2": 128000,
        "gpt-4o": 16384,
        "gpt-4o-mini": 16384,
    },
    "anthropic": {
        "claude-opus-4.5": 64000,
        "claude-sonnet-4.5": 64000,
        "claude-haiku-4.5": 64000,
    },
    "deepseek": {
        "deepseek-chat": 8000,
        "deepseek-reasoner": 64000,
    },
}


SUPPORTED_PARAMS_BY_PROVIDER: Dict[str, List[str]] = {
    "openai": [
        "streaming",
        "max_output_tokens",
        "temperature",
        "top_p",
        "top_logprobs",
        "reasoning_effort",
        "output_verbosity",
        "truncation",
        "service_tier",
        "store",
    ],
    "anthropic": [
        "streaming",
        "max_output_tokens",
        "temperature",
        "top_p",
        "top_k",
        "thinking_budget",
    ],
    "deepseek": [
        "streaming",
        "max_output_tokens",
        "temperature",
    ],
}


def get_default_model_params() -> Dict[str, Any]:
    """!
    @brief Zwraca kopię słownika domyślnych parametrów (temperature, max_output_tokens, streaming, itd.).

    @return Słownik parametrów.
    """
    return dict(DEFAULT_MODEL_PARAMS)


def get_max_output_tokens_limit(provider: str, model: str) -> int:
    """!
    @brief Zwraca maksymalny dozwolony limit output tokenów dla danego provider/model.

    @param provider Nazwa dostawcy.
    @param model Nazwa modelu.
    @return Limit (liczba), przy braku wpisu zwraca MAX_OUTPUT_TOKENS_FALLBACK.
    """
    provider_limits = MAX_OUTPUT_TOKENS_BY_MODEL.get(provider, {})
    return int(provider_limits.get(model, MAX_OUTPUT_TOKENS_FALLBACK))


def get_default_max_output_tokens(provider: str, model: str) -> int:
    """!
    @brief Zwraca domyślny limit output tokenów (zawsze maksimum dla danego modelu).

    @param provider Nazwa dostawcy.
    @param model Nazwa modelu.
    @return Limit (liczba).
    """
    return get_max_output_tokens_limit(provider, model)


def normalize_max_output_tokens(provider: str, model: str, value: Any) -> int:
    """!
    @brief Ogranicza value do [MIN_OUTPUT_TOKENS, max_limit], przy błędnej wartości zwraca domyślny limit.

    @param provider Nazwa dostawcy.
    @param model Nazwa modelu.
    @param value Wartość do skorygowania (int lub konwertowalne).
    @return Znormalizowana liczba tokenów.
    """
    max_limit = get_max_output_tokens_limit(provider, model)
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = get_default_max_output_tokens(provider, model)
    return max(MIN_OUTPUT_TOKENS, min(max_limit, numeric))


def with_model_token_limits(provider: str, model: str, model_params: Dict[str, Any]) -> Dict[str, Any]:
    """!
    @brief Zwraca kopię model_params z max_output_tokens i temperature dopasowanymi do limitów modelu/dostawcy.

    @param provider Nazwa dostawcy.
    @param model Nazwa modelu.
    @param model_params Słownik parametrów (nie jest modyfikowany).
    @return Nowy słownik z zaktualizowanym max_output_tokens i temperature.
    """
    normalized = dict(model_params)
    normalized["max_output_tokens"] = normalize_max_output_tokens(
        provider,
        model,
        normalized.get("max_output_tokens")
    )
    normalized["temperature"] = normalize_temperature(provider, normalized.get("temperature"))
    return normalized


def _is_gpt5_family(model: str) -> bool:
    return model.startswith("gpt-5")


def is_thinking_model(provider: str, model: str) -> bool:
    """!
    @brief Sprawdza, czy model używa jawnego etapu rozumowania przed odpowiedzią.

    @param provider Identyfikator dostawcy.
    @param model Identyfikator modelu.
    @return True dla m.in. gpt-5.2 (OpenAI), wszystkich modeli Anthropic, deepseek-reasoner.

    @details Wykorzystywane w UI (np. tekst „Trwa proces myślenia…”).
    """
    if provider == "openai":
        return model == "gpt-5.2"
    if provider == "anthropic":
        return True
    if provider == "deepseek":
        return model == "deepseek-reasoner"
    return False


def get_reasoning_effort_options(provider: str, model: str) -> List[str]:
    """!
    @brief Zwraca listę dozwolonych wartości reasoning_effort (np. none, low, medium, high).

    @param provider Nazwa dostawcy.
    @param model Nazwa modelu.
    @return Lista stringów.
    """
    if provider == "openai" and _is_gpt5_family(model):
        return ["none", "low", "medium", "high", "xhigh"]
    return ["none", "low", "medium", "high"]


def get_supported_param_keys(
    provider: str,
    model: str,
    model_params: Dict[str, Any] | None = None
) -> List[str]:
    """!
    @brief Zwraca listę kluczy parametrów wspieranych dla provider/model. GPT-5: temperature/top_p/top_logprobs tylko przy reasoning_effort=="none", gpt-4o nie ma reasoning_effort.

    @param provider Nazwa dostawcy.
    @param model Nazwa modelu.
    @param model_params Opcjonalne bieżące parametry (do odczytu reasoning_effort).
    @return Lista nazw parametrów.
    """
    keys = list(SUPPORTED_PARAMS_BY_PROVIDER.get(provider, []))
    if provider != "openai":
        return keys

    reasoning_effort = "medium"
    if model_params:
        reasoning_effort = str(model_params.get("reasoning_effort", "medium"))

    gated_sampling = {"temperature", "top_p", "top_logprobs"}
    if _is_gpt5_family(model):
        keys = [k for k in keys if k not in gated_sampling]
        if model in {"gpt-5.2", "gpt-5.1"} and reasoning_effort == "none":
            keys.extend(["temperature", "top_p", "top_logprobs"])
    else:
        keys = [k for k in keys if k != "reasoning_effort"]

    return keys


def filter_model_params(
    provider: str,
    model: str,
    model_params: Dict[str, Any]
) -> Dict[str, Any]:
    """!
    @brief Zostawia w model_params tylko klucze zwrócone przez get_supported_param_keys.

    @param provider Nazwa dostawcy.
    @param model Nazwa modelu.
    @param model_params Pełny słownik parametrów.
    @return Słownik z filtrowanymi kluczami.
    """
    allowed = set(get_supported_param_keys(provider, model, model_params))
    return {k: v for k, v in model_params.items() if k in allowed}
