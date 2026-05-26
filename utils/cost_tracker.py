##
## @file cost_tracker.py
## @brief Śledzenie kosztów API: tokeny, ceny per provider/model, historia zapytań z podglądem promptu/odpowiedzi (obciętym).

from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from config.api_config import get_model_price

## @brief Maksymalna długość podglądu promptu w historii kosztów (znaki).
MAX_PROMPT_DISPLAY_LEN = 80_000
## @brief Maksymalna długość podglądu odpowiedzi w historii kosztów (znaki).
MAX_RESPONSE_DISPLAY_LEN = 80_000


def _format_message_content(content: Any) -> str:
    """!
    @brief Konwertuje content (str lub listę bloków text/container_upload) na jeden string.

    @param content Treść wiadomości.
    @return Złączony tekst, container_upload → "[załączony plik]".
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "container_upload":
                    parts.append("[załączony plik]")
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def format_messages_for_preview(messages: List[Dict[str, Any]]) -> str:
    """!
    @brief Łączy wiadomości w jeden string z nagłówkami "--- ROLE ---" i treścią.

    @param messages Lista [{role, content}].
    @return Tekst do podglądu (przed obcięciem do MAX_*).
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = _format_message_content(msg.get("content", ""))
        lines.append(f"--- {role.upper()} ---\n{content}")
    return "\n\n".join(lines)


@dataclass
class ModelStats:
    """! @brief Liczniki i koszt dla jednego modelu: requests, tokeny, total_cost, first_used, last_used."""

    provider: str
    """! @brief Identyfikator dostawcy API."""
    model: str
    """! @brief Nazwa modelu."""
    requests: int = 0
    """! @brief Liczba zapytań do tego modelu."""
    tokens_input: int = 0
    """! @brief Suma tokenów wejściowych."""
    tokens_output: int = 0
    """! @brief Suma tokenów wyjściowych."""
    total_cost: float = 0.0
    """! @brief Łączny koszt (USD) dla tego modelu."""
    first_used: Optional[datetime] = None
    """! @brief Czas pierwszego użycia modelu w sesji."""
    last_used: Optional[datetime] = None
    """! @brief Czas ostatniego użycia modelu w sesji."""
    
    def add_request(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """! @brief Aktualizuje liczniki i daty first_used/last_used."""
        self.requests += 1
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
        self.total_cost += cost
        
        now = datetime.now()
        if self.first_used is None:
            self.first_used = now
        self.last_used = now
    
    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output
    
    @property
    def avg_tokens_per_request(self) -> float:
        if self.requests == 0:
            return 0
        return self.total_tokens / self.requests
    
    @property
    def avg_cost_per_request(self) -> float:
        if self.requests == 0:
            return 0
        return self.total_cost / self.requests
    
    def to_dict(self) -> Dict:
        """! @brief Zapisuje statystyki modelu w postaci słownika."""
        return {
            "provider": self.provider,
            "model": self.model,
            "requests": self.requests,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "avg_tokens_per_request": round(self.avg_tokens_per_request, 1),
            "avg_cost_per_request": round(self.avg_cost_per_request, 6),
            "first_used": self.first_used.isoformat() if self.first_used else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


@dataclass
class ProviderStats:
    """! @brief Agregacja statystyk per model, właściwości sumują po modelach."""

    provider: str
    """! @brief Identyfikator dostawcy API."""
    models: Dict[str, ModelStats] = field(default_factory=dict)
    """! @brief Mapa nazwa_modelu → ModelStats."""
    
    @property
    def requests(self) -> int:
        return sum(m.requests for m in self.models.values())
    
    @property
    def tokens_input(self) -> int:
        return sum(m.tokens_input for m in self.models.values())
    
    @property
    def tokens_output(self) -> int:
        return sum(m.tokens_output for m in self.models.values())
    
    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output
    
    @property
    def total_cost(self) -> float:
        return sum(m.total_cost for m in self.models.values())
    
    def get_or_create_model(self, model: str) -> ModelStats:
        """! @brief Zwraca ModelStats dla modelu, tworzy go przy pierwszym wywołaniu."""
        if model not in self.models:
            self.models[model] = ModelStats(provider=self.provider, model=model)
        return self.models[model]
    
    def to_dict(self) -> Dict:
        """! @brief Zwraca słownik z agregatami i models (name -> to_dict())."""
        return {
            "provider": self.provider,
            "requests": self.requests,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "models": {name: m.to_dict() for name, m in self.models.items()}
        }


class CostTracker:
    """! @brief Śledzi koszty sesji: agregaty, statystyki per provider/model, historia zapytań z podglądem."""

    def __init__(self) -> None:
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.total_cost = 0.0
        self.requests_count = 0
        self.session_start = datetime.now()
        self.providers: Dict[str, ProviderStats] = {}
        self.request_history: List[Dict] = []
        self.max_history = 100

    def track_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        response_text: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> float:
        """!
        @brief Rejestruje zapytanie, liczy koszt z get_model_price, aktualizuje providers i request_history (z obciętym podglądem).

        @param provider Nazwa dostawcy.
        @param model Nazwa modelu.
        @param input_tokens Tokeny wejściowe.
        @param output_tokens Tokeny wyjściowe.
        @param messages Opcjonalnie do podglądu promptu (obcinane do MAX_PROMPT_DISPLAY_LEN).
        @param response_text Opcjonalnie do podglądu odpowiedzi (obcinane do MAX_RESPONSE_DISPLAY_LEN).
        @param elapsed_seconds Czas generowania odpowiedzi w sekundach.
        @return Koszt zapytania w USD.
        """
        self.requests_count += 1
        self.total_tokens_input += input_tokens
        self.total_tokens_output += output_tokens

        prices = get_model_price(provider, model)
        cost = (
            (input_tokens / 1_000_000) * prices["input"] +
            (output_tokens / 1_000_000) * prices["output"]
        )
        
        self.total_cost += cost

        if provider not in self.providers:
            self.providers[provider] = ProviderStats(provider=provider)
        
        model_stats = self.providers[provider].get_or_create_model(model)
        model_stats.add_request(input_tokens, output_tokens, cost)

        prompt_display = None
        if messages:
            prompt_display = format_messages_for_preview(messages)
            if len(prompt_display) > MAX_PROMPT_DISPLAY_LEN:
                prompt_display = prompt_display[:MAX_PROMPT_DISPLAY_LEN] + "\n\n[... obcięto]"
        response_display = None
        if response_text is not None:
            response_display = response_text if len(response_text) <= MAX_RESPONSE_DISPLAY_LEN else (
                response_text[:MAX_RESPONSE_DISPLAY_LEN] + "\n\n[... obcięto]"
            )

        self.request_history.append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "tokens_input": input_tokens,
            "tokens_output": output_tokens,
            "cost": cost,
            "elapsed_seconds": elapsed_seconds,
            "prompt_display": prompt_display,
            "response_display": response_display,
        })

        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]
        
        return cost
    
    def get_summary(self) -> Dict:
        """! @brief Zwraca skrócone podsumowanie (requests, tokeny, total_cost_usd, costs_by_model) - zgodność wsteczna."""
        return {
            "requests": self.requests_count,
            "total_tokens": self.total_tokens_input + self.total_tokens_output,
            "tokens_input": self.total_tokens_input,
            "tokens_output": self.total_tokens_output,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_cost_per_request": round(
                self.total_cost / max(self.requests_count, 1), 4
            ),
            "costs_by_model": {
                f"{p}:{m}": stats.total_cost 
                for p, pstats in self.providers.items() 
                for m, stats in pstats.models.items()
            }
        }
    
    def get_detailed_stats(self) -> Dict:
        """! @brief Zwraca session (start, duration, tokeny, koszty, avg_elapsed) oraz providers i recent_requests (ostatnie 10)."""
        timed = [r["elapsed_seconds"] for r in self.request_history if r.get("elapsed_seconds") is not None]
        avg_elapsed = round(sum(timed) / len(timed), 2) if timed else None

        return {
            "session": {
                "start": self.session_start.isoformat(),
                "duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
                "requests": self.requests_count,
                "tokens_input": self.total_tokens_input,
                "tokens_output": self.total_tokens_output,
                "total_tokens": self.total_tokens_input + self.total_tokens_output,
                "total_cost_usd": round(self.total_cost, 6),
                "avg_cost_per_request": round(
                    self.total_cost / max(self.requests_count, 1), 6
                ),
                "avg_tokens_per_request": round(
                    (self.total_tokens_input + self.total_tokens_output) / max(self.requests_count, 1), 1
                ),
                "avg_elapsed_seconds": avg_elapsed,
            },
            "providers": {name: p.to_dict() for name, p in self.providers.items()},
            "recent_requests": self.request_history[-10:]
        }

    def get_provider_stats(self, provider: str) -> Optional[Dict]:
        """! @brief Zwraca to_dict() dla dostawcy lub None."""
        if provider in self.providers:
            return self.providers[provider].to_dict()
        return None
    
    def get_model_stats(self, provider: str, model: str) -> Optional[Dict]:
        """! @brief Zwraca to_dict() dla modelu lub None."""
        if provider in self.providers:
            if model in self.providers[provider].models:
                return self.providers[provider].models[model].to_dict()
        return None
    
    def check_budget(self, budget_limit: float) -> bool:
        """!
        @brief Sprawdza, czy łączny koszt sesji jest poniżej budget_limit.

        @param budget_limit Maks. dopuszczalny koszt (USD).
        @return True, jeśli total_cost < budget_limit.
        """
        return self.total_cost < budget_limit

    def reset(self) -> None:
        """! @brief Zeruje liczniki, session_start, providers i request_history."""
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.total_cost = 0.0
        self.requests_count = 0
        self.session_start = datetime.now()
        self.providers.clear()
        self.request_history.clear()
