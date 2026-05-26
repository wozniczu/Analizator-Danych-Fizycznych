##
## @file base.py
## @brief Abstrakcyjna klasa bazowa dla klientów API.
##
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass


@dataclass
class APIResponse:
    """! @brief Wspólny format odpowiedzi API."""

    text: str
    """! @brief Treść odpowiedzi modelu."""
    model: str
    """! @brief Nazwa użytego modelu."""
    tokens_input: int
    """! @brief Liczba tokenów wejściowych."""
    tokens_output: int
    """! @brief Liczba tokenów wyjściowych."""
    finish_reason: str
    """! @brief Powód zakończenia generowania."""
    metadata: Dict[str, Any]
    """! @brief Dodatkowe metadane odpowiedzi."""


@dataclass
class StreamChunk:
    """! @brief Fragment odpowiedzi w trybie streaming."""

    content: Optional[str]
    """! @brief Fragment tekstu odpowiedzi."""
    finish_reason: Optional[str]
    """! @brief Powód zakończenia (jeśli ostatni chunk)."""
    reasoning_content: Optional[str] = None
    """! @brief Fragment rozumowania."""
    metadata: Optional[Dict[str, Any]] = None
    """! @brief Dodatkowe metadane streamu, np. finalna odpowiedź API."""


class APIClient(ABC):
    """!
    @brief Abstrakcyjna klasa bazowa dla klientów API.
    
    @details Definiuje wspólny interfejs dla wszystkich dostawców AI.
             Klasy pochodne muszą implementować metody:
             - _validate_api_key()
             - query()
             - query_stream()
             - get_available_models()
    
    @see OpenAIClient, AnthropicClient, DeepSeekClient
    """
    
    def __init__(self, api_key: str) -> None:
        """!
        @brief Konstruktor klienta API.

        @param api_key Klucz autoryzacyjny API.
        @exception APIError Gdy klucz API jest nieprawidłowy.
        """
        self.api_key = api_key
        self._validate_api_key()
    
    @abstractmethod
    def _validate_api_key(self) -> None:
        """!
        @brief Sprawdza format klucza API, rzuca wyjątek przy błędnym formacie.
        """
        pass
    
    @abstractmethod
    def query(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        **kwargs
    ) -> APIResponse:
        """!
        @brief Wysyła zapytanie do modelu i zwraca pełną odpowiedź.

        @param messages Lista wiadomości w formacie [{role, content}].
        @param model Nazwa modelu.
        @param temperature Losowość generowania (zakres zależy od dostawcy API).
        @param max_tokens Maksymalna liczba tokenów odpowiedzi.
        @param kwargs  Dodatkowe argumenty specyficzne dla dostawcy API.
        @return APIResponse z tekstem, metadanymi i użyciem tokenów.
        """
        pass
    
    @abstractmethod
    def query_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """!
        @brief Wysyła zapytanie i zwraca kolejne fragmenty odpowiedzi.

        @param messages Lista wiadomości [{role, content}].
        @param model Nazwa modelu.
        @param temperature Losowość (zakres zależy od dostawcy API).
        @param max_tokens Maksymalna liczba tokenów odpowiedzi.
        @param kwargs  Dodatkowe argumenty specyficzne dla dostawcy API.
        @return Generator obiektów StreamChunk.
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """!
        @brief Zwraca listę nazw modeli dostępnych dla danego dostawcy.

        @return Lista nazw modeli.
        """
        pass
    
    def format_messages(
        self,
        system_prompt: Optional[str],
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """!
        @brief Buduje listę wiadomości w formacie API (system, historia, user).

        @param system_prompt Opcjonalny prompt systemowy.
        @param user_message Treść wiadomości użytkownika.
        @param history Opcjonalna lista poprzednich wiadomości {role, content}.
        @return Lista słowników gotowa do przekazania do query/query_stream.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_message})
        
        return messages

class APIError(Exception):
    """! @brief Wyjątek sygnalizujący błąd wywołania API (sieć, autoryzacja, limit)."""
    pass