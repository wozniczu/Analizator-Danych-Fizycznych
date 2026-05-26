##
## @file deepseek_client.py
## @brief Klient DeepSeek API (kompatybilny z OpenAI SDK, custom base_url).

from typing import List, Dict, Generator
from openai import OpenAI
from api.base import APIClient, APIResponse, StreamChunk, APIError
from utils.logger import logger

class DeepSeekClient(APIClient):
    """!
    @brief Klient DeepSeek przez OpenAI SDK z base_url api.deepseek.com.
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str) -> None:
        """!
        @brief Inicjalizuje klienta OpenAI z base_url DeepSeek, waliduje klucz (sk-).

        @param api_key Klucz API.
        """
        super().__init__(api_key)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.DEEPSEEK_BASE_URL
        )
        logger.info("Zainicjalizowano DeepSeek client")
    
    def _validate_api_key(self) -> None:
        """!
        @brief Sprawdza prefix sk-, rzuca ValueError przy błędnym formacie.
        """
        if not self.api_key.startswith('sk-'):
            raise ValueError("Nieprawidłowy format klucza DeepSeek")
    
    def query(
        self,
        messages: List[Dict[str, str]],
        model: str = 'deepseek-chat',
        temperature: float = 0.0,
        max_tokens: int = 1500,
        **kwargs
    ) -> APIResponse:
        """!
        @brief Wysyła zapytanie do chat.completions, obsługuje reasoning_content (deepseek-reasoner).

        @param messages Lista wiadomości [{role, content}].
        @param model Nazwa modelu (np. deepseek-chat, deepseek-reasoner).
        @param temperature Losowość.
        @param max_tokens Maks. tokenów.
        @return APIResponse z text i metadata.reasoning_content.
        @exception APIError Błąd API.
        """
        try:
            logger.debug(f"DeepSeek query: model={model}, temp={temperature}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or ""
            metadata: Dict[str, object] = {"id": response.id}
            if reasoning:
                metadata["reasoning_content"] = reasoning
            return APIResponse(
                text=msg.content or "",
                model=response.model,
                tokens_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens,
                finish_reason=response.choices[0].finish_reason,
                metadata=metadata,
            )
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}", exc_info=True)
            raise APIError(f"DeepSeek API error: {str(e)}") from e
    
    def query_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = 'deepseek-chat',
        temperature: float = 0.0,
        max_tokens: int = 1500,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """!
        @brief Zwraca generator chunków (content, reasoning_content) ze streamu chat.completions.

        @param messages Lista wiadomości.
        @param model Nazwa modelu.
        @param temperature Losowość.
        @return Generator StreamChunk.
        @exception APIError Błąd streamingu.
        """
        try:
            logger.debug(f"DeepSeek stream: model={model}")
            
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                reasoning = getattr(delta, "reasoning_content", None)
                if content is not None:
                    yield StreamChunk(
                        content=content,
                        finish_reason=chunk.choices[0].finish_reason,
                        reasoning_content=None,
                    )
                if reasoning is not None:
                    yield StreamChunk(
                        content=None,
                        finish_reason=None,
                        reasoning_content=reasoning,
                    )
            
        except Exception as e:
            logger.error(f"DeepSeek streaming error: {e}", exc_info=True)
            raise APIError(f"DeepSeek streaming error: {str(e)}") from e
    
    def get_available_models(self) -> List[str]:
        """!
        @brief Zwraca listę obsługiwanych modeli (deepseek-chat, deepseek-reasoner).

        @return Lista nazw modeli.
        """
        return ['deepseek-chat', 'deepseek-reasoner']