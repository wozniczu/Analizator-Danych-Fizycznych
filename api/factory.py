##
## @file factory.py
## @brief Fabryka tworząca klientów API (OpenAI, Anthropic, DeepSeek) według nazwy dostawcy.

from typing import Optional
from api.base import APIClient
from api.openai_client import OpenAIClient
from api.anthropic_client import AnthropicClient
from api.deepseek_client import DeepSeekClient
from utils.logger import logger


class APIFactory:
    """!
    @brief Tworzy instancję klienta API na podstawie nazwy dostawcy.
    """

    @staticmethod
    def create_client(provider: str, api_key: str) -> Optional[APIClient]:
        """!
        @brief Zwraca klienta API dla podanego dostawcy.

        @param provider Nazwa dostawcy: 'openai', 'anthropic', 'deepseek'.
        @param api_key Klucz API.
        @return Instancja APIClient lub None przy błędzie inicjalizacji.
        @exception ValueError Nieznany dostawca.
        """
        provider = provider.lower()
        
        try:
            if provider == "openai":
                logger.info("Tworzenie klienta OpenAI")
                return OpenAIClient(api_key)
            elif provider == "anthropic":
                logger.info("Tworzenie klienta Anthropic")
                return AnthropicClient(api_key)
            elif provider == "deepseek":
                logger.info("Tworzenie klienta DeepSeek")
                return DeepSeekClient(api_key)
            else:
                raise ValueError(f"Nieznany dostawca: {provider}")
                
        except Exception as e:
            logger.error(f"Błąd tworzenia klienta {provider}: {e}", exc_info=True)
            raise