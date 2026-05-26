##
## @file session_manager.py
## @brief Singleton zarządzający historią konwersacji, aktualnymi danymi i eksportem do JSON.

from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from utils.logger import logger


class SessionManager:
    """!
    @brief Singleton: historia wiadomości (role, content, timestamp), current_data, limity i eksport.
    """

    _instance: SessionManager | None = None

    def __new__(cls: type[SessionManager]) -> SessionManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._initialized = True
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_data = None
        self.session_start = datetime.now()
        self.max_history_length = 50
        
        logger.info("Zainicjalizowano SessionManager")
    
    def add_message(self, role: str, content: str) -> None:
        """!
        @brief Dodaje wiadomość do historii, przy przekroczeniu max_history_length obcina, zachowując system prompts.

        @param role Rola: 'user', 'assistant', 'system'.
        @param content Treść wiadomości.
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_history.append(message)

        if len(self.conversation_history) > self.max_history_length:
            system_msgs = [m for m in self.conversation_history if m['role'] == 'system']
            other_msgs = [m for m in self.conversation_history if m['role'] != 'system']
            
            keep_count = self.max_history_length - len(system_msgs)
            self.conversation_history = system_msgs + other_msgs[-keep_count:]
            
            logger.debug(f"Obcięto historię do {len(self.conversation_history)} wiadomości")
    
    def get_messages_for_api(
        self,
        include_system: bool = True,
        last_n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """!
        @brief Zwraca historię w formacie [{role, content}], opcjonalnie bez system lub tylko last_n.

        @param include_system Czy uwzględnić wiadomości systemowe.
        @param last_n Ograniczenie do ostatnich N wiadomości (system zawsze na początku).
        @return Lista słowników bez timestamp.
        """
        messages = self.conversation_history.copy()

        if not include_system:
            messages = [m for m in messages if m['role'] != 'system']

        if last_n:
            system_msgs = [m for m in messages if m['role'] == 'system']
            other_msgs = [m for m in messages if m['role'] != 'system']
            messages = system_msgs + other_msgs[-last_n:]

        return [{'role': m['role'], 'content': m['content']} for m in messages]

    def get_prior_messages_for_api(self) -> List[Dict[str, str]]:
        """!
        @brief Zwraca wcześniejsze tury (user/assistant, bez system) przed bieżącą wiadomością użytkownika.
        @details Ostatnia pozycja w historii to aktualna tura użytkownika - nie jest tu uwzględniana.
        @return Lista {role, content} lub [] przy pierwszej wiadomości.
        """
        msgs = self.get_messages_for_api(include_system=False)
        if len(msgs) <= 1:
            return []
        return msgs[:-1]

    def clear_history(self, keep_system_prompts: bool = True) -> None:
        """!
        @brief Czyści historię, opcjonalnie zostawia tylko wiadomości z role=='system'.

        @param keep_system_prompts True = zostaw system prompts.
        """
        if keep_system_prompts:
            self.conversation_history = [
                m for m in self.conversation_history if m['role'] == 'system'
            ]
        else:
            self.conversation_history = []
        
        logger.info("Wyczyszczono historię konwersacji")
    
    def set_data(self, data: Any) -> None:
        """!
        @brief Ustawia current_data (np. DataFrame), loguje shape.

        @param data Dane do analizy lub None.
        """
        self.current_data = data
        logger.info(f"Ustawiono dane: {data.shape if data is not None else None}")

    def get_data(self) -> Any:
        """!
        @brief Zwraca aktualnie ustawione dane (current_data).

        @return DataFrame lub None.
        """
        return self.current_data

    def get_session_summary(self) -> Dict[str, Any]:
        """!
        @brief Zwraca podsumowanie: duration_seconds, message_count, has_data, data_shape.

        @return Słownik z metrykami sesji.
        """
        duration = datetime.now() - self.session_start
        
        return {
            'duration_seconds': duration.total_seconds(),
            'message_count': len(self.conversation_history),
            'has_data': self.current_data is not None,
            'data_shape': self.current_data.shape if self.current_data is not None else None
        }
    
    def export_history(self, filepath: str) -> None:
        """!
        @brief Zapisuje session_start i conversation_history do pliku JSON.

        @param filepath Ścieżka do pliku (np. .json).
        """
        export_data = {
            'session_start': self.session_start.isoformat(),
            'conversation': self.conversation_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Wyeksportowano historię do {filepath}")