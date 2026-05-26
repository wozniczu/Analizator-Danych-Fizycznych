##
## @file logger.py
## @brief Singleton logger: plik dzienny w LOGS_DIR, stdout, nazwa PhysicsAnalyzer.

from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime
from config.settings import LOGS_DIR


class AppLogger:
    """!
    @brief Singleton: jeden logger z plikiem dziennym i StreamHandler.
    """

    _instance: AppLogger | None = None

    def __new__(cls: type[AppLogger]) -> AppLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._initialized = True
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """!
        @brief Włącza zapis do pliku (LOGS_DIR) i na standardowe wyjście, poziom INFO.
        """
        log_filename = LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("PhysicsAnalyzer")

    def info(self, message: str) -> None:
        """! @brief Zwykła informacja (INFO)."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """! @brief Ostrzeżenie (WARNING)."""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """! @brief Błąd (ERROR). Gdy exc_info=True, do logu trafia też ślad stosu wywołań."""
        self.logger.error(message, exc_info=exc_info)

    def debug(self, message: str) -> None:
        """! @brief Szczegóły do diagnozy (DEBUG)."""
        self.logger.debug(message)


logger = AppLogger()