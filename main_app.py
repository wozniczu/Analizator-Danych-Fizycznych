##
## @file main_app.py
## @brief Główny punkt wejścia aplikacji Analizatora Danych Fizycznych.
##
## @author Bartosz Woźnica
## @date 2026
## @version 1.0.0
## @copyright Praca magisterska - Politechnika Warszawska
##
## @mainpage Analizator Danych Fizycznych
##
## @section intro_sec Wprowadzenie
##
## Analizator Danych Fizycznych to aplikacja umożliwiająca analizę
## danych eksperymentalnych z wykorzystaniem modeli sztucznej inteligencji.
##
## @section features_sec Główne funkcjonalności
##
## - @b Import @b danych: CSV, Excel, JSON
## - @b Strategie @b analizy:
##   - Bezpośrednia (Direct) - AI analizuje dane w prompcie
##   - Wykonywanie kodu (Code) - AI wykonuje kod Python
##   - Hybrydowa (Hybrid) - AI + lokalne obliczenia
## - @b Wizualizacja: wykresy liniowe, punktowe, histogramy, pudełkowe, niestandardowe
## - @b Statystyki: śledzenie użycia API i kosztów
##
## @section install_sec Instalacja
##
## @code
## pip install -r requirements.txt
## python main_app.py
## @endcode

import sys
import customtkinter as ctk
from pathlib import Path

project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from gui.main_window import MainWindow
from config.settings import OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY
from utils.logger import logger


def check_api_keys() -> bool:
    """!
    @brief Sprawdza czy klucze API są skonfigurowane.
    
    @return True jeśli przynajmniej jeden klucz API jest ustawiony.
    
    @details Sprawdza zmienne środowiskowe oraz - jeśli istnieje - plik `.env`
             w katalogu aplikacji (obok `main_app.py`). Klucze: OpenAI, Anthropic, DeepSeek.
    """
    keys = {
        'OpenAI': OPENAI_API_KEY,
        'Anthropic': ANTHROPIC_API_KEY,
        'DeepSeek': DEEPSEEK_API_KEY
    }
    
    configured = [name for name, key in keys.items() if key]
    
    if not configured:
        logger.warning("Brak skonfigurowanych kluczy API!")
        return False
    
    logger.info(f"Skonfigurowane klucze API: {', '.join(configured)}")
    return True

def main() -> None:
    """!
    @brief Inicjalizuje CustomTkinter, loguje stan kluczy API, uruchamia MainWindow.
    """
    logger.info("="*60)
    logger.info("Uruchamianie Analizatora Danych Fizycznych")
    logger.info("="*60)

    if not check_api_keys():
        logger.warning(
            "Brak skonfigurowanych kluczy API - aplikacja uruchamia się normalnie; "
            "funkcje AI będą dostępne po ustawieniu zmiennych w pliku .env (patrz .env.example)."
        )

    try:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        logger.info("Inicjalizacja GUI...")
        app = MainWindow()
        logger.info("Aplikacja gotowa do użycia")
        logger.info("="*60)
        app.mainloop()

    except Exception as e:
        logger.error(f"Krytyczny błąd aplikacji: {e}", exc_info=True)
        error_dialog = ctk.CTk()
        error_dialog.title("Błąd aplikacji")
        error_dialog.geometry("500x300")
        error_dialog.resizable(False, False)

        error_dialog.update_idletasks()
        x = (error_dialog.winfo_screenwidth() // 2) - 250
        y = (error_dialog.winfo_screenheight() // 2) - 150
        error_dialog.geometry(f"500x300+{x}+{y}")

        frame = ctk.CTkFrame(error_dialog)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        title = ctk.CTkLabel(
            frame,
            text="❌ Wystąpił krytyczny błąd",
            font=("Arial", 18, "bold"),
            text_color="red"
        )
        title.pack(pady=(20, 10))

        error_text = ctk.CTkTextbox(frame, height=150)
        error_text.pack(pady=10, fill="both", expand=True)
        error_text.insert("1.0", f"Błąd: {str(e)}\n\nSprawdź logi w katalogu logs/")
        error_text.configure(state="disabled")

        btn = ctk.CTkButton(
            frame,
            text="Zamknij",
            command=lambda: sys.exit(1),
            height=40
        )
        btn.pack(pady=10)

        error_dialog.mainloop()
    
    finally:
        logger.info("Aplikacja zakończona")

if __name__ == "__main__":
    main()
