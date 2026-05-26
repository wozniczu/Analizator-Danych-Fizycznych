##
## @file settings.py
## @brief Zmienne konfiguracyjne aplikacji: ścieżki, klucze API (`.env` tylko w katalogu aplikacji), limity, GUI.

import os
from pathlib import Path
from dotenv import load_dotenv

_HERE = Path(__file__).resolve()
_APP_DIR = _HERE.parents[1]
_ENV_FILE = _APP_DIR / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
MAX_DAILY_COST = 5.0

MAX_TOKENS_DEFAULT = 1500
MAX_VISIBLE_MESSAGES = 120

WINDOW_TITLE = "ADF - Analizator Danych Fizycznych"
WINDOW_GEOMETRY = "1400x900"
WINDOW_MINSIZE = (1200, 700)
SIDEBAR_WIDTH = 320

PREFERENCES_FILE = BASE_DIR / "user_preferences.json"


def load_preferences() -> dict:
    """!
    @brief Wczytuje preferencje z user_preferences.json, zwraca domyślne przy braku pliku lub błędzie.

    @return Słownik theme, last_provider, last_model, temperature, model_params (opcjonalnie).
    """
    import json
    from config.model_params import get_default_model_params
    defaults = {
        "theme": "dark",
        "last_provider": "openai",
        "last_model": "gpt-4o-mini",
        "temperature": 0.0,
        "model_params": None,
        "include_chat_context_in_api": False,
    }
    
    if PREFERENCES_FILE.exists():
        try:
            with open(PREFERENCES_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                defaults.update(prefs)
        except Exception:
            pass
    
    if defaults.get("model_params"):
        base = get_default_model_params()
        base.update(defaults["model_params"])
        if str(base.get("service_tier", "")).lower() == "scale":
            base["service_tier"] = "auto"
        defaults["model_params"] = base
    else:
        defaults["model_params"] = get_default_model_params()
    
    return defaults

def save_preferences(prefs: dict) -> None:
    """!
    @brief Zapisuje słownik preferencji do user_preferences.json (JSON, indent=2).

    @param prefs Słownik do zapisania.
    """
    import json
    try:
        with open(PREFERENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=2, ensure_ascii=False)
    except Exception:
        pass