##
## @file main_window.py
## @brief Główne okno aplikacji Analizatora Danych Fizycznych.
##
import customtkinter as ctk
from typing import Any, Dict, List, Optional
import json
import pandas as pd
import threading
import queue
import time

from pathlib import Path

from gui.styles import COLORS, FONTS, DIMENSIONS, PREVIEW_MAX_COLUMNS, set_theme, THEMES
from tkinter import filedialog
from datetime import datetime
from gui.sidebar import Sidebar
from gui.quick_actions_panel import QuickActionsPanel
from gui.chat_panel import ChatPanel
from gui.visualization_panel import VisualizationPanel
from gui.generated_files_panel import GeneratedFilesPanel
from gui.stats_panel import StatsPanel
from gui.settings_panel import SettingsPanel
from config.settings import (
    WINDOW_TITLE, WINDOW_GEOMETRY, WINDOW_MINSIZE,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY,
    load_preferences, save_preferences
)
from config.api_config import get_models_for_provider
from config.model_params import (
    get_default_model_params,
    get_default_max_output_tokens,
    get_supported_param_keys,
    normalize_temperature,
    with_model_token_limits,
    is_thinking_model,
)
from core.session_manager import SessionManager
from core.strategies import DirectAnalysisStrategy, CodeGenerationStrategy, HybridStrategy
from data import DataLoader, DataValidator
from api.factory import APIFactory
from utils.cost_tracker import CostTracker
from utils.logger import logger

## @brief Co ile milisekund odświeżać tekst odpowiedzi na ekranie podczas napływu fragmentów (ms).
STREAMING_FLUSH_MS = 50

HYBRID_STATUS_PLAN = "Etap 1/3: Planowanie analizy…"
HYBRID_STATUS_COMPUTE = "Etap 2/3: Obliczenia lokalne…"
HYBRID_STATUS_INTERPRET = "Etap 3/3: Interpretacja wyników…"
HYBRID_STATUS_NON_STREAM = "Analiza hybrydowa w toku…"


class MainWindow(ctk.CTk):
    """!
    @brief Główne okno aplikacji.
    
    @details Klasa MainWindow dziedziczy po CTk i stanowi centralny punkt
             zarządzania aplikacją. Odpowiada za:
             - Inicjalizację i konfigurację interfejsu użytkownika
             - Zarządzanie połączeniami z API (OpenAI, Anthropic, DeepSeek)
             - Koordynację strategii analizy danych
             - Obsługę zdarzeń użytkownika
             - Zarządzanie sesjami i preferencjami
    
    @see ChatPanel, Sidebar, VisualizationPanel, StatsPanel
    """
    
    def __init__(self) -> None:
        """!
        @brief Konstruktor klasy MainWindow.
        
        @details Inicjalizuje główne okno aplikacji, wczytuje preferencje
                 użytkownika, konfiguruje komponenty i nawiązuje połączenie
                 z API.
        """
        super().__init__()

        self.preferences = load_preferences()
        self._apply_saved_model_preferences_to_state()
        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_GEOMETRY)
        self.minsize(WINDOW_MINSIZE[0], WINDOW_MINSIZE[1])
        _icon_path = Path(__file__).resolve().parent.parent.parent / "ikona.ico"
        if _icon_path.is_file():
            try:
                self.iconbitmap(str(_icon_path))
            except Exception:
                pass

        self.session = SessionManager()
        self.cost_tracker = CostTracker()
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()

        self.api_client = None
        self.current_temperature = float(self.model_params.get("temperature", 0.0))
        self.current_max_tokens = int(self.model_params.get("max_output_tokens", 1500))
        self.streaming_enabled = bool(self.model_params.get("streaming", True))
        self.current_strategy = "direct"
        
        self.strategies = {}
        
        self.loaded_data: Optional[pd.DataFrame] = None
        self.is_processing = False
        self._processing_lock = threading.Lock()
        self._streaming_chunk_queue = queue.Queue()
        self._streaming_flush_after_id = None
        self._streaming_hybrid_pending = False
        self._streaming_code_pending_result = None
        
        logger.info("Inicjalizacja MainWindow")
        
        self._initialize_api()
        self._apply_saved_theme()
    
    def _apply_saved_model_preferences_to_state(self) -> None:
        """!
        @brief Ustawia current_provider, current_model i model_params z preferencji.

        @details Waliduje, czy model istnieje u danego dostawcy, w razie potrzeby
                 podstawia pierwszy dostępny model z listy.
        """
        provider = self.preferences.get("last_provider", "openai")
        model = self.preferences.get("last_model", "gpt-4o-mini")
        models = get_models_for_provider(provider)
        if not models or model not in models:
            model = models[0] if models else "gpt-4o-mini"
        self.current_provider = provider
        self.current_model = model
        self.model_params = dict(self.preferences.get("model_params") or get_default_model_params())
        self.model_params["max_output_tokens"] = get_default_max_output_tokens(
            self.current_provider,
            self.current_model
        )
        self.model_params = with_model_token_limits(
            self.current_provider,
            self.current_model,
            self.model_params
        )
        self.include_chat_context_in_api = bool(
            self.preferences.get("include_chat_context_in_api", False)
        )

    def _apply_saved_theme(self) -> None:
        """!
        @brief Zastosuje zapisany motyw przy starcie aplikacji.
        
        @details Wczytuje nazwę motywu z preferencji użytkownika i stosuje
                 odpowiedni schemat kolorów. Następnie buduje interfejs
                 użytkownika i wyświetla wiadomość powitalną.
        """
        theme_name = self.preferences.get("theme", "dark")
        if theme_name in THEMES:
            set_theme(theme_name)
            ctk.set_appearance_mode(THEMES[theme_name]["ctk_mode"])
            logger.info(f"Zastosowano zapisany motyw: {theme_name}")
        
        self._setup_ui()
        
        self._sync_sidebar_to_model_state()
        
        self._show_welcome_message()
    
    def _sync_sidebar_to_model_state(self) -> None:
        """! @brief Synchronizuje panel boczny z bieżącym dostawcą, modelem i parametrami."""
        self.sidebar.current_provider = self.current_provider
        self.sidebar.current_model = self.current_model
        self.sidebar.current_strategy = self.current_strategy
        self.sidebar.model_params = dict(self.model_params)
        self.sidebar.provider_var.set(self.current_provider)
        provider_models = get_models_for_provider(self.current_provider)
        self.sidebar.model_menu.configure(values=provider_models)
        if self.current_model in provider_models:
            self.sidebar.model_var.set(self.current_model)
        elif provider_models:
            self.sidebar.model_var.set(provider_models[0])
            self.sidebar.current_model = provider_models[0]
            self.current_model = provider_models[0]
        self.sidebar.strategy_var.set(self.current_strategy)

    def _build_model_state(
        self,
        provider: str,
        model: str,
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """!
        @brief Przygotowuje zwalidowany stan modelu bez zapisywania go do obiektu.

        @param provider Identyfikator dostawcy API.
        @param model Żądany model (może zostać znormalizowany do pierwszego z listy).
        @param model_params Parametry przekazane z UI (łączone z domyślnymi).
        @return Słownik pól: provider, model, model_params, current_temperature, itd.
        """
        provider_models = get_models_for_provider(provider)
        normalized_model = model
        if provider_models and normalized_model not in provider_models:
            normalized_model = provider_models[0]

        merged_params = get_default_model_params()
        merged_params.update(dict(self.model_params or {}))
        merged_params.update(dict(model_params or {}))
        normalized_params = with_model_token_limits(
            provider,
            normalized_model,
            merged_params
        )
        current_temperature = float(normalized_params.get("temperature", 0.0))
        current_max_tokens = int(normalized_params.get("max_output_tokens", 1500))
        streaming_enabled = bool(normalized_params.get("streaming", True))

        return {
            "provider": provider,
            "model": normalized_model,
            "model_params": normalized_params,
            "current_temperature": current_temperature,
            "current_max_tokens": current_max_tokens,
            "streaming_enabled": streaming_enabled,
        }

    def _commit_model_state(self, state: Dict[str, Any]) -> None:
        """!
        @brief Stosuje wcześniej sprawdzony stan modelu do atrybutów okna.

        @param state Słownik zwrócony przez _build_model_state().
        """
        self.current_provider = state["provider"]
        self.current_model = state["model"]
        self.model_params = dict(state["model_params"])
        self.current_temperature = float(state["current_temperature"])
        self.current_max_tokens = int(state["current_max_tokens"])
        self.streaming_enabled = bool(state["streaming_enabled"])

    def _build_strategies_for_client(self, api_client: Any) -> Dict[str, Any]:
        """! @brief Tworzy komplet strategii analizy dla aktywnego klienta API."""
        return {
            'direct': DirectAnalysisStrategy(api_client),
            'code': CodeGenerationStrategy(api_client),
            'hybrid': HybridStrategy(api_client)
        }

    def _get_api_key_for_provider(self, provider: str) -> str:
        """! @brief Zwraca klucz API dla wskazanego dostawcy."""
        api_keys = {
            'openai': OPENAI_API_KEY,
            'anthropic': ANTHROPIC_API_KEY,
            'deepseek': DEEPSEEK_API_KEY
        }
        return api_keys.get(provider, "")

    def _save_current_model_preferences(self) -> None:
        """! @brief Zapisuje bieżący dostawcę, model i parametry do pliku preferencji."""
        self.preferences["last_provider"] = self.current_provider
        self.preferences["last_model"] = self.current_model
        self.preferences["temperature"] = self.current_temperature
        self.preferences["model_params"] = dict(self.model_params)
        save_preferences(self.preferences)
    
    def _initialize_api(self) -> None:
        """!
        @brief Inicjalizuje klienta API.
        
        @details Preferuje zapisanego w preferencjach dostawcę, jeśli ma klucz,
                 w przeciwnym razie wybiera pierwszego dostępnego.
        
        @note Kolejność: zapisany dostawca -> OpenAI -> Anthropic -> DeepSeek
        """
        preferred = self.current_provider
        if preferred and self._get_api_key_for_provider(preferred):
            try:
                self.api_client = APIFactory.create_client(
                    preferred,
                    self._get_api_key_for_provider(preferred)
                )
                self.strategies = self._build_strategies_for_client(self.api_client)
                logger.info(f"API zainicjalizowane (preferencje): {preferred}")
                return
            except Exception as e:
                logger.error(f"Błąd inicjalizacji {preferred}: {e}")
        for provider in ('openai', 'anthropic', 'deepseek'):
            key = self._get_api_key_for_provider(provider)
            if key:
                try:
                    next_state = self._build_model_state(
                        provider,
                        self.current_model,
                        self.model_params
                    )
                    next_client = APIFactory.create_client(provider, key)
                    self.api_client = next_client
                    self.strategies = self._build_strategies_for_client(next_client)
                    self._commit_model_state(next_state)
                    logger.info(f"API zainicjalizowane: {provider}")
                    return
                except Exception as e:
                    logger.error(f"Błąd inicjalizacji {provider}: {e}")
                    continue
        logger.error("Nie udało się zainicjalizować żadnego API")
    
    def _setup_ui(self) -> None:
        """! @brief Buduje layout głównego okna: sidebar, zakładki, panel szybkich akcji."""
        self._create_header()
        
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        self.sidebar = Sidebar(
            main_container,
            on_file_load=self.handle_file_load,
            on_model_change=self.handle_model_change,
            get_stats=self.get_stats,
            on_strategy_change=self.handle_strategy_change
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.tab_view = ctk.CTkTabview(
            main_container,
            fg_color=COLORS["surface"],
            segmented_button_fg_color=COLORS["background"],
            segmented_button_selected_color=COLORS["primary"],
            segmented_button_selected_hover_color=COLORS["primary_dark"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        self.tab_view.grid(row=0, column=1, sticky="nsew")
        
        self.tab_view._segmented_button.configure(
            font=("Arial", 16, "bold"),
            height=45
        )
        
        analysis_tab = self.tab_view.add("💬 Analiza")
        
        self.chat_panel = ChatPanel(
            analysis_tab,
            on_send_message=self.handle_send_message
        )
        self.chat_panel.pack(fill="both", expand=True)
        
        viz_tab = self.tab_view.add("📈 Wizualizacja")
        self.viz_panel = VisualizationPanel(viz_tab)
        self.viz_panel.pack(fill="both", expand=True)
        
        files_tab = self.tab_view.add("📁 Pliki")
        self.generated_files_panel = GeneratedFilesPanel(files_tab)
        self.generated_files_panel.pack(fill="both", expand=True)
        
        stats_tab = self.tab_view.add("📊 Statystyki")
        self.stats_panel = StatsPanel(stats_tab, self.get_detailed_stats)
        self.stats_panel.pack(fill="both", expand=True)
        
        settings_tab = self.tab_view.add("⚙️ Ustawienia")
        self.settings_panel = SettingsPanel(
            settings_tab,
            on_theme_change=self.handle_theme_change,
            on_reset_session=self.handle_reset_session,
            on_export_chat=self.handle_export_chat,
            include_chat_context_in_api=self.include_chat_context_in_api,
            on_include_chat_context_change=self.handle_include_chat_context_change,
        )
        self.settings_panel.pack(fill="both", expand=True)
        
        self.quick_actions_panel = QuickActionsPanel(
            main_container,
            on_quick_action=self.handle_quick_action
        )
        self.quick_actions_panel.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
    
    def _create_header(self) -> None:
        """! @brief Tworzy pasek nagłówka z tytułem i podtytułem."""
        header_frame = ctk.CTkFrame(
            self,
            fg_color=COLORS["primary"],
            corner_radius=0
        )
        header_frame.pack(fill="x", padx=0, pady=0)
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Analizator Danych Fizycznych",
            font=FONTS["title"],
            text_color=COLORS["text"]
        )
        title_label.pack(pady=20)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Wspomagany przez modele AI - OpenAI, Anthropic, DeepSeek",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        subtitle_label.pack(pady=(0, 20))
    
    def _show_welcome_message(self) -> None:
        """! @brief Wyświetla wiadomość powitalną w panelu czatu."""
        welcome = """Witaj w Analizatorze Danych Fizycznych!
Aby rozpocząć:
1. Załaduj dane z pliku (CSV, Excel, JSON)
2. Wybierz model AI i strategię analizy
3. Zadaj pytanie o dane lub użyj szybkich akcji"""
        
        self.chat_panel.add_system_message(welcome)
    
    def handle_file_load(self, filepath: str) -> None:
        """!
        @brief Obsługuje wczytanie pliku z danymi.
        
        @param filepath Ścieżka do pliku z danymi (CSV, Excel, JSON).
        
        @details Metoda wykonuje następujące kroki:
                 1. Wczytuje dane za pomocą DataLoader
                 2. Waliduje dane za pomocą DataValidator
                 3. Czyści dane z błędów
                 4. Aktualizuje interfejs użytkownika
                 5. Generuje wykres wizualizacji
        
        @exception Exception W przypadku błędu wyświetla komunikat w czacie.
        """
        try:
            logger.info(f"Wczytywanie pliku: {filepath}")
            
            loaded_df = self.data_loader.load(filepath)
            
            issues = self.data_validator.validate(loaded_df)
            
            self.loaded_data = self.data_validator.clean(loaded_df)
            if self.loaded_data is None:
                raise ValueError("Nie udało się przygotować danych po czyszczeniu")
            
            self.session.set_data(self.loaded_data)
            
            filename = filepath.split('/')[-1].split('\\')[-1]
            self.chat_panel.add_system_message(
                f"✅ Plik '{filename}' wczytany pomyślnie.\n"
                f"Wierszy: {len(self.loaded_data)}, Kolumn: {len(self.loaded_data.columns)}"
            )
            
            if issues:
                warnings = "\n".join(f"• {issue}" for issue in issues[:3])
                self.chat_panel.add_system_message(
                    f"⚠️ Znaleziono potencjalne problemy:\n{warnings}"
                )
            
            self.sidebar.update_data_preview(self.loaded_data)
            
            self.viz_panel.set_data(self.loaded_data)
            self.viz_panel.plot_dataframe(self.loaded_data, plot_type="line")
            
            self.chat_panel.input_field.focus_set()
            
        except Exception as e:
            logger.error(f"Błąd wczytywania pliku: {e}", exc_info=True)
            self.chat_panel.add_system_message(
                f"❌ Błąd wczytywania pliku: {str(e)}"
            )
    
    def handle_model_change(self, provider: str, model: str, model_params: Dict[str, Any]) -> None:
        """!
        @brief Obsługuje zmianę modelu AI.
        
        @param provider Nazwa dostawcy API (openai, anthropic, deepseek).
        @param model Nazwa modelu (np. gpt-4o, claude-3-sonnet).
        @param model_params Słownik parametrów modelu.
        
        @details Jeśli zmienił się dostawca, inicjalizuje nowego klienta API
                 i aktualizuje strategie analizy.
        """
        next_state = self._build_model_state(provider, model, model_params)
        should_refresh_client = (
            provider != self.current_provider
            or self.api_client is None
            or not self.strategies
        )
        next_client = self.api_client
        next_strategies = self.strategies

        if should_refresh_client:
            try:
                key = self._get_api_key_for_provider(provider)
                if not key:
                    self._sync_sidebar_to_model_state()
                    self.chat_panel.add_system_message(
                        f"❌ Brak klucza API dla {provider}"
                    )
                    return

                next_client = APIFactory.create_client(provider, key)
                if next_client is None:
                    raise ValueError(f"Nie udało się utworzyć klienta API dla: {provider}")
                next_strategies = self._build_strategies_for_client(next_client)
            except Exception as e:
                logger.error(f"Błąd zmiany dostawcy: {e}", exc_info=True)
                self._sync_sidebar_to_model_state()
                self.chat_panel.add_system_message(
                    f"❌ Błąd zmiany dostawcy: {str(e)}"
                )
                return

        self.api_client = next_client
        self.strategies = next_strategies
        self._commit_model_state(next_state)
        self._save_current_model_preferences()
        self._sync_sidebar_to_model_state()

        logger.info(
            f"Zmiana modelu: {self.current_provider}/{self.current_model}, temp={self.current_temperature}, "
            f"max_tokens={self.current_max_tokens}, streaming={self.streaming_enabled}"
        )
        self.chat_panel.add_system_message(
            f"Zmieniono model na: {self.current_model}\n"
            f"Parametry: temp={self.current_temperature}, "
            f"max_output_tokens={self.current_max_tokens}, "
            f"streaming={'ON' if self.streaming_enabled else 'OFF'}"
        )

    def _get_prior_chat_for_api(self) -> List[Dict[str, str]]:
        """!
        @brief Zwraca wcześniejsze tury czatu do dołączenia do żądania API (gdy włączone w ustawieniach).

        @return Pusta lista albo [user, assistant, ...] bez bieżącej wiadomości użytkownika.
        """
        if not self.include_chat_context_in_api:
            return []
        return self.session.get_prior_messages_for_api()

    def handle_include_chat_context_change(self, enabled: bool) -> None:
        """! @brief Zapisuje preferencję: czy do API dołączać pełną historię czatu."""
        self.include_chat_context_in_api = bool(enabled)
        self.preferences["include_chat_context_in_api"] = self.include_chat_context_in_api
        save_preferences(self.preferences)
        mode = (
            "pełna historia"
            if self.include_chat_context_in_api
            else "tylko bieżące pytanie"
        )
        logger.info(f"Kontekst czatu w API: {mode}")

    def _build_request_kwargs(self) -> Dict[str, Any]:
        """!
        @brief Buduje dodatkowe argumenty żądania API dla bieżącego dostawcy.

        @return Słownik dodatkowych parametrów przekazywanych do klienta (OpenAI / Anthropic).
        """
        if self.current_provider == "openai":
            return self._build_openai_request_kwargs()
        if self.current_provider == "anthropic":
            return self._build_anthropic_request_kwargs()
        return {}

    def _build_anthropic_request_kwargs(self) -> Dict[str, Any]:
        """!
        @brief Buduje parametry Messages API dla klienta Anthropic (top_p, top_k, thinking).

        @return Słownik parametrów charakterystycznych dla Anthropic.
        """
        if self.current_provider != "anthropic":
            return {}
        params = self.model_params
        supported = set(
            get_supported_param_keys(
                self.current_provider,
                self.current_model,
                params
            )
        )
        kwargs: Dict[str, Any] = {}

        if "top_p" in supported:
            top_p_val = float(params.get("top_p", 1.0))
            if top_p_val != 1.0:
                kwargs["top_p"] = top_p_val
        if "top_k" in supported:
            top_k = params.get("top_k")
            if top_k is not None:
                kwargs["top_k"] = int(top_k)
        budget = params.get("thinking_budget", 0)
        if isinstance(budget, (int, float)) and budget > 0:
            max_tok = getattr(self, "current_max_tokens", 4096) or 4096
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": min(int(budget), max(1, max_tok - 1))}
        return kwargs

    def _build_openai_request_kwargs(self) -> Dict[str, Any]:
        """!
        @brief Buduje parametry Responses API dla klienta OpenAI.

        @return Słownik dodatkowych opcji (np. top_p, truncation, reasoning, verbosity).
        """
        if self.current_provider != "openai":
            return {}

        params = self.model_params
        supported = set(get_supported_param_keys(self.current_provider, self.current_model, params))
        kwargs: Dict[str, Any] = {
        }

        if "top_p" in supported:
            kwargs["top_p"] = float(params.get("top_p", 1.0))
        if "truncation" in supported:
            kwargs["truncation"] = params.get("truncation", "disabled")
        if "service_tier" in supported:
            kwargs["service_tier"] = params.get("service_tier", "auto")
        if "store" in supported:
            kwargs["store"] = bool(params.get("store", False))

        top_logprobs = int(params.get("top_logprobs", 0))
        if "top_logprobs" in supported and top_logprobs > 0:
            kwargs["top_logprobs"] = top_logprobs

        effort = params.get("reasoning_effort", "none")
        if "reasoning_effort" in supported and effort and effort != "none":
            kwargs["reasoning"] = {"effort": effort}

        verbosity = params.get("output_verbosity", "medium")
        if "output_verbosity" in supported and verbosity in {"low", "medium", "high"}:
            kwargs["text"] = {"verbosity": verbosity}

        return kwargs

    def _get_temperature_for_request(self) -> float | None:
        """!
        @brief Zwraca temperature tylko jeśli jest wspierana dla bieżącego modelu.

        @details Przy włączonym Extended Thinking (thinking_budget > 0) API Anthropic
                 wymaga temperature=1, wtedy zwracana jest 1.0.

        @return Wartość temperature lub None, gdy parametr nie jest wspierany.
        """
        supported = set(
            get_supported_param_keys(
                self.current_provider,
                self.current_model,
                self.model_params
            )
        )
        if "temperature" not in supported:
            return None
        budget = self.model_params.get("thinking_budget", 0)
        if isinstance(budget, (int, float)) and budget > 0:
            return 1.0
        return normalize_temperature(self.current_provider, self.current_temperature)
    
    def handle_strategy_change(self, strategy: str) -> None:
        """! @brief Obsługuje zmianę strategii analizy (direct / code / hybrid)."""
        self.current_strategy = strategy
        
        strategy_names = {
            "direct": "Bezpośrednia analiza",
            "code": "Wykonywanie kodu Python",
            "hybrid": "Hybrydowa"
        }
        
        strategy_name = strategy_names.get(strategy, strategy)
        
        logger.info(f"Zmiana strategii: {strategy}")
        self.chat_panel.add_system_message(
            f"Zmieniono strategię na: {strategy_name}"
        )
    
    def handle_quick_action(self, action: str) -> None:
        """! @brief Obsługuje szybkie akcje z panelu (statystyki, analiza, wizualizacja itd.)."""
        if self.loaded_data is None:
            self.chat_panel.add_system_message("Najpierw załaduj dane!")
            return
        
        quick_action_display_names = {
            "statistics": "Statystyki",
            "analysis": "Analiza",
            "visualize": "Wizualizuj",
            "anomalies": "Anomalie",
            "correlation": "Korelacja",
        }
        display_name = quick_action_display_names.get(action, action)
        
        if action == "visualize":
            if self._can_use_code_interpreter():
                self.handle_send_message(
                    "Wygeneruj najlepszy wykres (matplotlib) "
                    "odpowiedni dla tych danych. Wybierz odpowiedni typ "
                    "wykresu i opisz co przedstawia.\n\n"
                    "Jeśli tworzysz więcej niż jeden wykres, każdy zapisz w osobnym pliku "
                    "obrazu (np. osobne plt.figure() i osobne plt.savefig(...) na każdy wykres). "
                    "Nie umieszczaj kilku niezależnych wykresów na jednej figurze z subplotami - "
                    "wtedy aplikacja otrzyma jeden plik zamiast osobnych wykresów.",
                    display_message=display_name,
                )
            else:
                self._generate_visualization()
            return
        
        actions_map = {
            "statistics": (
                "Oblicz podstawowe statystyki opisowe dla wszystkich kolumn numerycznych. "
                "Odpowiedź ma zawierać wyłącznie listę kolumn i ich wartości statystyk (np. mean, std, min, max, median). "
                "Bez wstępów, komentarzy ani analizy - tylko kolumny i liczby."
            ),
            "analysis": (
                "Przeanalizuj od razu dane numeryczne: zinterpretuj rozkłady, skale wartości, "
                "ewentualne odstępstwa i wnioski. Nie wypisuj na początku pełnej listy statystyk - "
                "podaj od razu analizę merytoryczną, odwołując się do konkretnych wartości (np. średnia, "
                "odchylenie, min/max) tylko wtedy, gdy są istotne dla wniosku."
            ),
            "anomalies": "Znajdź anomalie i wartości odstające w danych używając metody IQR.",
            "correlation": "Oblicz macierz korelacji między wszystkimi zmiennymi numerycznymi."
        }
        
        query = actions_map.get(action, "")
        if query:
            self.handle_send_message(query, display_message=display_name)
    
    def _can_use_code_interpreter(self) -> bool:
        """!
        @brief Sprawdza, czy wizualizacja z szybkiej akcji ma iść przez wykonywanie kodu (AI).

        @return True, jeśli strategia to 'code' (Wykonywanie kodu), działa z OpenAI i Anthropic.
        """
        return self.current_strategy == 'code'

    def _default_ai_status_text(self) -> str:
        """!
        @brief Domyślny status oczekiwania na odpowiedź modelu (bez etapów strategii).

        @return „Trwa proces myślenia…” dla modeli z rozumowaniem, inaczej „Generowanie odpowiedzi…”.
        """
        if is_thinking_model(self.current_provider, self.current_model):
            return "Trwa proces myślenia…"
        return "Generowanie odpowiedzi…"

    def _initial_processing_status_text(self) -> str:
        """!
        @brief Status przy starcie zapytania - zależy od strategii i trybu streamingu, nie od samego streamingu.
        """
        if self.current_strategy == "code":
            return self._code_execution_status_text()
        if self.current_strategy == "hybrid":
            if self.streaming_enabled:
                return HYBRID_STATUS_PLAN
            return HYBRID_STATUS_NON_STREAM
        return self._default_ai_status_text()

    def _streaming_placeholder_text(self) -> str:
        """! @brief Placeholder bąbelka streamingu (faza odpowiedzi tekstowej modelu)."""
        return self._default_ai_status_text()

    def _code_execution_status_text(self) -> str:
        """!
        @brief Tekst statusu w czacie podczas strategii wykonywania kodu.

        @return Etykieta narzędzia zależna od dostawcy (OpenAI / Anthropic).
        """
        tool_labels = {
            "openai": "Code Interpreter",
            "anthropic": "Code execution",
        }
        tool = tool_labels.get(self.current_provider, "Wykonywanie kodu")
        return f"⏳ {tool}: wykonywanie kodu…"
    
    def _generate_visualization(self) -> None:
        """! @brief Generuje wizualizację danych lokalnie (bez wywołania AI)."""
        try:
            numeric_cols = self.loaded_data.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) == 0:
                self.chat_panel.add_system_message(
                    "Brak kolumn numerycznych do wizualizacji."
                )
                return
            
            if len(numeric_cols) >= 2:
                plot_type = "line"
                description = "Wykres liniowy przedstawiający zależności między zmiennymi."
            else:
                plot_type = "hist"
                description = "Histogram rozkładu wartości."
            
            self.viz_panel.plot_dataframe(self.loaded_data, plot_type=plot_type)
            
            cols_info = ", ".join(list(numeric_cols)[:PREVIEW_MAX_COLUMNS])
            if len(numeric_cols) > PREVIEW_MAX_COLUMNS:
                cols_info += f" (+{len(numeric_cols) - PREVIEW_MAX_COLUMNS} więcej)"
            
            self.chat_panel.add_system_message(
                f"Wygenerowano wykres!\n\n"
                f"{description}\n"
                f"Kolumny: {cols_info}\n\n"
            )
            
            self.chat_panel.input_field.focus_set()
            
            logger.info(f"Wygenerowano wizualizację: {plot_type}")
            
        except Exception as e:
            logger.error(f"Błąd generowania wizualizacji: {e}", exc_info=True)
            self.chat_panel.add_system_message(
                f"❌ Błąd generowania wykresu: {str(e)}"
            )
    
    def handle_send_message(self, message: str, display_message: Optional[str] = None) -> None:
        """!
        @brief Obsługuje wysłanie wiadomości użytkownika (w tle w osobnym wątku).

        @param message Pełna treść wysyłana do modelu.
        @param display_message Opcjonalnie krótsza treść w czacie i w sesji (np. szybkie akcje).
        """
        if not message.strip():
            return
        
        if self.loaded_data is None:
            self.chat_panel.add_system_message("Najpierw załaduj dane!")
            return

        if not self.api_client:
            self.chat_panel.add_system_message(
                "❌ Brak zainicjalizowanego klienta API. Sprawdź konfigurację."
            )
            return

        if not self._try_start_processing():
            self.chat_panel.add_system_message(
                "Poczekaj na zakończenie poprzedniego zapytania..."
            )
            return
        
        text_to_show = display_message if display_message is not None else message
        self.chat_panel.add_user_message(text_to_show)
        self.session.add_message("user", text_to_show)
        
        self.chat_panel.set_input_enabled(False)
        self.chat_panel.show_typing_indicator(
            self.current_model,
            self._initial_processing_status_text(),
        )
        
        thread = threading.Thread(
            target=self._process_query,
            args=(message,),
            daemon=True
        )
        try:
            thread.start()
        except Exception:
            self._release_processing_lock()
            self._finish_processing()
            raise

    def _try_start_processing(self) -> bool:
        """!
        @brief Rezerwuje przetwarzanie nowego zapytania (flaga is_processing).

        @return False, jeśli poprzednie zapytanie jeszcze trwa.
        """
        with self._processing_lock:
            if self.is_processing:
                return False
            self.is_processing = True
            return True

    def _release_processing_lock(self) -> None:
        """! @brief Zwalnia blokadę aktywnego przetwarzania (is_processing = False)."""
        with self._processing_lock:
            self.is_processing = False

    def _is_limit_related_error(self, error: Exception) -> bool:
        """!
        @brief Sprawdza, czy błąd wskazuje na limit API (quota, rate limit, tokeny).

        @param error Wyjątek lub komunikat z API.
        @return True, jeśli tekst błędu pasuje do znanych markerów limitów.
        """
        error_text = str(error).lower()
        limit_markers = (
            "rate limit",
            "ratelimit",
            "too many requests",
            "insufficient_quota",
            "quota",
            "exceeded your current quota",
            "request too large",
            "context_length_exceeded",
            "token limit",
            "429",
        )
        return any(marker in error_text for marker in limit_markers)

    def _format_user_error(self, error: Exception, streaming: bool = False) -> str:
        """!
        @brief Formatuje komunikat błędu dla użytkownika (w tym przy limitach API).

        @param error Wyjątek źródłowy.
        @param streaming True, jeśli błąd dotyczy trybu streamingu.
        @return Łańcuch do wyświetlenia w czacie.
        """
        if self._is_limit_related_error(error):
            return (
                "Przerwano odpowiedź z powodu osiągnięcia limitu API "
                "(quota/rate limit). Spróbuj ponownie za chwilę lub zmniejsz liczbę "
                "tokenów."
            )
        prefix = "❌ Błąd streaming" if streaming else "❌ Błąd"
        return f"{prefix}: {str(error)}"
    
    def _process_query(self, message: str) -> None:
        """! @brief Przetwarza zapytanie w wątku roboczym (streaming lub pełna odpowiedź)."""
        try:
            if self.streaming_enabled:
                if self.current_strategy == 'hybrid':
                    self._process_query_streaming_hybrid(message)
                elif self.current_strategy == 'code':
                    self._process_query_streaming_code(message)
                else:
                    self._process_query_streaming(message)
                return
            
            strategy = self.strategies.get(
                self.current_strategy,
                self.strategies['direct']
            )
            
            logger.info(f"Używam strategii: {self.current_strategy}")

            t_start = time.perf_counter()
            prior = self._get_prior_chat_for_api()
            result = strategy.analyze(
                data=self.loaded_data,
                question=message,
                model=self.current_model,
                temperature=self._get_temperature_for_request(),
                max_tokens=self.current_max_tokens,
                prior_chat_messages=prior or None,
                **self._build_request_kwargs()
            )
            result["_elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
            
            self.after(0, self._handle_response, result)
            
        except Exception as e:
            logger.error(f"Błąd przetwarzania: {e}", exc_info=True)
            self.after(
                0,
                self.chat_panel.add_system_message,
                self._format_user_error(e, streaming=False)
            )
        finally:
            self._release_processing_lock()
            self.after(0, self._finish_processing)
    
    def _start_streaming_and_flush_timer(self, model: str | None) -> None:
        """! @brief Czyści kolejkę chunków, uruchamia bąbelek streamingu i timer odświeżania UI."""
        while True:
            try:
                self._streaming_chunk_queue.get_nowait()
            except queue.Empty:
                break

        self.chat_panel.start_streaming_message(
            model,
            placeholder_text=self._streaming_placeholder_text(),
        )
        self._streaming_flush_after_id = self.after(STREAMING_FLUSH_MS, self._flush_streaming_updates)
    
    def _flush_streaming_updates(self) -> None:
        """! @brief Odczytuje chunki z kolejki, aktualizuje UI i planuje kolejne odświeżenie lub koniec streamingu."""
        self._streaming_flush_after_id = None
        chunks = []
        seen_end = False
        reasoning_from_stream = ""
        try:
            while True:
                try:
                    item = self._streaming_chunk_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    seen_end = True
                    break
                if isinstance(item, tuple) and len(item) == 2 and item[0] is None:
                    seen_end = True
                    reasoning_from_stream = item[1] or ""
                    break
                chunks.append(item)
        except Exception:
            seen_end = True
        if chunks:
            self.chat_panel.append_streaming_content("".join(chunks))
        if seen_end:
            self.chat_panel.finish_streaming_message()
            hybrid_pending = self._streaming_hybrid_pending
            code_result = self._streaming_code_pending_result
            if hybrid_pending:
                self._streaming_hybrid_pending = False
                plan_t = getattr(self, "_streaming_hybrid_plan_text", "") or ""
                res_t = getattr(self, "_streaming_hybrid_results_text", "") or ""
                self.chat_panel.add_hybrid_detail_buttons_row(
                    plan_t,
                    res_t,
                    reasoning_from_stream or "",
                )
            elif code_result is not None:
                self._streaming_code_pending_result = None
                if reasoning_from_stream and reasoning_from_stream.strip():
                    self.chat_panel.add_show_reasoning_button(reasoning_from_stream)
                self._handle_response(
                    code_result,
                    show_response=False,
                    show_reasoning=False,
                )
            elif reasoning_from_stream and reasoning_from_stream.strip():
                self.chat_panel.add_show_reasoning_button(reasoning_from_stream)
            return
        self._streaming_flush_after_id = self.after(STREAMING_FLUSH_MS, self._flush_streaming_updates)
    
    def _process_query_streaming(self, message: str) -> None:
        """! @brief Przetwarza zapytanie strategii direct z użyciem streamingu odpowiedzi."""
        try:
            from core.prompt_builder import PromptBuilder, system_prompt_with_formatting
            if self.loaded_data is None:
                raise ValueError("Brak załadowanych danych do streamingu")
            
            prompt_builder = PromptBuilder()
            data_context = prompt_builder.format_data_summary(self.loaded_data, max_rows=None)
            
            base_system = "Jesteś ekspertem w analizie danych fizycznych. Odpowiadaj po polsku."
            prior = self._get_prior_chat_for_api()
            system_msg = {"role": "system", "content": system_prompt_with_formatting(base_system)}
            user_msg = {
                "role": "user",
                "content": f"Dane:\n{data_context}\n\nPytanie: {message}",
            }
            messages = [system_msg] + prior + [user_msg]
            
            self.after(0, self._start_streaming_and_flush_timer, self.current_model)
            
            full_response = ""
            reasoning_parts = []
            total_tokens = 0

            t_start = time.perf_counter()

            for chunk in self.api_client.query_stream(
                messages=messages,
                model=self.current_model,
                temperature=self._get_temperature_for_request(),
                max_tokens=self.current_max_tokens,
                **self._build_request_kwargs()
            ):
                if chunk.content:
                    full_response += chunk.content
                    total_tokens += 1
                    self._streaming_chunk_queue.put(chunk.content)
                if getattr(chunk, "reasoning_content", None):
                    reasoning_parts.append(chunk.reasoning_content)

            elapsed = time.perf_counter() - t_start

            self._streaming_chunk_queue.put((None, "".join(reasoning_parts)))
            
            self.session.add_message("assistant", full_response)
            
            self.cost_tracker.track_request(
                provider=self.current_provider,
                model=self.current_model,
                input_tokens=len(data_context.split()) * 2,
                output_tokens=total_tokens,
                messages=messages,
                response_text=full_response,
                elapsed_seconds=round(elapsed, 2),
            )
            
            self.after(0, self.stats_panel.refresh_stats)
            
            logger.info(f"Streaming zakończony, tokenów: ~{total_tokens}")
            
        except Exception as e:
            logger.error(f"Błąd streaming: {e}", exc_info=True)
            self._streaming_chunk_queue.put((None, ""))
            self.after(
                0,
                self.chat_panel.add_system_message,
                self._format_user_error(e, streaming=True)
            )
        finally:
            self._release_processing_lock()
            self.after(0, self._finish_processing)
    
    def _process_query_streaming_hybrid(self, message: str) -> None:
        """!
        @brief Przetwarza zapytanie hybrydowe ze streamingiem etapu interpretacji.

        @details Etapy: (1) planowanie AI, (2) obliczenia lokalne, (3) interpretacja ze streamingiem.
        """
        try:
            self._streaming_hybrid_pending = False

            strategy = self.strategies.get('hybrid')
            if strategy is None:
                raise ValueError("Strategia hybrydowa nie jest zainicjalizowana")

            api_kwargs = {
                'max_tokens': self.current_max_tokens,
                'prior_chat_messages': self._get_prior_chat_for_api() or None,
                **self._build_request_kwargs()
            }

            t_start = time.perf_counter()

            self.after(
                0, self.chat_panel.update_typing_text,
                HYBRID_STATUS_PLAN,
            )

            plan = strategy._plan_analysis(
                self.loaded_data, message,
                self.current_model,
                self._get_temperature_for_request(),
                **api_kwargs
            )

            if not plan['success']:
                self.after(
                    0, self.chat_panel.add_system_message,
                    f"❌ {plan.get('error', 'Błąd planowania')}"
                )
                return

            self.after(
                0, self.chat_panel.update_typing_text,
                HYBRID_STATUS_COMPUTE,
            )

            results = strategy._execute_computations(
                self.loaded_data, plan['operations']
            )

            self.after(
                0, self.chat_panel.update_typing_text,
                HYBRID_STATUS_INTERPRET,
            )

            interpretation_messages = strategy.build_interpretation_messages(
                message,
                results,
                prior_chat_messages=self._get_prior_chat_for_api() or None,
            )

            self.after(0, self._start_streaming_and_flush_timer, self.current_model)

            full_response = ""
            reasoning_parts = []
            total_stream_tokens = 0
            extra_stream_kwargs = {
                k: v for k, v in api_kwargs.items()
                if k not in ("max_tokens", "prior_chat_messages")
            }

            for chunk in self.api_client.query_stream(
                messages=interpretation_messages,
                model=self.current_model,
                temperature=self._get_temperature_for_request(),
                max_tokens=self.current_max_tokens,
                **extra_stream_kwargs
            ):
                if chunk.content:
                    full_response += chunk.content
                    total_stream_tokens += 1
                    self._streaming_chunk_queue.put(chunk.content)
                if getattr(chunk, "reasoning_content", None):
                    reasoning_parts.append(chunk.reasoning_content)

            plan_txt = (plan.get("plan_text") or "").strip()
            try:
                results_txt = json.dumps(
                    results, ensure_ascii=False, indent=2, default=str
                )
            except (TypeError, ValueError):
                results_txt = str(results)
            self._streaming_hybrid_plan_text = plan_txt
            self._streaming_hybrid_results_text = results_txt.strip()
            self._streaming_hybrid_pending = True

            self._streaming_chunk_queue.put((None, "".join(reasoning_parts)))

            elapsed = time.perf_counter() - t_start

            self.session.add_message("assistant", full_response)

            total_input = plan.get('tokens_input', 0) + len(
                str(results).split()
            ) * 2
            total_output = plan.get('tokens_output', 0) + total_stream_tokens

            cost = self.cost_tracker.track_request(
                provider=self.current_provider,
                model=self.current_model,
                input_tokens=total_input,
                output_tokens=total_output,
                messages=interpretation_messages,
                response_text=full_response,
                elapsed_seconds=round(elapsed, 2),
            )

            summary = self.cost_tracker.get_summary()
            self.after(
                0, self.sidebar.update_stats,
                summary['requests'], summary['total_tokens'],
                summary['total_cost_usd']
            )
            self.after(0, self.stats_panel.refresh_stats)

            logger.info(
                f"Hybrid streaming zakończony: koszt=${cost:.4f}, "
                f"tokeny≈{total_input + total_output}"
            )

        except Exception as e:
            logger.error(f"Błąd streaming hybrid: {e}", exc_info=True)
            self._streaming_hybrid_pending = False
            self._streaming_chunk_queue.put((None, ""))
            self.after(
                0, self.chat_panel.add_system_message,
                self._format_user_error(e, streaming=True)
            )
        finally:
            self._release_processing_lock()
            self.after(0, self._finish_processing)

    def _process_query_streaming_code(self, message: str) -> None:
        """!
        @brief Uruchamia strategię wykonywania kodu i strumieniuje końcową interpretację.

        @details Wykonanie kodu (Code Interpreter / Anthropic code execution) przebiega
                 przez strategię; interpretacja modelu jest strumieniowana przez query_stream,
                 a po zakończeniu dodawane są metadane (kod, pliki, koszty).
        """
        try:
            strategy = self.strategies.get('code')
            if strategy is None:
                raise ValueError(
                    "Strategia wykonywania kodu nie jest zainicjalizowana"
                )

            self.after(
                0, self.chat_panel.update_typing_text,
                self._code_execution_status_text()
            )

            streaming_started = threading.Event()

            def _start_code_streaming() -> None:
                try:
                    self._start_streaming_and_flush_timer(self.current_model)
                finally:
                    streaming_started.set()

            self.after(0, _start_code_streaming)
            streaming_started.wait()

            def _on_code_stream_chunk(content: str) -> None:
                self._streaming_chunk_queue.put(content)

            t_start = time.perf_counter()

            prior = self._get_prior_chat_for_api()
            result = strategy.analyze(
                data=self.loaded_data,
                question=message,
                model=self.current_model,
                temperature=self._get_temperature_for_request(),
                max_tokens=self.current_max_tokens,
                prior_chat_messages=prior or None,
                stream_callback=_on_code_stream_chunk,
                **self._build_request_kwargs()
            )

            result['_elapsed_seconds'] = round(time.perf_counter() - t_start, 2)

            if not result.get('success', False):
                self._streaming_chunk_queue.put((None, ""))
                self.after(0, self._handle_response, result)
                return

            response_text = result.get('response', 'Brak odpowiedzi')
            self._streaming_code_pending_result = result

            self._streaming_chunk_queue.put((
                None,
                (result.get("reasoning_content") or "").strip(),
            ))
            self.session.add_message("assistant", response_text)

        except Exception as e:
            logger.error(f"Błąd streaming code: {e}", exc_info=True)
            self._streaming_code_pending_result = None
            self.after(
                0, self.chat_panel.add_system_message,
                self._format_user_error(e, streaming=True)
            )
        finally:
            self._release_processing_lock()
            self.after(0, self._finish_processing)

    def _finish_processing(self) -> None:
        """! @brief Kończy przetwarzanie: ukrywa wskaźnik pisania i włącza pole wejścia."""
        self.chat_panel.hide_typing_indicator()
        self.chat_panel.set_input_enabled(True)
        self.chat_panel.input_field.focus_set()
    
    def _handle_response(
        self,
        result: dict,
        show_response: bool = True,
        show_reasoning: bool = True,
    ) -> None:
        """! @brief Obsługuje wynik strategii analyze() w głównym wątku UI (odpowiedź, koszty, pliki)."""
        if not result.get('success', False):
            error_msg = result.get('error', 'Nieznany błąd')
            self.chat_panel.add_system_message(f"❌ Błąd: {error_msg}")
            return
        
        response_text = result.get('response', 'Brak odpowiedzi')
        if show_response:
            self.chat_panel.add_assistant_message(response_text, self.current_model)
            self.session.add_message("assistant", response_text)

        if result.get("strategy") == "hybrid":
            plan_txt = (result.get("plan") or "").strip()
            num_raw = result.get("numerical_results")
            try:
                results_txt = (
                    json.dumps(num_raw, ensure_ascii=False, indent=2, default=str)
                    if num_raw is not None
                    else ""
                )
            except (TypeError, ValueError):
                results_txt = str(num_raw) if num_raw is not None else ""
            reasoning_txt = (result.get("reasoning_content") or "").strip()
            if plan_txt or results_txt.strip() or reasoning_txt:
                self.chat_panel.add_hybrid_detail_buttons_row(
                    plan_txt, results_txt.strip(), reasoning_txt
                )
        elif show_reasoning and result.get("reasoning_content"):
            self.chat_panel.add_show_reasoning_button(result["reasoning_content"])
        
        if result.get('strategy') == 'code_generation':
            code = result.get('generated_code') or ''
            out = result.get('execution_output') or None
            if code or out:
                self.chat_panel.add_show_code_button(code, out)
        
        generated_files = result.get('generated_files') or []
        if generated_files:
            self.generated_files_panel.set_generated_files(generated_files)
            self.chat_panel.add_system_message(
                f"Zapisano {len(generated_files)} plik(ów) wygenerowanych przez AI. "
                "Otwórz zakładkę „Pliki”, aby je wyświetlić."
            )
        generated_images = result.get('generated_images') or []
        if generated_images:
            try:
                self.viz_panel.set_ai_images(generated_images)
                last_img = generated_images[-1]
                self.viz_panel.display_image_bytes(
                    last_img['data'],
                    title=last_img.get('filename') or "Wykres wygenerowany przez AI",
                )
                n = len(generated_images)
                if n > 1:
                    self.chat_panel.add_system_message(
                        f"AI wygenerowało {n} wykres(ów). "
                        f"Wybierz typ wykresu AI w zakładce Wizualizacja, aby wyświetlić."
                    )
                else:
                    self.chat_panel.add_system_message(
                        "AI wygenerowało wykres - "
                        "zobacz zakładkę Wizualizacja."
                    )
            except Exception as img_err:
                logger.error(f"Błąd wyświetlania obrazu AI: {img_err}")
        
        if 'tokens_input' in result and 'tokens_output' in result:
            cost = self.cost_tracker.track_request(
                provider=self.current_provider,
                model=self.current_model,
                input_tokens=result['tokens_input'],
                output_tokens=result['tokens_output'],
                messages=result.get('request_messages'),
                response_text=result.get('response'),
                elapsed_seconds=result.get('_elapsed_seconds'),
            )
            
            summary = self.cost_tracker.get_summary()
            self.sidebar.update_stats(
                requests=summary['requests'],
                tokens=summary['total_tokens'],
                cost=summary['total_cost_usd']
            )
            
            logger.info(f"Koszt zapytania: ${cost:.4f}, Suma: ${summary['total_cost_usd']:.4f}")
            
            self.stats_panel.refresh_stats()
    
    def get_stats(self) -> dict:
        """! @brief Zwraca skrócone statystyki sesji (koszty, tokeny)."""
        return self.cost_tracker.get_summary()
    
    def get_detailed_stats(self) -> dict:
        """! @brief Zwraca szczegółowe statystyki sesji (dla panelu statystyk)."""
        return self.cost_tracker.get_detailed_stats()
    
    def handle_export_chat(self) -> None:
        """! @brief Eksportuje historię konwersacji do pliku tekstowego."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"analiza_czat_{timestamp}.txt"
        
        filepath = filedialog.asksaveasfilename(
            title="Eksportuj historię czatu",
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[
                ("Plik tekstowy", "*.txt"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            messages = self.session.conversation_history
            
            lines = []
            lines.append("=" * 60)
            lines.append("ANALIZATOR DANYCH FIZYCZNYCH - HISTORIA KONWERSACJI")
            lines.append(f"Data eksportu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Model: {self.current_provider}/{self.current_model}")
            lines.append(f"Strategia: {self.current_strategy}")
            lines.append("=" * 60)
            lines.append("")
            
            stats = self.cost_tracker.get_summary()
            lines.append(f"Statystyki sesji:")
            lines.append(f"  - Zapytania: {stats['requests']}")
            lines.append(f"  - Tokeny: {stats['total_tokens']:,}")
            lines.append(f"  - Koszt: ${stats['total_cost_usd']:.4f}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")
            
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                timestamp = msg.get('timestamp', '')
                
                if role == 'user':
                    role_label = "UŻYTKOWNIK"
                elif role == 'assistant':
                    role_label = "AI"
                else:
                    role_label = "SYSTEM"
                
                lines.append(f"[{timestamp}] {role_label}")
                lines.append("-" * 40)
                lines.append(content)
                lines.append("")
                lines.append("")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            self.chat_panel.add_system_message(
                f"Historia czatu wyeksportowana do:\n{filepath}"
            )
            self.tab_view.set("💬 Analiza")
            
            logger.info(f"Wyeksportowano historię czatu do: {filepath}")
            
        except Exception as e:
            logger.error(f"Błąd eksportu czatu: {e}", exc_info=True)
            self.chat_panel.add_system_message(
                f"❌ Błąd eksportu: {str(e)}"
            )
    
    def handle_theme_change(self, theme_name: str) -> None:
        """! @brief Obsługuje zmianę motywu kolorystycznego i przebudowę UI."""
        if set_theme(theme_name):
            ctk_mode = THEMES[theme_name]["ctk_mode"]
            ctk.set_appearance_mode(ctk_mode)
            
            self.preferences["theme"] = theme_name
            save_preferences(self.preferences)
            
            self._rebuild_ui_for_theme_change()
            
            logger.info(f"Zmieniono motyw na: {theme_name}")
            self.chat_panel.add_system_message(
                f"Motyw zmieniony na: {THEMES[theme_name]['name']}"
            )
        else:
            self.chat_panel.add_system_message(
                f"❌ Nie udało się zmienić motywu na: {theme_name}"
            )

    def _rebuild_ui_for_theme_change(self) -> None:
        """!
        @brief Przebudowuje interfejs po zmianie motywu i odtwarza stan aplikacji.

        @details Okno jest chwilowo ukrywane, aby uniknąć migotania widgetów podczas przebudowy.
        """
        previous_tab = None
        previous_chat_messages = list(self.session.conversation_history)
        previous_plot_data = None
        previous_plot_type = "line"

        if hasattr(self, "tab_view"):
            try:
                previous_tab = self.tab_view.get()
            except Exception:
                previous_tab = None

        if hasattr(self, "viz_panel"):
            previous_plot_data = self.viz_panel.current_data
            previous_plot_type = self.viz_panel.current_plot_type

        self.withdraw()
        self.update_idletasks()

        for child in self.winfo_children():
            child.destroy()
        self._setup_ui()

        self._sync_sidebar_to_model_state()

        if self.loaded_data is not None:
            self.sidebar.update_data_preview(self.loaded_data)
        if previous_plot_data is not None:
            self.viz_panel.plot_dataframe(previous_plot_data, previous_plot_type)

        if previous_chat_messages:
            for msg in previous_chat_messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    self.chat_panel.add_user_message(content)
                elif role == "assistant":
                    self.chat_panel.add_assistant_message(content)
                elif role == "system":
                    self.chat_panel.add_system_message(content)
        else:
            self._show_welcome_message()

        try:
            self.stats_panel.refresh_stats()
        except Exception:
            pass

        if previous_tab:
            try:
                self.tab_view.set(previous_tab)
            except Exception:
                pass

        try:
            self.update_idletasks()
            self.chat_panel._scroll_chat_to_end()
        except Exception:
            pass

        self.deiconify()
    
    def handle_reset_session(self, reset_type: str) -> None:
        """! @brief Obsługuje reset czatu lub pełnej sesji (dane, koszty, wykres)."""
        if reset_type == "chat":
            self.chat_panel.clear_messages()
            self.session.clear_history(keep_system_prompts=False)
            self.chat_panel.add_system_message(
                "🗑️ Historia czatu została wyczyszczona."
            )
            logger.info("Wyczyszczono historię czatu")
            
        elif reset_type == "all":
            self.chat_panel.clear_messages()
            self.session.clear_history(keep_system_prompts=False)
            self.cost_tracker.reset()
            self.loaded_data = None
            self.viz_panel.clear_plot()
            
            self.sidebar.reset_data_preview()
            
            self.sidebar.update_stats(0, 0, 0.0)
            
            self._show_welcome_message()
            self.chat_panel.add_system_message(
                "🆕 Rozpoczęto nową sesję. Wszystkie dane zostały zresetowane."
            )
            
            logger.info("Zresetowano całą sesję")
        
        self.tab_view.set("💬 Analiza")
