##
## @file sidebar.py
## @brief Panel boczny z kontrolkami konfiguracji aplikacji.
##
import customtkinter as ctk
from tkinter import filedialog
from typing import Any, Callable, Dict, Optional
from pathlib import Path
import pandas as pd

from gui.styles import COLORS, FONTS, DIMENSIONS, PREVIEW_MAX_COLUMNS
from config.api_config import PROVIDERS, get_models_for_provider, get_display_name
from config.model_params import (
    filter_model_params,
    get_default_max_output_tokens,
    get_default_model_params,
    get_max_output_tokens_limit,
    get_temperature_max,
    MIN_OUTPUT_TOKENS,
    get_reasoning_effort_options,
    get_supported_param_keys,
    normalize_temperature,
    with_model_token_limits,
)
from utils.logger import logger


class Sidebar(ctk.CTkScrollableFrame):
    """!
    @brief Panel boczny z konfiguracją aplikacji.
    
    @details Klasa Sidebar zapewnia kontrolki dla:
             - Importu danych z plików (CSV, Excel, JSON)
             - Ładowania przykładowych danych testowych
             - Wyboru dostawcy i modelu AI
             - Wyboru strategii analizy (bezpośrednia, kod, hybrydowa)
             - Szybkich akcji (statystyki, wizualizacja, anomalie, korelacja)
    
    @see MainWindow
    """
    def __init__(
        self,
        parent: Any,
        on_file_load: Callable[[str], None],
        on_model_change: Callable[[str, str, Dict[str, Any]], None],
        get_stats: Callable[[], Dict[str, Any]],
        on_strategy_change: Optional[Callable[[str], None]] = None
    ) -> None:
        """!
        @brief Inicjalizuje sidebar: callbacks, stan provider/model/params, buduje sekcje importu, modelu, strategii, akcji.

        @param parent Widget rodzica.
        @param on_file_load Callback(ścieżka) po wczytaniu pliku.
        @param on_model_change Callback(provider, model, params) po zmianie modelu.
        @param get_stats Callback zwracający statystyki (do wyświetlenia).
        @param on_strategy_change Callback(nazwa strategii) przy zmianie strategii (opcjonalny).
        """
        super().__init__(
            parent,
            width=DIMENSIONS["sidebar_width"],
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        
        self.on_file_load = on_file_load
        self.on_model_change = on_model_change
        self.get_stats = get_stats
        self.on_strategy_change = on_strategy_change

        self.current_provider = "openai"
        self.current_model = "gpt-5.2"
        self.model_params: Dict[str, Any] = get_default_model_params()
        self.model_params["max_output_tokens"] = get_default_max_output_tokens(
            self.current_provider,
            self.current_model
        )
        self.current_strategy = "direct"
        self._strategy_tooltip: Optional[ctk.CTkToplevel] = None
        self._strategy_tooltip_after_id: Optional[str] = None
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """! @brief Tworzy sekcje: import, model, strategia."""
        self._create_import_section()
        self._create_model_section()
        self._create_strategy_section()
    
    def _create_import_section(self) -> None:
        """! @brief Buduje sekcję importu pliku, podglądu i danych testowych."""
        label = ctk.CTkLabel(
            self,
            text="Import Danych",
            font=FONTS["subheading"],
            anchor="w"
        )
        label.pack(pady=(20, 10), padx=20, fill="x")
        
        self.upload_btn = ctk.CTkButton(
            self,
            text="Wybierz plik",
            command=self._select_file,
            height=DIMENSIONS["button_height"],
            corner_radius=DIMENSIONS["corner_radius"],
            text_color_disabled=COLORS["text_disabled"]
        )
        self.upload_btn.pack(pady=10, padx=20, fill="x")
        
        preview_label = ctk.CTkLabel(
            self,
            text="Podgląd:",
            font=FONTS["small"],
            anchor="w"
        )
        preview_label.pack(pady=(10, 5), padx=20, anchor="w")
        
        self.data_preview = ctk.CTkTextbox(
            self,
            height=120,
            corner_radius=DIMENSIONS["corner_radius"],
            font=FONTS["code"]
        )
        self.data_preview.pack(pady=(0, 15), padx=20, fill="x")
        self.data_preview.insert("1.0", "Brak załadowanych danych")
        self.data_preview.configure(state="disabled")
        
        self.sample_data_btn = ctk.CTkButton(
            self,
            text="Dane testowe",
            command=self._toggle_sample_data_menu,
            height=35,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["surface"],
            border_width=2,
            border_color=COLORS["accent"],
            text_color_disabled=COLORS["text_disabled"]
        )
        self.sample_data_btn.pack(pady=(5, 10), padx=20, fill="x")
        
        self.sample_data_frame = ctk.CTkFrame(self, fg_color=COLORS["background"], corner_radius=8)
        self.sample_data_expanded = False
        
        self.sample_data_list = [
            ("Ruch jednostajnie przyspieszony", "constaccdata.xlsx"),
            ("Zależność okresu drgań od długości wahadła", "Simple pendulum data.xlsx"),
            ("Prawo Beer-Lamberta", "rsos211103_si_001.csv"),
            ("Zdarzenia z parą mionów", "dimuon-Jpsi.csv"),
            ("Pobór mocy urządzenia elektronicznego", "PCMS_meas.csv"),
        ]
        
        for text, filename in self.sample_data_list:
            btn = ctk.CTkButton(
                self.sample_data_frame,
                text=text,
                command=lambda f=filename: self._load_sample_data(f),
                height=28,
                corner_radius=5,
                fg_color="transparent",
                hover_color=COLORS["primary"],
                anchor="w",
                font=FONTS["small"],
                text_color_disabled=COLORS["text_disabled"]
            )
            btn.pack(pady=2, padx=10, fill="x")
    
    def _toggle_sample_data_menu(self) -> None:
        """! @brief Rozwija lub zwija listę wbudowanych plików przykładowych."""
        if self.sample_data_expanded:
            self.sample_data_frame.pack_forget()
            self.sample_data_btn.configure(text="Dane testowe")
            self.sample_data_expanded = False
        else:
            self.sample_data_frame.pack(after=self.sample_data_btn, pady=(0, 10), padx=20, fill="x")
            self.sample_data_btn.configure(text="Dane testowe ▲")
            self.sample_data_expanded = True
    
    def _load_sample_data(self, filename: str) -> None:
        """! @brief Wczytuje plik z katalogu sample_data przez callback on_file_load."""
        current_file = Path(__file__).parent.parent
        sample_path = current_file / "sample_data" / filename
        
        if sample_path.exists():
            self.on_file_load(str(sample_path))
            if self.sample_data_expanded:
                self._toggle_sample_data_menu()
            logger.info(f"Załadowano przykładowe dane: {filename}")
        else:
            logger.warning(f"Nie znaleziono pliku: {sample_path}")
    
    def _create_model_section(self) -> None:
        """! @brief Buduje sekcję wyboru dostawcy, modelu i przycisk parametrów."""
        label = ctk.CTkLabel(
            self,
            text="Model AI",
            font=FONTS["subheading"],
            anchor="w"
        )
        label.pack(pady=(10, 10), padx=20, fill="x")
        
        provider_label = ctk.CTkLabel(
            self,
            text="Dostawca:",
            font=FONTS["small"],
            anchor="w"
        )
        provider_label.pack(pady=(5, 2), padx=20, anchor="w")
        
        self.provider_var = ctk.StringVar(value="openai")
        self.provider_menu = ctk.CTkOptionMenu(
            self,
            variable=self.provider_var,
            values=PROVIDERS,
            command=self._on_provider_change
        )
        self.provider_menu.pack(pady=(0, 10), padx=20, fill="x")
        
        model_label = ctk.CTkLabel(
            self,
            text="Model:",
            font=FONTS["small"],
            anchor="w"
        )
        model_label.pack(pady=(5, 2), padx=20, anchor="w")
        
        self.model_var = ctk.StringVar(value="gpt-5.2")
        self.model_menu = ctk.CTkOptionMenu(
            self,
            variable=self.model_var,
            values=get_models_for_provider("openai"),
            command=self._on_model_change
        )
        self.model_menu.pack(pady=(0, 10), padx=20, fill="x")

        self.model_params_btn = ctk.CTkButton(
            self,
            text="Parametry modelu",
            command=self._open_model_params_dialog,
            height=35,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["surface"],
            border_width=2,
            border_color=COLORS["primary"],
            text_color_disabled=COLORS["text_disabled"]
        )
        self.model_params_btn.pack(pady=(8, 6), padx=20, fill="x")
        self.model_params_btn.pack_configure(pady=(8, 15))
    
    def _create_strategy_section(self) -> None:
        """! @brief Buduje sekcję wyboru strategii analizy (radio + tooltipy)."""
        label = ctk.CTkLabel(
            self,
            text="Strategia Analizy",
            font=FONTS["subheading"],
            anchor="w"
        )
        label.pack(pady=(10, 10), padx=20, fill="x")
        
        strategies_info = {
            "direct": ("Bezpośrednia", "Dane są przesyłane bezpośrednio do AI w prompcie."),
            "code": ("Wykonywanie kodu", "AI generuje kod Python uruchamiany w izolowanym środowisku po stronie dostawcy API."),
            "hybrid": ("Hybrydowa", "AI planuje analizę, obliczenia są wykonywane lokalnie, a potem AI interpretuje wyniki.")
        }
        
        self.strategy_var = ctk.StringVar(value="direct")
        
        for strategy_key, (strategy_name, strategy_desc) in strategies_info.items():
            strategy_frame = ctk.CTkFrame(self, fg_color="transparent")
            strategy_frame.pack(pady=2, padx=20, fill="x")
            
            radio = ctk.CTkRadioButton(
                strategy_frame,
                text=strategy_name,
                variable=self.strategy_var,
                value=strategy_key,
                command=self._on_strategy_change,
                font=FONTS["body"]
            )
            radio.pack(side="left", anchor="w")
            radio.bind("<Enter>", lambda event, desc=strategy_desc: self._show_strategy_tooltip(event, desc))
            radio.bind("<Motion>", self._move_strategy_tooltip)
            radio.bind("<Leave>", self._hide_strategy_tooltip)
            radio.bind("<ButtonPress-1>", self._hide_strategy_tooltip)

    def _show_strategy_tooltip(self, event: Any, text: str) -> None:
        """! @brief Planuje wyświetlenie tooltipa z opisem strategii (opóźnienie ~150 ms).

        @param event Zdarzenie Tk (pozycja kursora).
        @param text Treść podpowiedzi.

        @details Opóźnienie ogranicza migotanie przy szybkim ruchu myszy, zdarzenie Leave na tooltipie
                 zapewnia jego zamknięcie.
        """
        if self._strategy_tooltip_after_id is not None:
            self.after_cancel(self._strategy_tooltip_after_id)
            self._strategy_tooltip_after_id = None

        x_root, y_root = event.x_root, event.y_root

        def _do_show() -> None:
            self._strategy_tooltip_after_id = None
            self._hide_strategy_tooltip()
            tooltip = ctk.CTkToplevel(self)
            tooltip.overrideredirect(True)
            tooltip.attributes("-topmost", True)
            ctk.CTkLabel(
                tooltip,
                text=text,
                font=FONTS["small"],
                fg_color=COLORS["surface"],
                text_color=COLORS["text_secondary"],
                corner_radius=8,
                wraplength=260,
                padx=10,
                pady=6,
                justify="left"
            ).pack()
            tooltip.geometry(f"+{x_root + 14}+{y_root + 14}")
            tooltip.bind("<Leave>", self._hide_strategy_tooltip)
            self._strategy_tooltip = tooltip

        self._strategy_tooltip_after_id = self.after(150, _do_show)

    def _move_strategy_tooltip(self, event: Any) -> None:
        """! @brief Przesuwa okno tooltipa wraz z kursorem."""
        if self._strategy_tooltip is not None and self._strategy_tooltip.winfo_exists():
            self._strategy_tooltip.geometry(f"+{event.x_root + 14}+{event.y_root + 14}")

    def _hide_strategy_tooltip(self, _event: Any = None) -> None:
        """! @brief Zamyka tooltip strategii i anuluje zaplanowane after()."""
        if self._strategy_tooltip_after_id is not None:
            self.after_cancel(self._strategy_tooltip_after_id)
            self._strategy_tooltip_after_id = None
        if self._strategy_tooltip is not None and self._strategy_tooltip.winfo_exists():
            self._strategy_tooltip.destroy()
        self._strategy_tooltip = None
    
    def _on_strategy_change(self) -> None:
        """! @brief Wywoływane przy zmianie radiobuttona strategii, informuje MainWindow."""
        strategy = self.strategy_var.get()
        self.current_strategy = strategy
        self._hide_strategy_tooltip()
        
        if self.on_strategy_change:
            self.on_strategy_change(strategy)
        
        logger.info(f"Zmieniono strategię na: {strategy}")
    
    def _select_file(self) -> None:
        """! @brief Otwiera natywny dialog wyboru pliku danych."""
        filepath = filedialog.askopenfilename(
            title="Wybierz plik z danymi",
            filetypes=[
                ("Pliki CSV", "*.csv"),
                ("Pliki Excel", "*.xlsx *.xls"),
                ("Pliki JSON", "*.json"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        
        if filepath:
            self.on_file_load(filepath)
    
    def _on_provider_change(self, provider: str) -> None:
        """! @brief Aktualizuje listę modeli i limity tokenów po zmianie dostawcy."""
        self.current_provider = provider
        
        models = get_models_for_provider(provider)
        self.model_menu.configure(values=models)
        self.model_var.set(models[0])
        self.current_model = models[0]
        self.model_params["max_output_tokens"] = get_default_max_output_tokens(
            self.current_provider,
            self.current_model
        )
        self.model_params = with_model_token_limits(
            self.current_provider,
            self.current_model,
            self.model_params
        )
        
        self._notify_model_change()
    
    def _on_model_change(self, model: str) -> None:
        """! @brief Aktualizuje limity tokenów po zmianie modelu z menu."""
        self.current_model = model
        self.model_params["max_output_tokens"] = get_default_max_output_tokens(
            self.current_provider,
            self.current_model
        )
        self.model_params = with_model_token_limits(
            self.current_provider,
            self.current_model,
            self.model_params
        )
        
        self._notify_model_change()

    def _notify_model_change(self) -> None:
        """! @brief Wywołuje callback on_model_change z przefiltrowanymi parametrami."""
        self.model_params = with_model_token_limits(
            self.current_provider,
            self.current_model,
            self.model_params
        )
        effective_params = self._get_effective_model_params()
        self.on_model_change(
            self.current_provider,
            self.current_model,
            effective_params
        )

    def _get_supported_param_keys(self) -> list[str]:
        """! @brief Zwraca listę kluczy parametrów wspieranych przez bieżący model."""
        return get_supported_param_keys(
            self.current_provider,
            self.current_model,
            self.model_params
        )

    def _get_effective_model_params(self) -> Dict[str, Any]:
        """! @brief Zwraca model_params ograniczone do wspieranych kluczy."""
        return filter_model_params(
            self.current_provider,
            self.current_model,
            self.model_params
        )

    def _open_model_params_dialog(self) -> None:
        """! @brief Otwiera modalne okno edycji parametrów (temperature, top_p, itd.)."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Parametry modelu")
        dialog.geometry("520x760")
        dialog.grab_set()

        container = ctk.CTkScrollableFrame(dialog, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=16, pady=16)

        param_tooltip_ref: list = [None]
        param_tooltip_after_ref: list = [None]

        def show_param_tooltip(event: Any, tooltip_text: str) -> None:
            if param_tooltip_after_ref[0] is not None:
                try:
                    dialog.after_cancel(param_tooltip_after_ref[0])
                except Exception:
                    pass
                param_tooltip_after_ref[0] = None
            hide_param_tooltip()
            x_root, y_root = event.x_root, event.y_root

            def _do_show() -> None:
                param_tooltip_after_ref[0] = None
                tt = ctk.CTkToplevel(dialog)
                tt.overrideredirect(True)
                tt.attributes("-topmost", True)
                ctk.CTkLabel(
                    tt,
                    text=tooltip_text,
                    font=FONTS["small"],
                    fg_color=COLORS["surface"],
                    text_color=COLORS["text_secondary"],
                    corner_radius=8,
                    wraplength=260,
                    padx=10,
                    pady=6,
                    justify="left",
                ).pack()
                tt.geometry(f"+{x_root + 14}+{y_root + 14}")
                tt.bind("<Leave>", lambda e: hide_param_tooltip())
                param_tooltip_ref[0] = tt

            param_tooltip_after_ref[0] = dialog.after(150, _do_show)

        def move_param_tooltip(event: Any) -> None:
            if param_tooltip_ref[0] is not None and param_tooltip_ref[0].winfo_exists():
                param_tooltip_ref[0].geometry(f"+{event.x_root + 14}+{event.y_root + 14}")

        def hide_param_tooltip(_event: Any = None) -> None:
            if param_tooltip_after_ref[0] is not None:
                try:
                    dialog.after_cancel(param_tooltip_after_ref[0])
                except Exception:
                    pass
                param_tooltip_after_ref[0] = None
            if param_tooltip_ref[0] is not None and param_tooltip_ref[0].winfo_exists():
                param_tooltip_ref[0].destroy()
            param_tooltip_ref[0] = None

        def bind_param_tooltip(widget: Any, tooltip_text: str) -> None:
            widget.bind("<Enter>", lambda e, t=tooltip_text: show_param_tooltip(e, t))
            widget.bind("<Motion>", move_param_tooltip)
            widget.bind("<Leave>", lambda e: hide_param_tooltip(e))

        supported_keys = set(self._get_supported_param_keys())

        top_k_val = self.model_params.get("top_k")
        temp_max = get_temperature_max(self.current_provider)
        vars_map: Dict[str, Any] = {
            "streaming": ctk.BooleanVar(value=self.model_params["streaming"]),
            "temperature": ctk.DoubleVar(
                value=normalize_temperature(self.current_provider, self.model_params["temperature"])
            ),
            "top_p": ctk.DoubleVar(value=float(self.model_params["top_p"])),
            "top_k": ctk.IntVar(value=int(top_k_val) if top_k_val is not None else 0),
            "top_logprobs": ctk.IntVar(value=int(self.model_params["top_logprobs"])),
            "max_output_tokens": ctk.IntVar(value=int(self.model_params["max_output_tokens"])),
            "reasoning_effort": ctk.StringVar(value=str(self.model_params["reasoning_effort"])),
            "output_verbosity": ctk.StringVar(value=str(self.model_params["output_verbosity"])),
            "truncation": ctk.StringVar(value=str(self.model_params["truncation"])),
            "service_tier": ctk.StringVar(value=str(self.model_params["service_tier"])),
            "store": ctk.BooleanVar(value=self.model_params["store"]),
            "thinking_budget": ctk.IntVar(value=int(self.model_params.get("thinking_budget", 0))),
        }

        if "streaming" in supported_keys:
            streaming_cb = ctk.CTkCheckBox(
                container,
                text="streaming",
                variable=vars_map["streaming"],
                font=FONTS["small"]
            )
            streaming_cb.pack(anchor="w", pady=(0, 10))
            bind_param_tooltip(streaming_cb, "Strumieniowanie odpowiedzi.")

        def add_slider(
            param_name: str,
            tooltip_text: str,
            key: str,
            from_: float,
            to: float,
            steps: int
        ) -> None:
            label = ctk.CTkLabel(container, text=param_name, font=FONTS["small"], anchor="w")
            label.pack(fill="x")
            bind_param_tooltip(label, tooltip_text)
            value_label = ctk.CTkLabel(container, text="", font=FONTS["small"], anchor="w")
            value_label.pack(fill="x", pady=(0, 2))
            int_keys = {"max_output_tokens", "top_logprobs", "top_k", "thinking_budget"}
            format_value = lambda value: str(int(value)) if key in int_keys else f"{float(value):.2f}"
            slider = ctk.CTkSlider(
                container,
                from_=from_,
                to=to,
                number_of_steps=steps,
                command=lambda value: value_label.configure(text=format_value(value))
            )
            slider.pack(fill="x", pady=(0, 8))
            initial = vars_map[key].get()
            slider.set(initial)
            value_label.configure(text=format_value(initial))
            slider.configure(command=lambda value, v=vars_map[key], vl=value_label, k=key: (
                v.set(int(value) if k in int_keys else float(value)),
                vl.configure(text=format_value(value))
            ))

        if "thinking_budget" in supported_keys:
            max_tokens_limit = get_max_output_tokens_limit(self.current_provider, self.current_model)
            cap = min(16000, max(1, max_tokens_limit - 1))
            vars_map["thinking_budget"].set(min(cap, max(0, int(vars_map["thinking_budget"].get()))))
            add_slider(
                "thinking_budget",
                "Budżet tokenów na rozumowanie (Extended Thinking). 0 = wyłączone.",
                "thinking_budget",
                0,
                cap,
                max(1, min(63, cap // 256))
            )

        temp_slider_ref: list = []
        if "temperature" in supported_keys:
            thinking_on = (
                "thinking_budget" in supported_keys
                and int(vars_map["thinking_budget"].get()) > 0
            )
            temp_label = ctk.CTkLabel(container, text="temperature", font=FONTS["small"], anchor="w")
            temp_label.pack(fill="x")
            bind_param_tooltip(
                temp_label,
                "Losowość odpowiedzi (większa wartość = bardziej kreatywna). "
                "OpenAI i DeepSeek: 0–2; Anthropic: 0–1. Przy włączonym Extended Thinking ustawiane na 1."
            )
            temp_value_label = ctk.CTkLabel(container, text="", font=FONTS["small"], anchor="w")
            temp_value_label.pack(fill="x", pady=(0, 2))
            if thinking_on:
                vars_map["temperature"].set(1.0)
            initial_t = float(vars_map["temperature"].get())
            temp_steps = max(20, min(63, int(round(temp_max * 20))))
            temp_slider = ctk.CTkSlider(
                container,
                from_=0.0,
                to=temp_max,
                number_of_steps=temp_steps,
                command=lambda value: temp_value_label.configure(text=f"{float(value):.2f}")
            )
            temp_slider.pack(fill="x", pady=(0, 8))
            temp_slider.set(initial_t)
            temp_value_label.configure(text=f"{initial_t:.2f}")
            temp_slider.configure(
                command=lambda value, v=vars_map["temperature"], vl=temp_value_label: (
                    v.set(float(value)),
                    vl.configure(text=f"{float(value):.2f}")
                )
            )
            if thinking_on:
                temp_slider.configure(state="disabled")
            temp_slider_ref[:] = [temp_slider, temp_value_label]

            def _sync_temp_to_thinking(*_args: object) -> None:
                budget = int(vars_map["thinking_budget"].get())
                if budget > 0:
                    vars_map["temperature"].set(1.0)
                    temp_slider_ref[0].set(1.0)
                    temp_slider_ref[1].configure(text="1.00")
                    temp_slider_ref[0].configure(state="disabled")
                else:
                    temp_slider_ref[0].configure(state="normal")

            if "thinking_budget" in supported_keys:
                vars_map["thinking_budget"].trace_add("write", _sync_temp_to_thinking)

        if "top_p" in supported_keys:
            add_slider("top_p", "Różnorodność generowanego tekstu.", "top_p", 0.0, 1.0, 20)
        if "top_k" in supported_keys:
            add_slider("top_k", "Próbkowanie z K najbardziej prawdopodobnych tokenów.", "top_k", 0, 100, 20)
        if "top_logprobs" in supported_keys:
            add_slider("top_logprobs", "Liczba alternatywnych tokenów do analizy.", "top_logprobs", 0, 20, 20)
        if "max_output_tokens" in supported_keys:
            max_tokens_limit = get_max_output_tokens_limit(self.current_provider, self.current_model)
            vars_map["max_output_tokens"].set(
                min(max_tokens_limit, int(vars_map["max_output_tokens"].get()))
            )
            span = max_tokens_limit - MIN_OUTPUT_TOKENS
            steps = max(1, min(63, span))
            add_slider(
                "max_output_tokens",
                "Maksymalna długość odpowiedzi (w tokenach).",
                "max_output_tokens",
                MIN_OUTPUT_TOKENS,
                max_tokens_limit,
                steps
            )

        if "reasoning_effort" in supported_keys:
            re_label = ctk.CTkLabel(
                container,
                text="reasoning_effort",
                font=FONTS["small"],
                anchor="w"
            )
            re_label.pack(fill="x")
            bind_param_tooltip(re_label, "Głębokość rozumowania modelu.")
            ctk.CTkOptionMenu(
                container,
                variable=vars_map["reasoning_effort"],
                values=get_reasoning_effort_options(self.current_provider, self.current_model),
            ).pack(fill="x", pady=(0, 8))

        if "output_verbosity" in supported_keys:
            ov_label = ctk.CTkLabel(
                container,
                text="output_verbosity",
                font=FONTS["small"],
                anchor="w"
            )
            ov_label.pack(fill="x")
            bind_param_tooltip(ov_label, "Szczegółowość odpowiedzi.")
            ctk.CTkOptionMenu(
                container,
                variable=vars_map["output_verbosity"],
                values=["low", "medium", "high"],
            ).pack(fill="x", pady=(0, 8))

        if "truncation" in supported_keys:
            tr_label = ctk.CTkLabel(
                container,
                text="truncation",
                font=FONTS["small"],
                anchor="w"
            )
            tr_label.pack(fill="x")
            bind_param_tooltip(tr_label, "Obsługa zbyt długiego kontekstu.")
            ctk.CTkOptionMenu(
                container,
                variable=vars_map["truncation"],
                values=["disabled", "auto"],
            ).pack(fill="x", pady=(0, 8))

        if "service_tier" in supported_keys:
            st_label = ctk.CTkLabel(
                container,
                text="service_tier",
                font=FONTS["small"],
                anchor="w"
            )
            st_label.pack(fill="x")
            bind_param_tooltip(st_label, "Priorytet i klasa obsługi zapytania.")
            ctk.CTkOptionMenu(
                container,
                variable=vars_map["service_tier"],
                values=["auto", "default", "flex", "priority"],
            ).pack(fill="x", pady=(0, 8))

        if "store" in supported_keys:
            store_cb = ctk.CTkCheckBox(
                container,
                text="store",
                variable=vars_map["store"],
                font=FONTS["small"]
            )
            store_cb.pack(anchor="w", pady=(0, 8))
            bind_param_tooltip(store_cb, "Zapamiętaj odpowiedź po stronie API.")

        buttons = ctk.CTkFrame(container, fg_color="transparent")
        buttons.pack(fill="x", pady=(14, 0))

        def reset_defaults():
            self.model_params = get_default_model_params()
            self.model_params["max_output_tokens"] = get_max_output_tokens_limit(
                self.current_provider,
                self.current_model
            )
            self._notify_model_change()
            dialog.destroy()

        def apply_changes():
            top_k_raw = int(vars_map["top_k"].get())
            thinking_budget_val = int(vars_map["thinking_budget"].get())
            if thinking_budget_val > 0 and "thinking_budget" in supported_keys:
                temp_val = 1.0
            else:
                temp_val = round(
                    normalize_temperature(self.current_provider, vars_map["temperature"].get()),
                    2,
                )
            self.model_params = {
                "streaming": bool(vars_map["streaming"].get()),
                "temperature": temp_val,
                "top_p": round(float(vars_map["top_p"].get()), 2),
                "top_k": top_k_raw if top_k_raw > 0 else None,
                "top_logprobs": int(vars_map["top_logprobs"].get()),
                "max_output_tokens": int(vars_map["max_output_tokens"].get()),
                "reasoning_effort": str(vars_map["reasoning_effort"].get()),
                "output_verbosity": str(vars_map["output_verbosity"].get()),
                "truncation": str(vars_map["truncation"].get()),
                "service_tier": str(vars_map["service_tier"].get()),
                "store": bool(vars_map["store"].get()),
                "thinking_budget": int(vars_map["thinking_budget"].get()),
            }
            self.model_params = with_model_token_limits(
                self.current_provider,
                self.current_model,
                self.model_params
            )
            self._notify_model_change()
            dialog.destroy()

        ctk.CTkButton(
            buttons,
            text="Domyślne",
            command=reset_defaults,
            fg_color=COLORS["surface"],
            border_width=2,
            border_color=COLORS["accent"],
            text_color_disabled=COLORS["text_disabled"]
        ).pack(side="left", padx=(0, 8), fill="x", expand=True)

        ctk.CTkButton(
            buttons,
            text="Zastosuj",
            command=apply_changes,
            fg_color=COLORS["primary"],
            text_color_disabled=COLORS["text_disabled"]
        ).pack(side="left", fill="x", expand=True)
    
    def update_data_preview(self, df: pd.DataFrame) -> None:
        """!
        Aktualizuj podgląd danych.
        
        Args:
            df: DataFrame z danymi
        """
        preview_text = f"✅ Dane wczytane\n\n"
        preview_text += f"Wierszy: {len(df)}\n"
        preview_text += f"Kolumn: {len(df.columns)}\n\n"
        preview_text += f"Kolumny:\n"
        
        for col in df.columns[:PREVIEW_MAX_COLUMNS]:
            preview_text += f"  • {col}\n"
        
        if len(df.columns) > PREVIEW_MAX_COLUMNS:
            preview_text += f"  ... i {len(df.columns) - PREVIEW_MAX_COLUMNS} więcej"
        
        self.data_preview.configure(state="normal")
        self.data_preview.delete("1.0", "end")
        self.data_preview.insert("1.0", preview_text)
        self.data_preview.configure(state="disabled")
    
    def reset_data_preview(self) -> None:
        """! @brief Przywraca podgląd danych do stanu „brak danych” (np. przy resecie sesji)."""
        self.data_preview.configure(state="normal")
        self.data_preview.delete("1.0", "end")
        self.data_preview.insert("1.0", "Brak załadowanych danych")
        self.data_preview.configure(state="disabled")

    def update_stats(self, requests: int, tokens: int, cost: float) -> None:
        """!
        Aktualizuj statystyki (zachowane dla kompatybilności).
        Statystyki są teraz w osobnej zakładce.
        """
        pass