##
## @file stats_panel.py
## @brief Panel szczegółowych statystyk użycia API.
##
import customtkinter as ctk
from typing import Any, Callable, Dict, List
from gui.styles import COLORS, FONTS, DIMENSIONS
from utils.logger import logger


class StatsPanel(ctk.CTkFrame):
    """!
    @brief Panel szczegółowych statystyk użycia API.
    
    @details Klasa StatsPanel wyświetla szczegółowe informacje o:
             - Całkowitym użyciu API w bieżącej sesji
             - Rozbicie kosztów na dostawców (OpenAI, Anthropic, DeepSeek)
             - Statystyki per model
             - Historię ostatnich 10 zapytań (z przyciskiem „Podgląd” promptu i odpowiedzi)
    
    @see MainWindow, CostTracker
    """
    
    def __init__(self, parent: Any, get_detailed_stats: Callable[[], Dict[str, Any]]) -> None:
        """!
        @brief Konstruktor panelu statystyk.
        
        @param parent Widget rodzica (CTkTabview).
        @param get_detailed_stats Callback zwracający słownik ze statystykami.
        """
        super().__init__(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        
        self.get_detailed_stats = get_detailed_stats
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """! @brief Buduje nagłówek z przyciskiem Odśwież, content_frame ze statystykami i historią."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(pady=15, padx=20, fill="x")
        
        header = ctk.CTkLabel(
            header_frame,
            text="Szczegółowe statystyki",
            font=FONTS["heading"],
            anchor="w"
        )
        header.pack(side="left")
        refresh_btn = ctk.CTkButton(
            header_frame,
            text="Odśwież",
            command=self.refresh_stats,
            width=100,
            height=35,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_dark"],
            text_color_disabled=COLORS["text_disabled"]
        )
        refresh_btn.pack(side="right")
        self.content_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["background"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        self.content_frame.pack(
            pady=(0, 15),
            padx=20,
            fill="both",
            expand=True
        )
        
        self._show_placeholder()
    
    def _show_placeholder(self) -> None:
        """! @brief Wyświetla placeholder, gdy brak danych statystycznych."""
        placeholder = ctk.CTkLabel(
            self.content_frame,
            text="Statystyki pojawią się tutaj po wykonaniu zapytań do AI\n\nKliknij 'Odśwież' aby zaktualizować",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        placeholder.pack(expand=True, pady=50)
    
    def _clear_content(self) -> None:
        """! @brief Usuwa widgety z głównej ramki treści."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def refresh_stats(self) -> None:
        """! @brief Ponownie pobiera statystyki z callbacku i rysuje UI."""
        stats = self.get_detailed_stats()
        self._display_stats(stats)
        logger.info("Odświeżono statystyki")
    
    def _display_stats(self, stats: Dict[str, Any]) -> None:
        """! @brief Buduje widżety podsumowań, dostawców i historii zapytań."""
        self._clear_content()
        
        session = stats.get("session", {})
        providers = stats.get("providers", {})
        
        if session.get("requests", 0) == 0:
            self._show_placeholder()
            return
        
        self._create_section_header("Podsumowanie sesji")
        self._create_session_summary(session)
        
        if providers:
            self._create_section_header("Statystyki dostawców")
            for provider_name, provider_stats in providers.items():
                self._create_provider_card(provider_name, provider_stats)
        
        recent = stats.get("recent_requests", [])
        if recent:
            self._create_section_header("Ostatnie zapytania")
            self._create_recent_requests(recent)
    
    def _create_section_header(self, text: str) -> None:
        """! @brief Dodaje nagłówek sekcji (tytuł + separator)."""
        header = ctk.CTkLabel(
            self.content_frame,
            text=text,
            font=FONTS["subheading"],
            anchor="w"
        )
        header.pack(pady=(20, 10), padx=10, fill="x")
    
    def _create_session_summary(self, session: Dict[str, Any]) -> None:
        """! @brief Wyświetla karty z sumą zapytań, tokenów, kosztem i czasem."""
        frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        frame.pack(pady=5, padx=10, fill="x")
        
        avg_elapsed = session.get("avg_elapsed_seconds")
        avg_elapsed_str = f"{avg_elapsed:.1f} s" if avg_elapsed is not None else "-"

        stats_data = [
            ("Czas trwania", f"{session.get('duration_minutes', 0):.1f} min"),
            ("Zapytania", str(session.get("requests", 0))),
            ("Tokeny wejściowe", f"{session.get('tokens_input', 0):,}"),
            ("Tokeny wyjściowe", f"{session.get('tokens_output', 0):,}"),
            ("Łącznie tokenów", f"{session.get('total_tokens', 0):,}"),
            ("Łączny koszt", f"${session.get('total_cost_usd', 0):.6f}"),
            ("Śr. koszt/zapytanie", f"${session.get('avg_cost_per_request', 0):.6f}"),
            ("Śr. tokeny/zapytanie", f"{session.get('avg_tokens_per_request', 0):.0f}"),
            ("Śr. czas odpowiedzi", avg_elapsed_str),
        ]
        
        for i, (label, value) in enumerate(stats_data):
            row = i // 2
            col = i % 2
            
            stat_frame = ctk.CTkFrame(frame, fg_color="transparent")
            stat_frame.grid(row=row, column=col, padx=15, pady=8, sticky="w")
            
            label_widget = ctk.CTkLabel(
                stat_frame,
                text=label,
                font=FONTS["small"],
                text_color=COLORS["text_secondary"]
            )
            label_widget.pack(anchor="w")
            
            value_widget = ctk.CTkLabel(
                stat_frame,
                text=value,
                font=("Arial", 16, "bold"),
                text_color=COLORS["accent"]
            )
            value_widget.pack(anchor="w")
        
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
    
    def _create_provider_card(self, provider_name: str, provider_stats: Dict[str, Any]) -> None:
        """! @brief Tworzy ramkę z agregatami dla jednego dostawcy API."""
        provider_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        provider_frame.pack(pady=5, padx=10, fill="x")
        
        header = ctk.CTkLabel(
            provider_frame,
            text=f"{provider_name.upper()}",
            font=FONTS["subheading"],
            anchor="w"
        )
        header.pack(pady=(15, 5), padx=15, fill="x")
        
        summary_frame = ctk.CTkFrame(provider_frame, fg_color="transparent")
        summary_frame.pack(pady=5, padx=15, fill="x")
        
        summary_data = [
            ("Zapytania", str(provider_stats.get("requests", 0))),
            ("Tokeny", f"{provider_stats.get('total_tokens', 0):,}"),
            ("Koszt", f"${provider_stats.get('total_cost', 0):.6f}"),
        ]
        
        for label, value in summary_data:
            item = ctk.CTkFrame(summary_frame, fg_color="transparent")
            item.pack(side="left", padx=20)
            
            ctk.CTkLabel(
                item,
                text=label,
                font=FONTS["small"],
                text_color=COLORS["text_secondary"]
            ).pack()
            
            ctk.CTkLabel(
                item,
                text=value,
                font=("Arial", 14, "bold"),
                text_color=COLORS["text"]
            ).pack()
        
        models = provider_stats.get("models", {})
        if models:
            models_label = ctk.CTkLabel(
                provider_frame,
                text="Modele:",
                font=FONTS["small"],
                text_color=COLORS["text_secondary"],
                anchor="w"
            )
            models_label.pack(pady=(10, 5), padx=15, fill="x")
            
            for model_name, model_stats in models.items():
                self._create_model_row(provider_frame, model_name, model_stats)
        
        ctk.CTkFrame(provider_frame, fg_color="transparent", height=10).pack()
    
    def _create_model_row(self, parent: Any, model_name: str, model_stats: Dict[str, Any]) -> None:
        """! @brief Dodaje wiersz tabeli statystyk dla pojedynczego modelu."""
        row = ctk.CTkFrame(
            parent,
            fg_color=COLORS["background"],
            corner_radius=5
        )
        row.pack(pady=3, padx=15, fill="x")
        
        name_label = ctk.CTkLabel(
            row,
            text=f"{model_name}",
            font=FONTS["body"],
            anchor="w",
            width=200
        )
        name_label.pack(side="left", padx=5, pady=8)
        
        stats_text = (
            f"{model_stats.get('requests', 0)} | "
            f"{model_stats.get('total_tokens', 0):,} | "
            f"${model_stats.get('total_cost', 0):.6f}"
        )
        
        stats_label = ctk.CTkLabel(
            row,
            text=stats_text,
            font=FONTS["small"],
            text_color=COLORS["text_secondary"]
        )
        stats_label.pack(side="right", padx=10, pady=8)
    
    def _create_recent_requests(self, recent: List[Dict[str, Any]]) -> None:
        """! @brief Buduje listę ostatnich zapytań z przyciskiem podglądu promptu."""
        frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        frame.pack(pady=5, padx=10, fill="x")
        
        header_row = ctk.CTkFrame(frame, fg_color=COLORS["primary_dark"])
        header_row.pack(fill="x", padx=10, pady=(10, 5))
        
        headers = ["Czas", "Dostawca", "Model", "Tokeny", "Koszt", "Czas gen.", ""]
        header_widths = {"": 80, "Czas gen.": 80}
        for header in headers:
            w = header_widths.get(header, 100)
            ctk.CTkLabel(
                header_row,
                text=header,
                font=("Arial", 12, "bold"),
                width=w
            ).pack(side="left", padx=5, pady=5)
        
        for req in reversed(recent[-10:]):
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            
            timestamp = req.get("timestamp", "")
            if timestamp:
                time_str = timestamp.split("T")[1][:5] if "T" in timestamp else timestamp[:5]
            else:
                time_str = "-"
            
            elapsed = req.get("elapsed_seconds")
            elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "-"
            
            values = [
                (time_str, 100),
                (req.get("provider", "-"), 100),
                (req.get("model", "-")[:15], 100),
                (f"{req.get('tokens_input', 0) + req.get('tokens_output', 0):,}", 100),
                (f"${req.get('cost', 0):.4f}", 100),
                (elapsed_str, 80),
            ]
            
            for value, w in values:
                ctk.CTkLabel(
                    row,
                    text=value,
                    font=FONTS["small"],
                    width=w
                ).pack(side="left", padx=5, pady=3)
            
            has_preview = req.get("prompt_display") is not None or req.get("response_display") is not None
            btn = ctk.CTkButton(
                row,
                text="Podgląd",
                width=80,
                height=28,
                font=FONTS["small"],
                fg_color=COLORS["primary"] if has_preview else COLORS["surface"],
                hover_color=COLORS["primary_dark"] if has_preview else None,
                state="normal" if has_preview else "disabled",
                command=lambda r=req: self._show_prompt_response_preview(r),
            )
            btn.pack(side="left", padx=5, pady=3)
        
        ctk.CTkFrame(frame, fg_color="transparent", height=10).pack()
    
    def _show_prompt_response_preview(self, req: Dict):
        """! @brief Otwiera okno z pełnym (lub obciętym) promptem i odpowiedzią."""
        prompt_display = req.get("prompt_display") or "Brak zapisanych danych promptu."
        response_display = req.get("response_display") or "Brak zapisanych danych odpowiedzi."
        timestamp = req.get("timestamp", "")[:19].replace("T", " ")
        title = f"Podgląd zapytania - {req.get('provider', '')} / {req.get('model', '')} - {timestamp}"
        
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry("800x600")
        win.minsize(500, 400)
        
        tabview = ctk.CTkTabview(win, fg_color=COLORS["background"])
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        tab_prompt = tabview.add("Prompt wysłany do modelu")
        tab_response = tabview.add("Odpowiedź modelu")
        
        font_tuple = ("Consolas", 11) if isinstance(FONTS.get("body"), tuple) else ("Consolas", 11)
        
        prompt_text = ctk.CTkTextbox(
            tab_prompt,
            font=font_tuple,
            wrap="word",
            fg_color=COLORS["surface"],
            text_color=COLORS["text"],
        )
        prompt_text.pack(fill="both", expand=True, padx=5, pady=5)
        prompt_text.insert("1.0", prompt_display)
        prompt_text.configure(state="disabled")
        
        response_text = ctk.CTkTextbox(
            tab_response,
            font=font_tuple,
            wrap="word",
            fg_color=COLORS["surface"],
            text_color=COLORS["text"],
        )
        response_text.pack(fill="both", expand=True, padx=5, pady=5)
        response_text.insert("1.0", response_display)
        response_text.configure(state="disabled")
        
        btn_close = ctk.CTkButton(
            win,
            text="Zamknij",
            command=win.destroy,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_dark"],
        )
        btn_close.pack(pady=(0, 10))

