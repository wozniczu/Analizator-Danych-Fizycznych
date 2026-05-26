##
## @file settings_panel.py
## @brief Panel ustawień aplikacji.
##
import customtkinter as ctk
from typing import Any, Callable, Optional
from gui.styles import COLORS, FONTS, DIMENSIONS, THEMES, get_theme_names, get_current_theme
from utils.logger import logger


class SettingsPanel(ctk.CTkFrame):
    """!
    @brief Panel ustawień aplikacji.
    
    @details Klasa SettingsPanel zapewnia:
             - Wybór motywu kolorystycznego (ciemny, jasny, niebieski, zielony, fioletowy)
             - Zarządzanie sesją (czyszczenie czatu, reset)
             - Wyświetlanie informacji o aplikacji
    
    @see MainWindow
    """
    
    def __init__(
        self,
        parent: Any,
        on_theme_change: Callable[[str], None],
        on_reset_session: Optional[Callable[[str], None]] = None,
        on_export_chat: Optional[Callable[[], None]] = None,
        include_chat_context_in_api: bool = False,
        on_include_chat_context_change: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """!
        @brief Inicjalizuje panel (motyw, sesja, info), wywołuje _create_widgets.

        @param parent Widget rodzica.
        @param on_theme_change Callback przy zmianie motywu (nazwa).
        @param on_reset_session Callback przy resecie sesji (opcjonalny).
        @param on_export_chat Callback przy eksporcie czatu (opcjonalny).
        @param include_chat_context_in_api Stan przełącznika kontekstu API.
        @param on_include_chat_context_change Callback przy zmianie (True = pełna historia do API).
        """
        super().__init__(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        
        self.on_theme_change = on_theme_change
        self.on_reset_session = on_reset_session
        self.on_export_chat = on_export_chat
        self.on_include_chat_context_change = on_include_chat_context_change
        self._include_chat_context_initial = bool(include_chat_context_in_api)

        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """! @brief Buduje sekcje: Wygląd, Sesja, Informacje."""
        self.content_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent"
        )
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self._create_appearance_section()
        self._create_session_section()
        self._create_info_section()

    def _create_section_header(self, text: str) -> None:
        """! @brief Dodaje nagłówek sekcji (FONTS['heading'])."""
        header = ctk.CTkLabel(
            self.content_frame,
            text=text,
            font=FONTS["heading"],
            anchor="w"
        )
        header.pack(pady=(20, 15), padx=10, fill="x")
    
    def _create_appearance_section(self) -> None:
        """! @brief Buduje sekcję wyboru motywu (dropdown + przycisk)."""
        self._create_section_header("Wygląd")
        section_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=COLORS["background"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        section_frame.pack(pady=5, padx=10, fill="x")
        theme_label = ctk.CTkLabel(
            section_frame,
            text="Motyw kolorystyczny:",
            font=FONTS["body"],
            anchor="w"
        )
        theme_label.pack(pady=(15, 5), padx=20, fill="x")
        
        theme_desc = ctk.CTkLabel(
            section_frame,
            text="Wybierz schemat kolorów dla aplikacji",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            anchor="w"
        )
        theme_desc.pack(pady=(0, 10), padx=20, fill="x")
        
        self.theme_var = ctk.StringVar(value=get_current_theme())
        
        themes_container = ctk.CTkFrame(section_frame, fg_color="transparent")
        themes_container.pack(pady=5, padx=20, fill="x")
        
        theme_icons = {
            "dark": "",
            "light": "",
            "blue": "",
            "green": "",
            "purple": ""
        }
        
        for theme_key, theme_name in get_theme_names():
            icon = theme_icons.get(theme_key, "⚪")
            
            radio = ctk.CTkRadioButton(
                themes_container,
                text=f"{icon} {theme_name}",
                variable=self.theme_var,
                value=theme_key,
                command=self._on_theme_select,
                font=FONTS["body"]
            )
            radio.pack(pady=5, anchor="w")
        
        apply_btn = ctk.CTkButton(
            section_frame,
            text="Zastosuj motyw",
            command=self._apply_theme,
            height=40,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_dark"],
            text_color_disabled=COLORS["text_disabled"]
        )
        apply_btn.pack(pady=15, padx=20, fill="x")
    
    def _create_session_section(self) -> None:
        """! @brief Buduje sekcję czyszczenia czatu, nowej sesji i eksportu czatu."""
        self._create_section_header("Sesja")
        
        section_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=COLORS["background"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        section_frame.pack(pady=5, padx=10, fill="x")
        
        desc = ctk.CTkLabel(
            section_frame,
            text="Zarządzaj bieżącą sesją analizy",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            anchor="w"
        )
        desc.pack(pady=(15, 10), padx=20, fill="x")

        ctx_label = ctk.CTkLabel(
            section_frame,
            text="Zapytania do modelu AI:",
            font=FONTS["body"],
            anchor="w",
        )
        ctx_label.pack(pady=(5, 4), padx=20, fill="x")
        ctx_hint = ctk.CTkLabel(
            section_frame,
            text=(
                "Włączenie opcji dołącza wcześniejsze wiadomości z czatu do każdego "
                "kolejnego żądania API."
            ),
            font=FONTS["small"],
            text_color=COLORS["text_secondary"],
            anchor="w",
            justify="left",
        )
        ctx_hint.pack(pady=(0, 8), padx=20, fill="x")
        self.include_chat_context_var = ctk.BooleanVar(
            value=self._include_chat_context_initial
        )
        include_ctx_switch = ctk.CTkSwitch(
            section_frame,
            text="Pełna historia czatu w żądaniach API",
            variable=self.include_chat_context_var,
            command=self._on_include_chat_context_toggle,
            font=FONTS["body"],
        )
        include_ctx_switch.pack(pady=(0, 12), padx=20, anchor="w")
        
        clear_chat_btn = ctk.CTkButton(
            section_frame,
            text="Wyczyść historię czatu",
            command=self._clear_chat,
            height=40,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["primary_dark"],
            border_width=2,
            border_color=COLORS["primary"],
            text_color_disabled=COLORS["text_disabled"]
        )
        clear_chat_btn.pack(pady=5, padx=20, fill="x")

        export_chat_btn = ctk.CTkButton(
            section_frame,
            text="Eksportuj czat",
            command=self._export_chat,
            height=40,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["surface"],
            hover_color=COLORS["primary_dark"],
            border_width=2,
            border_color=COLORS["accent"],
            text_color_disabled=COLORS["text_disabled"]
        )
        export_chat_btn.pack(pady=5, padx=20, fill="x")
        
        new_session_btn = ctk.CTkButton(
            section_frame,
            text="Nowa sesja",
            command=self._new_session,
            height=40,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["warning"],
            hover_color=COLORS["error"],
            text_color=COLORS["background"],
            text_color_disabled=COLORS["text_disabled"]
        )
        new_session_btn.pack(pady=(5, 15), padx=20, fill="x")
    
    def _create_info_section(self) -> None:
        """! @brief Wyświetla blok informacyjny o wersji i autorze."""
        self._create_section_header("Informacje")
        
        section_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=COLORS["background"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        section_frame.pack(pady=5, padx=10, fill="x")
        
        info_text = """Analizator Danych Fizycznych

Wersja: 1.0.0
Autor: Bartosz Woźnica"""
        
        info_label = ctk.CTkLabel(
            section_frame,
            text=info_text,
            font=FONTS["body"],
            anchor="w",
            justify="left"
        )
        info_label.pack(pady=15, padx=20, fill="x")
        
    
    def _on_theme_select(self) -> None:
        """! @brief Wywoływane przy zmianie wartości menu motywu."""
        selected = self.theme_var.get()
        logger.info(f"Wybrano motyw: {selected}")
    
    def _apply_theme(self) -> None:
        """! @brief Stosuje motyw i wywołuje callback on_theme_change."""
        selected = self.theme_var.get()
        if self.on_theme_change:
            self.on_theme_change(selected)
    
    def _clear_chat(self) -> None:
        """! @brief Prosi główne okno o wyczyszczenie historii czatu."""
        if self.on_reset_session:
            self.on_reset_session("chat")
    
    def _new_session(self) -> None:
        """! @brief Prosi główne okno o pełny reset sesji."""
        if self.on_reset_session:
            self.on_reset_session("all")

    def _export_chat(self) -> None:
        """! @brief Wywołuje callback eksportu historii czatu."""
        if self.on_export_chat:
            self.on_export_chat()

    def _on_include_chat_context_toggle(self) -> None:
        """! @brief Zapisuje wybór: czy wysyłać pełną historię czatu do API."""
        if self.on_include_chat_context_change:
            self.on_include_chat_context_change(
                bool(self.include_chat_context_var.get())
            )

