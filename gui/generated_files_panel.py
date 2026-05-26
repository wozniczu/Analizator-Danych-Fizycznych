##
## @file generated_files_panel.py
## @brief Panel listy plików z Code Interpreter/Code execution: zapis do temp, przyciski Otwórz/Otwórz folder.

import os
import tempfile
import customtkinter as ctk
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from gui.styles import COLORS, FONTS, DIMENSIONS
from utils.logger import logger


class GeneratedFilesPanel(ctk.CTkFrame):
    """!
    @brief Wyświetla listę plików z wyników strategii Code, zapisuje je w temp, umożliwia otwarcie pliku/folderu.
    """

    def __init__(self, parent: Any, **kwargs: Any) -> None:
        """!
        @brief Inicjalizuje panel listy plików (nagłówek, placeholder, scroll).

        @param parent Widget rodzica.
        """
        super().__init__(parent, fg_color=COLORS["surface"], corner_radius=DIMENSIONS["corner_radius"], **kwargs)
        self._output_dir: Optional[str] = None
        self._file_entries: List[Tuple[str, str]] = []

        self._create_widgets()

    def _create_widgets(self) -> None:
        """! @brief Tworzy nagłówek, przycisk folderu, placeholder i ramkę przewijaną."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=15)
        ctk.CTkLabel(
            header,
            text="📁 Pliki wygenerowane przez AI",
            font=FONTS["heading"],
            anchor="w",
        ).pack(side="left")
        self._open_folder_btn = ctk.CTkButton(
            header,
            text="Otwórz folder",
            command=self._open_folder,
            width=120,
            height=35,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["primary"],
            state="disabled",
            text_color_disabled=COLORS["text_disabled"],
        )
        self._open_folder_btn.pack(side="right")
        self._placeholder = ctk.CTkLabel(
            self,
            text="Brak plików. Użyj strategii „Wykonywanie kodu” (OpenAI lub Anthropic),\naby wygenerować wykresy i raporty.",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"],
            justify="center",
        )
        self._placeholder.pack(expand=True, pady=40, padx=20)
        self._scroll = ctk.CTkScrollableFrame(self, fg_color=COLORS["background"], corner_radius=8)

    def _open_folder(self) -> None:
        """! @brief Otwiera katalog wyjściowy w Eksploratorze Windows (os.startfile)."""
        if self._output_dir and os.path.isdir(self._output_dir):
            try:
                os.startfile(self._output_dir)
            except Exception as e:
                logger.warning(f"Otwieranie folderu: {e}")

    def _open_file(self, path: str) -> None:
        """! @brief Otwiera pojedynczy plik domyślną aplikacją systemu."""
        if path and os.path.isfile(path):
            try:
                os.startfile(path)
            except Exception as e:
                logger.warning(f"Otwieranie pliku: {e}")

    def set_generated_files(self, files: List[Dict[str, Any]]) -> None:
        """!
        @brief Zapisuje pliki w katalogu temp z timestampem, buduje listę przycisków Otwórz, przy duplikatach nazw dodaje _n.

        @param files Lista słowników z kluczami filename (str) i data (bytes).
        """
        self._file_entries.clear()
        for child in self._scroll.winfo_children():
            child.destroy()
        if not files:
            self._output_dir = None
            self._open_folder_btn.configure(state="disabled")
            self._scroll.pack_forget()
            self._placeholder.pack(expand=True, pady=40, padx=20)
            return
        base = os.path.join(tempfile.gettempdir(), "physics_analyzer_output")
        os.makedirs(base, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_dir = os.path.join(base, timestamp)
        os.makedirs(self._output_dir, exist_ok=True)
        seen: Dict[str, int] = {}
        for entry in files:
            filename = entry.get("filename") or "plik.bin"
            data = entry.get("data", b"")
            if isinstance(filename, str) and os.path.sep in filename:
                filename = os.path.basename(filename)
            key = filename
            n = seen.get(key, 0)
            seen[key] = n + 1
            if n:
                base_name, ext = os.path.splitext(filename)
                filename = f"{base_name}_{n}{ext}"
            path = os.path.join(self._output_dir, filename)
            try:
                with open(path, "wb") as f:
                    f.write(data)
                self._file_entries.append((filename, path))
            except Exception as e:
                logger.warning(f"Zapis pliku {filename}: {e}")
        self._placeholder.pack_forget()
        self._scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        for name, path in self._file_entries:
            row = ctk.CTkFrame(self._scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=name, font=FONTS["body"], anchor="w").pack(side="left", fill="x", expand=True)
            ctk.CTkButton(
                row,
                text="Otwórz",
                width=80,
                height=28,
                command=lambda p=path: self._open_file(p),
                fg_color=COLORS["primary"],
                corner_radius=DIMENSIONS["corner_radius"],
                text_color_disabled=COLORS["text_disabled"],
            ).pack(side="right", padx=(8, 0))
        self._open_folder_btn.configure(state="normal")
