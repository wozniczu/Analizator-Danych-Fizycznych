##
## @file visualization_panel.py
## @brief Panel wizualizacji danych z wykresami matplotlib.
##
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from gui.styles import (
    COLORS, FONTS, DIMENSIONS,
    FIGURE_DEFAULT_SIZE, FIGURE_DPI, FIGURE_EXPORT_DPI,
    CHART_MAX_COLUMNS_LINE, CHART_MAX_COLUMNS_HIST, CHART_MAX_COLUMNS_BOX,
    CHART_MAX_COLUMNS_BAR, CHART_LABEL_MAX_LEN,
    FIGURE_ASPECT_WIDTH_MIN, FIGURE_ASPECT_WIDTH_MAX, FIGURE_ASPECT_FACTOR,
)
from utils.logger import logger


class VisualizationPanel(ctk.CTkFrame):
    """!
    @brief Panel wizualizacji danych z wykresami.
    
    @details Klasa VisualizationPanel zapewnia:
             - Generowanie wykresów (liniowy, punktowy, histogram, pudełkowy, słupkowy)
             - Dynamiczną zmianę typu wykresu
             - Eksport wykresów do plików (PNG, PDF, SVG, JPEG)
             - Automatyczne dostosowanie kolorów do motywu aplikacji
    
    @see MainWindow, matplotlib
    """
    
    def __init__(self, parent: Any) -> None:
        """!
        @brief Konstruktor panelu wizualizacji.
        
        @param parent Widget rodzica (CTkTabview).
        """
        super().__init__(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        
        self.current_figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.current_data: Optional[pd.DataFrame] = None
        self.current_plot_type: str = "line"
        self._ai_image_data: Optional[bytes] = None
        self._ai_image_title: str = "Wykres wygenerowany przez AI"
        self._ai_images: List[Dict[str, Any]] = []
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """! @brief Buduje nagłówek, typ wykresu, przyciski, canvas i sekcję obrazów AI."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(pady=15, padx=20, fill="x")
        
        header = ctk.CTkLabel(
            header_frame,
            text="📊 Wizualizacja",
            font=FONTS["heading"],
            anchor="w"
        )
        header.pack(side="left")
        
        self.clear_btn = ctk.CTkButton(
            header_frame,
            text="Wyczyść",
            command=self.clear_plot,
            width=90,
            height=35,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["surface"],
            border_width=2,
            border_color=COLORS["primary"],
            text_color_disabled=COLORS["text_disabled"]
        )
        self.clear_btn.pack(side="right", padx=(5, 0))
        
        self.export_btn = ctk.CTkButton(
            header_frame,
            text="Eksportuj",
            command=self._export_plot,
            width=90,
            height=35,
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["surface"],
            border_width=2,
            border_color=COLORS["accent"],
            text_color_disabled=COLORS["text_disabled"]
        )
        self.export_btn.pack(side="right", padx=(5, 0))
        
        options_frame = ctk.CTkFrame(self, fg_color=COLORS["background"], corner_radius=8)
        options_frame.pack(pady=(0, 10), padx=20, fill="x")
        
        options_label = ctk.CTkLabel(
            options_frame,
            text="Typ wykresu:",
            font=FONTS["body"],
            anchor="w"
        )
        options_label.pack(side="left", padx=(15, 10), pady=10)
        
        self.plot_type_var = ctk.StringVar(value="Liniowy")
        
        plot_types = [
            "Liniowy",
            "Punktowy",
            "Histogram",
            "Pudełkowy",
            "Słupkowy",
            "AI"
        ]
        
        self.plot_type_menu = ctk.CTkOptionMenu(
            options_frame,
            variable=self.plot_type_var,
            values=plot_types,
            command=self._on_plot_type_change,
            width=150,
            height=35,
            font=FONTS["body"],
            fg_color=COLORS["surface"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_dark"]
        )
        self.plot_type_menu.pack(side="left", padx=5, pady=10)
        
        self.ai_file_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        self.ai_file_label = ctk.CTkLabel(
            self.ai_file_frame,
            text="Plik:",
            font=FONTS["body"],
            anchor="w"
        )
        self.ai_file_label.pack(side="left", padx=(15, 5), pady=10)
        self.ai_file_var = ctk.StringVar(value="")
        self.ai_file_menu = ctk.CTkOptionMenu(
            self.ai_file_frame,
            variable=self.ai_file_var,
            values=["- brak -"],
            command=self._on_ai_file_change,
            width=220,
            height=35,
            font=FONTS["body"],
            fg_color=COLORS["surface"],
            button_color=COLORS["primary"],
            button_hover_color=COLORS["primary_dark"]
        )
        self.ai_file_menu.pack(side="left", padx=5, pady=10)
        self.ai_file_frame.pack(side="left", padx=5, pady=10)
        self.ai_file_frame.pack_forget()
        
        self.data_info_label = ctk.CTkLabel(
            options_frame,
            text="Brak danych",
            font=FONTS["small"],
            text_color=COLORS["text_secondary"]
        )
        self.data_info_label.pack(side="right", padx=15, pady=10)
        
        self.plot_container = ctk.CTkFrame(
            self,
            fg_color=COLORS["background"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        self.plot_container.pack(
            pady=(0, 15),
            padx=20,
            fill="both",
            expand=True
        )
        
        self._show_placeholder()
    
    def _show_placeholder(self) -> None:
        """! @brief Wyświetla komunikat zastępczy, gdy nie ma wykresu do pokazania."""
        placeholder = ctk.CTkLabel(
            self.plot_container,
            text="Wizualizacje pojawią się tutaj\n\n"
                 "1. Załaduj dane w zakładce Analiza\n"
                 "2. Zapytaj AI o wygenerowanie wykresu lub użyj szybkiej akcji\n"
                 "3. Wybierz typ wykresu do wyświetlenia",
            font=FONTS["body"],
            text_color=COLORS["text_secondary"]
        )
        placeholder.pack(expand=True)
    
    def _on_plot_type_change(self, choice: str) -> None:
        """! @brief Obsługuje zmianę typu wykresu w menu (np. linia, histogram)."""
        if choice == "AI":
            self.current_plot_type = "ai"
            self.plot_type_var.set("AI")
            if not self._ai_images and self._ai_image_data is not None:
                self._ai_images = [{"filename": self._ai_image_title, "data": self._ai_image_data}]
            self._update_ai_file_dropdown()
            self.ai_file_frame.pack(side="left", padx=5, pady=10)
            self._show_ai_view()
            return

        self.ai_file_frame.pack_forget()
        type_map = {
            "Liniowy": "line",
            "Punktowy": "scatter",
            "Histogram": "hist",
            "Pudełkowy": "box",
            "Słupkowy": "bar"
        }
        self.current_plot_type = type_map.get(choice, "line")
        logger.info(f"Zmieniono typ wykresu na: {self.current_plot_type}")
        if self.current_data is not None:
            self.plot_dataframe(self.current_data, self.current_plot_type)

    def _show_ai_view(self) -> None:
        """!
        Pokaż widok wykresu AI: obraz z listy _ai_images albo komunikat,
        że trzeba wygenerować wykres w trybie strategii „Wykonywanie kodu”.
        """
        if self._ai_images:
            if self.ai_file_var.get() not in [e.get("filename") or "" for e in self._ai_images]:
                self.ai_file_var.set(self._ai_images[-1].get("filename") or "Obraz")
            self._on_ai_file_change(self.ai_file_var.get())
        else:
            self._clear_container()
            no_ai = ctk.CTkLabel(
                self.plot_container,
                text="Brak wykresu AI\n\n"
                     "Aby wygenerować wykres AI, uruchom analizę w trybie strategii\n"
                     "„Wykonywanie kodu”. Wykres wygenerowany w Pythonie\n"
                     "pojawi się w tym miejscu.",
                font=FONTS["body"],
                text_color=COLORS["text_secondary"]
            )
            no_ai.pack(expand=True)

    def _update_ai_file_dropdown(self) -> None:
        """! @brief Uzupełnia listę rozwijaną nazwami plików obrazów z Code Interpreter."""
        if not self._ai_images:
            self.ai_file_menu.configure(values=["- brak -"])
            self.ai_file_var.set("- brak -")
            return
        names = [e.get("filename") or f"Obraz {i+1}" for i, e in enumerate(self._ai_images)]
        self.ai_file_menu.configure(values=names)
        if self.ai_file_var.get() not in names:
            self.ai_file_var.set(names[-1])

    def _on_ai_file_change(self, choice: str) -> None:
        """! @brief Wyświetla wybrany obraz AI z listy po zmianie selekcji."""
        if not choice or choice == "- brak -":
            return
        for entry in self._ai_images:
            fn = entry.get("filename") or ""
            if fn == choice:
                self._ai_image_data = entry.get("data")
                self._ai_image_title = fn
                self._display_ai_image()
                return
        
        if self.current_data is not None:
            self._generate_current()
    
    def _generate_current(self) -> None:
        """! @brief Rysuje wykres dla current_data i bieżącego typu/kolumn."""
        if self.current_data is not None:
            self.plot_dataframe(self.current_data, self.current_plot_type)
    
    def set_data(self, df: pd.DataFrame) -> None:
        """!
        Ustaw dane do wizualizacji.
        
        Args:
            df: DataFrame z danymi
        """
        self.current_data = df
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        self.data_info_label.configure(
            text=f"{len(df)} wierszy, {len(numeric_cols)} kolumn numerycznych"
        )
    
    def plot_dataframe(self, df: pd.DataFrame, plot_type: str = "auto"):
        """!
        Wyświetl wykres DataFrame.
        
        Args:
            df: DataFrame do wizualizacji
            plot_type: Typ wykresu ('auto', 'line', 'scatter', 'hist', 'box', 'bar', 'ai')
        """
        logger.info(f"Tworzenie wykresu: type={plot_type}")
        
        if plot_type != "ai" and self.plot_type_var.get() == "AI":
            self.current_data = df
            self.current_plot_type = plot_type if plot_type != "auto" else "line"
            return

        self.current_data = df
        self.current_plot_type = plot_type if plot_type != "auto" else "line"
        
        type_display = {
            "line": "Liniowy",
            "scatter": "Punktowy",
            "hist": "Histogram",
            "box": "Pudełkowy",
            "bar": "Słupkowy",
            "ai": "AI"
        }
        self.plot_type_var.set(type_display.get(self.current_plot_type, "Liniowy"))
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        self.data_info_label.configure(
            text=f"{len(df)} wierszy, {len(numeric_cols)} kolumn numerycznych"
        )
        
        if plot_type == "ai":
            self._update_ai_file_dropdown()
            self.ai_file_frame.pack(side="left", padx=5, pady=10)
            self._show_ai_view()
            return
        
        if self.canvas is None:
            self._clear_container()

        bg_color = COLORS["background"]
        text_color = COLORS["text"]
        
        fig = Figure(figsize=FIGURE_DEFAULT_SIZE, dpi=FIGURE_DPI, facecolor=bg_color)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['top'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        ax.spines['right'].set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        try:
            if plot_type == "auto" or plot_type == "line":
                for col in numeric_cols[:CHART_MAX_COLUMNS_LINE]:
                    ax.plot(df.index, df[col], marker='o', label=col, linewidth=2)
                ax.set_xlabel("Index")
                ax.set_ylabel("Wartość")
                ax.set_title("Wykres liniowy")
                if len(numeric_cols) > 0:
                    ax.legend(facecolor=COLORS["surface"], edgecolor=text_color, labelcolor=text_color)
                ax.grid(True, alpha=0.3, color=text_color)
                
            elif plot_type == "scatter":
                if len(numeric_cols) >= 2:
                    ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                             alpha=0.6, s=100, color='#8ba5d3')
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                    ax.set_title("Wykres punktowy (rozrzutu)")
                    ax.grid(True, alpha=0.3, color=text_color)
                else:
                    ax.text(0.5, 0.5, "Potrzeba min. 2 kolumn numerycznych", 
                           ha='center', va='center', color=text_color)
                    
            elif plot_type == "hist":
                if len(numeric_cols) > 0:
                    colors = ['#8ba5d3', '#f6a307', '#00D26A', '#FF4444', '#CE93D8']
                    for i, col in enumerate(numeric_cols[:CHART_MAX_COLUMNS_HIST]):
                        ax.hist(df[col].dropna(), bins=20, alpha=0.6, 
                               label=col, color=colors[i % len(colors)], edgecolor='white')
                    ax.set_xlabel("Wartość")
                    ax.set_ylabel("Częstość")
                    ax.set_title("Histogram")
                    ax.legend(facecolor='#3B3B3B', edgecolor=text_color, labelcolor=text_color)
                
            elif plot_type == "box":
                if len(numeric_cols) > 0:
                    bp = ax.boxplot([df[col].dropna() for col in numeric_cols[:CHART_MAX_COLUMNS_BOX]],
                              labels=[c[:CHART_LABEL_MAX_LEN] for c in numeric_cols[:CHART_MAX_COLUMNS_BOX]],
                              patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('#8ba5d3')
                    ax.set_ylabel("Wartość")
                    ax.set_title("Wykres pudełkowy")
                    ax.grid(True, alpha=0.3, color=text_color, axis='y')
                    
            elif plot_type == "bar":
                if len(numeric_cols) > 0:
                    means = df[numeric_cols[:CHART_MAX_COLUMNS_BAR]].mean()
                    bars = ax.bar(range(len(means)), means, color='#8ba5d3', alpha=0.8)
                    ax.set_xticks(range(len(means)))
                    ax.set_xticklabels([c[:CHART_LABEL_MAX_LEN] for c in means.index], rotation=45, ha='right')
                    ax.set_ylabel("Średnia wartość")
                    ax.set_title("Wykres słupkowy (średnie)")
                    ax.grid(True, alpha=0.3, color=text_color, axis='y')
            
            fig.tight_layout()
            
            self._display_figure(fig)
            
        except Exception as e:
            logger.error(f"Błąd tworzenia wykresu: {e}", exc_info=True)
            self._show_error(f"Nie udało się utworzyć wykresu: {str(e)}")
    
    def plot_custom(self, x_data, y_data, 
                   title: str = "Wykres",
                   xlabel: str = "X",
                   ylabel: str = "Y",
                   plot_type: str = "line"):
        """!
        Wyświetl niestandardowy wykres.
        
        Args:
            x_data: Dane osi X
            y_data: Dane osi Y (może być lista dla wielu serii)
            title: Tytuł wykresu
            xlabel: Etykieta osi X
            ylabel: Etykieta osi Y
            plot_type: Typ ('line', 'scatter', 'bar')
        """
        logger.info(f"Tworzenie niestandardowego wykresu: {title}")

        if self.canvas is None:
            self._clear_container()

        bg_color = COLORS["background"]
        text_color = COLORS["text"]
        
        fig = Figure(figsize=FIGURE_DEFAULT_SIZE, dpi=FIGURE_DPI, facecolor=bg_color)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg_color)
        
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        
        try:
            if plot_type == "line":
                if isinstance(y_data[0], (list, np.ndarray)):
                    for i, y in enumerate(y_data):
                        ax.plot(x_data, y, marker='o', label=f"Serie {i+1}", linewidth=2)
                    ax.legend(facecolor=COLORS["surface"], edgecolor=text_color, labelcolor=text_color)
                else:
                    ax.plot(x_data, y_data, marker='o', color='#8ba5d3', linewidth=2)
                    
            elif plot_type == "scatter":
                ax.scatter(x_data, y_data, alpha=0.6, s=100, color='#8ba5d3')
                
            elif plot_type == "bar":
                ax.bar(x_data, y_data, color='#8ba5d3', alpha=0.7)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, color=text_color)
            
            fig.tight_layout()
            self._display_figure(fig)
            
        except Exception as e:
            logger.error(f"Błąd tworzenia wykresu: {e}", exc_info=True)
            self._show_error(f"Błąd: {str(e)}")
    
    def _display_figure(self, fig: Figure) -> None:
        """!
        Wyświetl figure w kontenerze.
        
        Przy zamianie wykres→wykres: nowy canvas jest pakowany PRZED usunięciem
        starego, dzięki czemu Tkinter odświeża ekran tylko raz (brak migotania).
        
        Args:
            fig: Matplotlib Figure
        """
        old_canvas = self.canvas
        old_figure = self.current_figure

        self.current_figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas.draw()

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas_widget.configure(takefocus=0)

        if old_canvas is not None:
            try:
                old_canvas.get_tk_widget().destroy()
            except Exception:
                pass
        if old_figure is not None:
            try:
                plt.close(old_figure)
            except Exception:
                pass

        logger.debug("Wykres wyświetlony")
    
    def _clear_container(self) -> None:
        """! @brief Usuwa widgety matplotlib z kontenera wykresu."""
        for widget in self.plot_container.winfo_children():
            widget.destroy()
        
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None
        
        self.canvas = None
    
    def _show_error(self, message: str) -> None:
        """! @brief Wyświetla komunikat błędu zamiast wykresu."""
        self._clear_container()
        
        error_label = ctk.CTkLabel(
            self.plot_container,
            text=f"❌ {message}",
            font=FONTS["body"],
            text_color=COLORS["error"]
        )
        error_label.pack(expand=True)
    
    def set_ai_images(self, images: List[Dict[str, Any]]) -> None:
        """!
        Ustawia listę obrazów AI (np. z generated_images). Każdy element: {filename, data}.
        Po wywołaniu przy wyborze "AI" w dropdownie pojawi się lista tych plików.
        """
        self._ai_images = [
            {"filename": e.get("filename") or f"Obraz {i+1}", "data": e.get("data", b"")}
            for i, e in enumerate(images)
            if e.get("data")
        ]
        if self.current_plot_type == "ai" and self._ai_images:
            self._update_ai_file_dropdown()
            self.ai_file_var.set(self._ai_images[-1].get("filename") or "Obraz")
            self._on_ai_file_change(self.ai_file_var.get())

    def display_image_bytes(
        self,
        image_data: bytes,
        title: str = "Wykres wygenerowany przez AI"
    ):
        """!
        @brief Wyświetla obraz (np. z Code Interpreter) w panelu wizualizacji.

        @details Zapisuje dane obrazu, aby użytkownik mógł wrócić do niego
                 wybierając opcję "AI" z dropdown. Jeśli wcześniej
                 wywołano set_ai_images(), w dropdownie AI można wybrać który plik pokazać.

        @param image_data Surowe bajty obrazu (PNG, JPEG, …).
        @param title      Tytuł wyświetlany nad obrazem.
        """
        self._ai_image_data = image_data
        self._ai_image_title = title
        if not self._ai_images:
            self._ai_images = [{"filename": title, "data": image_data}]

        self.current_plot_type = "ai"
        self.plot_type_var.set("AI")
        self._update_ai_file_dropdown()
        self.ai_file_frame.pack(side="left", padx=5, pady=10)
        if self._ai_images and self.ai_file_var.get() != title:
            self.ai_file_var.set(self._ai_images[-1].get("filename") or title)
        self._display_ai_image()

    def _display_ai_image(self) -> None:
        """!
        @brief Renderuje zapisany obraz AI w panelu wizualizacji.

        @details Wywoływana zarówno z display_image_bytes (przy nowym obrazie)
                 jak i z _on_plot_type_change (przy wyborze opcji AI z dropdown).
        """
        import io
        from PIL import Image

        if self._ai_image_data is None:
            return

        try:
            img = Image.open(io.BytesIO(self._ai_image_data))
            width, height = img.size
            aspect = width / max(height, 1)

            if self.canvas is None:
                self._clear_container()

            bg_color = COLORS["background"]
            text_color = COLORS["text"]

            fig_width = min(FIGURE_ASPECT_WIDTH_MAX, max(FIGURE_ASPECT_WIDTH_MIN, aspect * FIGURE_ASPECT_FACTOR))
            fig_height = fig_width / aspect

            fig = Figure(
                figsize=(fig_width, fig_height),
                dpi=FIGURE_DPI,
                facecolor=bg_color,
            )
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            if self._ai_image_title:
                ax.set_title(
                    self._ai_image_title,
                    color=text_color,
                    fontsize=14,
                    pad=10,
                )
            fig.tight_layout(pad=0.5)

            self._display_figure(fig)

            self.data_info_label.configure(
                text=f"Obraz AI: {width}×{height}"
            )
            logger.info(
                f"Wyświetlono obraz AI: {self._ai_image_title} ({width}×{height})"
            )

        except Exception as e:
            logger.error(f"Błąd wyświetlania obrazu AI: {e}", exc_info=True)
            self._show_error(f"Nie udało się wyświetlić obrazu: {str(e)}")

    def clear_plot(self) -> None:
        """! @brief Czyści wykres i przywraca widok placeholdera."""
        self._clear_container()
        self._show_placeholder()
        self.current_data = None
        self._ai_image_data = None
        self._ai_image_title = "Wykres wygenerowany przez AI"
        self._ai_images = []
        self.ai_file_frame.pack_forget()
        self.ai_file_menu.configure(values=["- brak -"])
        self.ai_file_var.set("- brak -")
        self.data_info_label.configure(text="Brak danych")
        logger.info("Wyczyszczono wykres")
    
    def refresh_theme(self) -> None:
        """! @brief Aktualizuje kolory matplotlib po przełączeniu motywu aplikacji."""
        self.configure(fg_color=COLORS["surface"])
        self.plot_container.configure(fg_color=COLORS["background"])
        
        if self.current_data is not None:
            self.plot_dataframe(self.current_data, self.current_plot_type)
    
    def _export_plot(self) -> None:
        """! @brief Zapisuje bieżący wykres do pliku obrazu (dialog użytkownika)."""
        if self.current_figure is None:
            logger.warning("Brak wykresu do eksportu")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"wykres_{timestamp}.png"
        
        filepath = filedialog.asksaveasfilename(
            title="Eksportuj wykres",
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("JPEG", "*.jpg"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            self.current_figure.savefig(
                filepath,
                dpi=FIGURE_EXPORT_DPI,
                bbox_inches='tight',
                facecolor=self.current_figure.get_facecolor(),
                edgecolor='none'
            )
            
            logger.info(f"Wyeksportowano wykres do: {filepath}")
            
        except Exception as e:
            logger.error(f"Błąd eksportu wykresu: {e}", exc_info=True)
