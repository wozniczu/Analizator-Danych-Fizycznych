##
## @file math_renderer.py
## @brief Renderowanie wzorów matematycznych LaTeX do obrazów.
##
import re
import io
from typing import Any, List, Tuple, Union, cast
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import mathtext
from utils.logger import logger


class MathRenderer:
    r"""!
    @brief Klasa do renderowania wzorów matematycznych LaTeX.
    
    @details Wykrywa wzory LaTeX w tekście i renderuje je jako obrazy PNG.
             Obsługuje wzory inline \( ... \) oraz display \[ ... \].
    """
    
    INLINE_PAREN_PATTERN = re.compile(r'\\\((.*?)\\\)', re.DOTALL)
    INLINE_DOLLAR_PATTERN = re.compile(
        r'(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)(?<!\$)\$(?!\$)',
        re.DOTALL
    )
    DISPLAY_BRACKET_PATTERN = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
    DISPLAY_DOLLAR_PATTERN = re.compile(r'(?<!\\)\$\$(.+?)(?<!\\)\$\$', re.DOTALL)
    TABLE_ENV_PATTERN = re.compile(r'\\begin\{(array|tabular)\}', re.DOTALL)
    TEXT_COMMAND_PATTERN = re.compile(r'\\{1,2}(?:text|mathrm)\{([^{}]*)\}')
    BOLD_TEXT_PATTERN = re.compile(r'\\textbf\{([^{}]*)\}')
    PLAIN_TABLE_CELL_PATTERN = re.compile(r'^[\w.\-+]+$')
    
    def __init__(self, dpi: int = 150) -> None:
        """!
        @brief Konstruktor renderera wzorów.
        
        @param dpi Rozdzielczość renderowanych obrazów (domyślnie 150).
        """
        self.dpi = dpi
        self.parser = mathtext.MathTextParser("path")
    
    def detect_math(self, text: str) -> List[Tuple[str, str, int, int]]:
        """!
        @brief Wykrywa wzory matematyczne w tekście.
        
        @param text Tekst do przeszukania.
        @return Lista krotek (typ, wzór, start, koniec), gdzie:
                - typ: 'inline' lub 'display'
                - wzór: treść wzoru bez znaczników
                - start: pozycja początkowa w tekście
                - koniec: pozycja końcowa w tekście
        """
        matches = []
        
        for pattern in (self.DISPLAY_BRACKET_PATTERN, self.DISPLAY_DOLLAR_PATTERN):
            for match in pattern.finditer(text):
                matches.append(('display', match.group(1), match.start(), match.end()))
        
        for pattern in (self.INLINE_PAREN_PATTERN, self.INLINE_DOLLAR_PATTERN):
            for match in pattern.finditer(text):
                overlap = False
                for _, _, start, end in matches:
                    if not (match.end() <= start or match.start() >= end):
                        overlap = True
                        break
                if not overlap:
                    matches.append(('inline', match.group(1), match.start(), match.end()))
        
        matches.sort(key=lambda x: x[2])
        return matches
    
    @staticmethod
    def _split_latex_row(row_text: str) -> List[str]:
        """! @brief Dzieli wiersz tabeli LaTeX po separatorze '&' z pominięciem znaków escapowanych."""
        return [cell.strip() for cell in re.split(r'(?<!\\)&', row_text)]

    def _parse_latex_table(self, formula: str) -> Tuple[str, str, List[List[str]]]:
        """!
        @brief Parsuje środowisko array/tabular do postaci (env, alignment, rows).

        @param formula Fragment LaTeX ze środowiskiem array/tabular.
        @return Krotka (nazwa środowiska, wyrównanie kolumn, wiersze komórek).
        """
        match = re.search(
            r'\\begin\{(array|tabular)\}\{([^}]*)\}(.*?)\\end\{\1\}',
            formula,
            re.DOTALL
        )
        if not match:
            raise ValueError("Nie rozpoznano środowiska tabeli LaTeX.")

        env_name = match.group(1)
        alignment = match.group(2)
        body = match.group(3)

        raw_rows = re.split(r'(?<!\\)\\\\', body)
        rows: List[List[str]] = []

        for raw_row in raw_rows:
            cleaned_row = re.sub(r'\\hline|\\cline\{[^}]+\}', '', raw_row).strip()
            if not cleaned_row:
                continue
            cells = self._split_latex_row(cleaned_row)
            if any(cell for cell in cells):
                rows.append(cells)

        if not rows:
            raise ValueError("Tabela LaTeX nie zawiera danych do renderowania.")

        max_cols = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
        return env_name, alignment, normalized_rows

    @staticmethod
    def _sanitize_table_cell(cell: str) -> str:
        """! @brief Czyści prostą zawartość komórki tabeli przed renderowaniem."""
        cleaned = cell.strip()
        while True:
            updated = MathRenderer.TEXT_COMMAND_PATTERN.sub(r'\1', cleaned)
            if updated == cleaned:
                break
            cleaned = updated
        while True:
            updated = MathRenderer.BOLD_TEXT_PATTERN.sub(r'\\mathbf{\1}', cleaned)
            if updated == cleaned:
                break
            cleaned = updated
        cleaned = cleaned.replace(r'\_', '_')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    @staticmethod
    def _cell_needs_math_mode(cell: str) -> bool:
        """! @brief Określa, czy komórka ma być renderowana jako mathtext ($...$)."""
        if not cell:
            return False
        if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', cell):
            return False
        if MathRenderer.PLAIN_TABLE_CELL_PATTERN.fullmatch(cell):
            return False
        return any(token in cell for token in ("^", "_", r"\frac", r"\sqrt", "{", "}", "="))

    @staticmethod
    def _fit_table_image(image: Image.Image, max_width_px: int = 1400) -> Image.Image:
        """! @brief Zmniejsza obraz tabeli tylko gdy jest ekstremalnie szeroki (>1400px)."""
        if image.width <= max_width_px:
            return image
        scale = max_width_px / float(image.width)
        new_size = (max_width_px, max(1, int(image.height * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _render_latex_table(self, formula: str, display_mode: bool = False) -> Image.Image:
        """!
        @brief Renderuje środowisko array/tabular jako tabelę matplotlib.

        @param formula Fragment LaTeX tabeli.
        @param display_mode Tryb wyświetlania (wpływa na skalowanie).
        @return Obraz PIL z wyrenderowaną tabelą.
        """
        env_name, alignment, rows = self._parse_latex_table(formula)
        row_count = len(rows)
        col_count = max(len(row) for row in rows)

        font_size = 8 if display_mode else 7

        alignments = [char for char in alignment if char in ("l", "c", "r")]
        if len(alignments) < col_count:
            alignments.extend(["c"] * (col_count - len(alignments)))

        cell_texts: list[list[str]] = []
        for row in rows:
            cell_row = []
            for col_idx in range(col_count):
                raw_cell = row[col_idx] if col_idx < len(row) else ""
                cell_text = self._sanitize_table_cell(raw_cell)
                if env_name == "array" and self._cell_needs_math_mode(cell_text):
                    cell_row.append(f"${cell_text}$")
                else:
                    cell_row.append(cell_text)
            cell_texts.append(cell_row)

        col_max_lens = [
            max(3, max(len(cell_texts[r][c]) for r in range(row_count)))
            for c in range(col_count)
        ]
        total_chars = max(1, sum(col_max_lens))
        col_widths_norm = [cl / total_chars for cl in col_max_lens]

        row_h_inch = font_size / self.dpi * 3.0
        fig_height = max(0.3, row_count * row_h_inch)

        char_w_inch = font_size / self.dpi * 1.4
        fig_width = max(1.5, sum(col_max_lens) * char_w_inch + col_count * 0.05)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=self.dpi)
        fig.patch.set_facecolor('none')
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])

        table = ax.table(
            cellText=cell_texts,
            colWidths=col_widths_norm,
            bbox=cast(Any, [0, 0, 1, 1])
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)

        loc_map = {"l": "left", "c": "center", "r": "right"}
        for row_idx in range(row_count):
            for col_idx in range(col_count):
                table_cell = table[(row_idx, col_idx)]
                table_cell.set_facecolor((0, 0, 0, 0))
                table_cell.set_edgecolor('white')
                table_cell.set_linewidth(0.5)
                table_cell.PAD = 0.04
                loc = loc_map.get(alignments[col_idx], "center")
                setattr(table_cell, "_loc", loc)
                text_obj = table_cell.get_text()
                text_obj.set_color('white')
                text_obj.set_fontsize(font_size)

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            dpi=self.dpi,
            bbox_inches='tight',
            pad_inches=0.01,
            transparent=True
        )
        plt.close(fig)

        buf.seek(0)
        image = Image.open(buf).copy()
        buf.close()
        return self._fit_table_image(image)

    def render_math(self, formula: str, display_mode: bool = False) -> Image.Image:
        r"""!
        @brief Renderuje wzór matematyczny do obrazu PIL.
        
        @param formula Wzór LaTeX (bez znaczników \( \) lub \[ \]).
        @param display_mode True dla wzorów w trybie display (większe).
        @return Obiekt PIL.Image z wyrenderowanym wzorem.
        """
        try:
            formula = formula.strip()

            if self.TABLE_ENV_PATTERN.search(formula):
                return self._render_latex_table(formula, display_mode)
            
            fontsize = 11 if display_mode else 9
            
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.patch.set_facecolor('none')
            
            text = fig.text(
                0, 0, f'${formula}$',
                fontsize=fontsize,
                color='white',
                ha='left',
                va='baseline'
            )
            
            fig.canvas.draw()
            bbox = text.get_window_extent(fig.canvas.get_renderer())  # type: ignore
            
            width = bbox.width / self.dpi
            height = bbox.height / self.dpi
            fig.set_size_inches(width + 0.2, height + 0.2)
            
            text.set_position((0.1, 0.5))
            
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format='png',
                dpi=self.dpi,
                bbox_inches='tight',
                pad_inches=0.05,
                transparent=True
            )
            plt.close(fig)
            
            buf.seek(0)
            image = Image.open(buf).copy()
            buf.close()
            
            logger.debug(f"Wyrenderowano wzór: {formula[:50]}...")
            return image
            
        except Exception as e:
            logger.error(f"Błąd podczas renderowania wzoru '{formula}': {e}")
            return Image.new('RGBA', (100, 20), (0, 0, 0, 0))
    
    def split_text_with_math(self, text: str) -> List[Tuple[str, Union[str, Image.Image]]]:
        """!
        @brief Dzieli tekst na fragmenty tekstowe i wyrenderowane wzory.
        
        @param text Tekst zawierający wzory LaTeX.
        @return Lista krotek (typ, zawartość), gdzie:
                - typ: 'text' lub 'math'
                - zawartość: tekst (str) lub obraz (PIL.Image)
        """
        math_matches = self.detect_math(text)
        
        if not math_matches:
            return [('text', text)]
        
        result: List[Tuple[str, Union[str, Image.Image]]] = []
        last_end = 0
        
        for math_type, formula, start, end in math_matches:
            if start > last_end:
                result.append(('text', text[last_end:start]))
            
            display_mode = (math_type == 'display')
            image = self.render_math(formula, display_mode)
            result.append(('math', image))
            
            last_end = end
        
        if last_end < len(text):
            result.append(('text', text[last_end:]))
        
        return result


_renderer_instance = None


def get_math_renderer() -> MathRenderer:
    """!
    @brief Zwraca singleton instancję MathRenderer.

    @return Globalny obiekt MathRenderer.
    """
    global _renderer_instance
    if _renderer_instance is None:
        _renderer_instance = MathRenderer()
    return _renderer_instance