##
## @file chat_panel.py
## @brief Panel czatu z asystentem AI.
##
## @details Moduł zawiera klasę ChatPanel odpowiedzialną za wyświetlanie
##          historii konwersacji oraz obsługę wprowadzania wiadomości
##          przez użytkownika. Obsługuje formatowanie LaTeX i Markdown
##          (pogrubienie, kursywa, nagłówki).
##
import customtkinter as ctk
from typing import Any, Callable, List, Optional, Tuple
import math
from tkinter import font as tkfont
from PIL import Image, ImageTk
from gui.styles import COLORS, FONTS, DIMENSIONS
from config.settings import MAX_VISIBLE_MESSAGES
from utils.logger import logger
from utils.math_renderer import get_math_renderer


class ChatPanel(ctk.CTkFrame):
    """!
    @brief Panel czatu z historią konwersacji.
    
    @details Klasa ChatPanel zapewnia interfejs do komunikacji z AI:
             - Wyświetla historię wiadomości (użytkownik, asystent, system)
             - Obsługuje wprowadzanie tekstu przez użytkownika
             - Pokazuje wskaźnik pisania podczas oczekiwania na odpowiedź
             - Obsługuje skróty klawiszowe (Enter, Shift+Enter)
    
    @see MainWindow
    """
    
    def __init__(self, parent: Any, on_send_message: Callable[[str], None]) -> None:
        """!
        @brief Konstruktor panelu czatu.
        
        @param parent Widget rodzica (CTkFrame lub CTkTabview).
        @param on_send_message Callback wywoływany po wysłaniu wiadomości.
        """
        super().__init__(
            parent,
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        
        self.on_send_message = on_send_message
        self.max_visible_messages = MAX_VISIBLE_MESSAGES
        self.typing_indicator = None
        self.typing_animation_id = None
        self.typing_dots = 0
        self._cached_body_font = None
        self._cached_scaling = None
        self._fit_bubble_after_id = None

        self._create_widgets()

    @staticmethod
    def _parse_markdown_segments(text: str) -> list[tuple[str, tuple[str, ...] | None]]:
        """!
        @brief Dzieli tekst na segmenty z tagami Markdown (**pogrubienie**, *kursywa*, nagłówki #).

        @param text Tekst wejściowy.
        @return Lista par (fragment, tagi lub None).
        """
        if not text:
            return []
        segments: list[tuple[str, tuple[str, ...] | None]] = []
        lines = text.split("\n")
        for line_idx, line in enumerate(lines):
            if line.startswith("### "):
                seg_text = line[4:] + ("\n" if line_idx < len(lines) - 1 else "")
                segments.append((seg_text, ("h3",)))
                continue
            if line.startswith("## "):
                seg_text = line[3:] + ("\n" if line_idx < len(lines) - 1 else "")
                segments.append((seg_text, ("h2",)))
                continue
            if line.startswith("# "):
                seg_text = line[2:] + ("\n" if line_idx < len(lines) - 1 else "")
                segments.append((seg_text, ("h1",)))
                continue
            i = 0
            while i < len(line):
                if i + 2 <= len(line) and line[i : i + 2] == "**":
                    end = line.find("**", i + 2)
                    if end == -1:
                        segments.append((line[i:], None))
                        break
                    segments.append((line[i + 2 : end], ("bold",)))
                    i = end + 2
                elif line[i] == "*":
                    if i + 1 < len(line) and line[i + 1] == "*":
                        i += 1
                        continue
                    end = i + 1
                    end_found = False
                    while end < len(line):
                        end = line.find("*", end)
                        if end == -1:
                            break
                        if end + 1 >= len(line) or line[end + 1] != "*":
                            end_found = True
                            break
                        end += 2
                    if not end_found or end == -1:
                        segments.append((line[i:], None))
                        break
                    segments.append((line[i + 1 : end], ("italic",)))
                    i = end + 1
                else:
                    next_bold = line.find("**", i)
                    next_italic = line.find("*", i)
                    while next_italic != -1 and next_italic + 1 < len(line) and line[next_italic + 1] == "*":
                        next_italic = line.find("*", next_italic + 2)
                    candidates = []
                    if next_bold != -1:
                        candidates.append(next_bold)
                    if next_italic != -1:
                        candidates.append(next_italic)
                    if not candidates:
                        segments.append((line[i:], None))
                        break
                    j = min(candidates)
                    segments.append((line[i:j], None))
                    i = j
            if line_idx < len(lines) - 1:
                segments.append(("\n", None))
        return segments

    def _ensure_markdown_tags(self, textbox: ctk.CTkTextbox) -> None:
        """! @brief Konfiguruje tagi Markdown (bold, italic, h1–h3) na wewnętrznym Tk Text."""
        txt = textbox._textbox
        if getattr(txt, "_markdown_tags_done", False):
            return
        try:
            font_spec = txt.cget("font")
            font_obj = tkfont.Font(font=font_spec)
            actual = font_obj.actual()
            family = actual["family"]
            size = actual["size"]
        except Exception:
            family, size = FONTS["body"][0], FONTS["body"][1]
        txt.tag_configure("bold", font=(family, size, "bold"))
        txt.tag_configure("italic", font=(family, size, "italic"))
        txt.tag_configure("h1", font=(family, size + 4, "bold"))
        txt.tag_configure("h2", font=(family, size + 2, "bold"))
        txt.tag_configure("h3", font=(family, size, "bold"))
        txt._markdown_tags_done = True

    def _insert_text_with_markdown(self, textbox: ctk.CTkTextbox, text_segment: str) -> None:
        """! @brief Wstawia fragment tekstu z formatowaniem Markdown (tagi Tk Text)."""
        self._ensure_markdown_tags(textbox)
        txt = textbox._textbox
        for part, tags in self._parse_markdown_segments(text_segment):
            if not part and not tags:
                continue
            if tags:
                txt.insert("end", part, *tags)
            else:
                txt.insert("end", part)

    def _compute_wraplength(self) -> int:
        """! @brief Zwraca szerokość zawijania treści w pikselach w zależności od panelu."""
        width = self.messages_frame.winfo_width()
        if width <= 1:
            return 780
        return max(320, width - 120)

    def _get_body_font_and_scaling(self) -> tuple:
        """!
        @brief Zwraca efektywną czcionkę treści CTkTextbox i współczynnik skalowania CTk.

        @details Wynik jest buforowany; czcionka pochodzi z tymczasowego CTkTextbox (DPI/skala).
        """
        if self._cached_body_font is not None:
            return self._cached_body_font, self._cached_scaling

        self._cached_scaling = getattr(self, '_widget_scaling', 1.0)

        try:
            temp = ctk.CTkTextbox(self, font=FONTS["body"])
            font_name = temp._textbox.cget("font")
            self._cached_body_font = tkfont.Font(font=font_name)
            temp.destroy()
        except Exception:
            effective_size = max(1, int(round(
                FONTS["body"][1] / self._cached_scaling
            )))
            self._cached_body_font = tkfont.Font(
                family=FONTS["body"][0],
                size=effective_size,
                weight=(FONTS["body"][2]
                        if len(FONTS["body"]) > 2 else "normal")
            )

        return self._cached_body_font, self._cached_scaling

    def _estimate_bubble_size(self, message: str, force_max_width: bool = False) -> tuple[int, int]:
        """!
        @brief Szacuje szerokość i wysokość bąbelka wiadomości (piksele CTk).

        @param message Treść wiadomości.
        @param force_max_width Jeśli True, szerokość jest ustawiana na maksimum (np. komunikaty systemowe).
        @return Para (szerokość, wysokość) w jednostkach CTk.
        """
        max_text_width = self._compute_wraplength()
        body_font, scaling = self._get_body_font_and_scaling()

        logical_lines = message.splitlines() or [""]
        non_empty_lines = [line for line in logical_lines if line]
        longest_line_width = max(
            (body_font.measure(line) for line in non_empty_lines),
            default=0,
        )

        min_text_width = 80
        if force_max_width:
            text_width = max_text_width
        else:
            text_width = max(min_text_width, min(max_text_width, longest_line_width))
        wrapped_lines = 0

        for line in logical_lines:
            if not line:
                wrapped_lines += 1
                continue
            pixel_width = body_font.measure(line)
            wrapped_lines += max(1, math.ceil(pixel_width / text_width))

        line_height = body_font.metrics("linespace")
        horizontal_padding = 22
        ctk_vertical_overhead = 24

        bubble_width_px = text_width + horizontal_padding

        text_height_ctk = wrapped_lines * line_height / scaling
        min_text_ctk = line_height / scaling
        bubble_height = int(text_height_ctk + ctk_vertical_overhead)
        min_height = int(min_text_ctk + ctk_vertical_overhead)

        bubble_width = int(bubble_width_px / scaling)
        bubble_height = max(min_height, bubble_height)

        return bubble_width, bubble_height

    def _fit_bubble_height_to_content(self, bubble: ctk.CTkTextbox, content: str, defer: bool = True) -> None:
        """!
        @brief Ustawia wysokość bąbelka tak, by treść była widoczna bez przewijania.

        @param bubble Widget tekstowy wiadomości.
        @param content Treść do zmierzenia.
        @param defer False podczas streamingu (mniej artefaktów); True dla zwykłych wiadomości.
        """
        if self._fit_bubble_after_id is not None:
            self.after_cancel(self._fit_bubble_after_id)
            self._fit_bubble_after_id = None

        def _do_fit():
            self._fit_bubble_after_id = None
            try:
                txt = bubble._textbox
                self.messages_frame.update_idletasks()
                bubble.update_idletasks()
                res = txt.count("1.0", "end", "ypixels")
                content_height_px = (res[0] if res and res[0] is not None else 0) or 0
                _, scaling = self._get_body_font_and_scaling()
                ctk_vertical_overhead = 24
                if content_height_px > 0:
                    new_height = int((content_height_px + ctk_vertical_overhead) / scaling)
                else:
                    _, h = self._estimate_bubble_size(content)
                    new_height = int(h * 1.08) if h else 80
                new_height = max(40, new_height)
                bubble.configure(height=new_height)
            except Exception:
                _, h = self._estimate_bubble_size(content)
                bubble.configure(height=max(40, int(h * 1.08) if h else 80))
        self.messages_frame.update_idletasks()
        bubble.update_idletasks()
        _do_fit()
        if defer:
            self._fit_bubble_after_id = self.after_idle(_do_fit)

    def _insert_message_with_math(self, textbox: ctk.CTkTextbox, message: str) -> None:
        r"""!
        @brief Wstawia wiadomość z obsługą wzorów matematycznych LaTeX.
        
        @param textbox Widget CTkTextbox do wstawienia treści.
        @param message Treść wiadomości (może zawierać wzory LaTeX).
        
        @details Wykrywa wzory LaTeX w formacie \( ... \) i \[ ... \],
                 renderuje je jako obrazy i wstawia do textbox'a.
        """
        try:
            renderer = get_math_renderer()
            segments = renderer.split_text_with_math(message)
            
            if not hasattr(textbox, '_math_images'):
                textbox._math_images = []
            
            for segment_type, content in segments:
                if segment_type == 'text':
                    self._insert_text_with_markdown(textbox, content)
                elif segment_type == 'math':
                    photo = ImageTk.PhotoImage(content)
                    textbox._math_images.append(photo)
                    
                    textbox._textbox.image_create("end", image=photo)
        
        except Exception as e:
            logger.error(f"Błąd podczas wstawiania wzorów matematycznych: {e}")
            textbox.insert("1.0", message)
    
    def _trim_message_history(self) -> None:
        """! @brief Ogranicza liczbę widocznych wiadomości (usuwa najstarsze ramki)."""
        message_frames = [
            child for child in self.messages_frame.winfo_children()
            if getattr(child, "_is_chat_message", False)
        ]

        overflow = len(message_frames) - self.max_visible_messages
        if overflow <= 0:
            return

        for old_frame in message_frames[:overflow]:
            old_frame.destroy()

        logger.debug(f"Przycięto historię czatu: usunięto {overflow} najstarszych wiadomości")

    def _scroll_chat_to_end(self) -> None:
        """! @brief Przewija scrollable frame wiadomości na koniec (najnowsza treść)."""
        self.update_idletasks()
        canvas = self.messages_frame._parent_canvas
        canvas.yview_moveto(1.0)

    def _scroll_bubble_text_viewport(
        self,
        bubble: ctk.CTkTextbox,
        *,
        x_at_start: bool = True,
        y_at_start: bool = True,
    ) -> None:
        """! @brief Ustawia przewijanie wewnątrz bąbelka (Tk Text): lewo/prawo, góra/dół."""
        try:
            if not bubble.winfo_exists():
                return
            txt = bubble._textbox
            txt.xview_moveto(0.0 if x_at_start else 1.0)
            txt.yview_moveto(0.0 if y_at_start else 1.0)
        except Exception:
            pass

    def _create_widgets(self) -> None:
        """! @brief Tworzy nagłówek, obszar wiadomości i pole wprowadzania."""
        header = ctk.CTkLabel(
            self,
            text="Czat",
            font=FONTS["heading"],
            anchor="w"
        )
        header.pack(pady=15, padx=20, fill="x")
        
        self.messages_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=COLORS["background"],
            corner_radius=DIMENSIONS["corner_radius"]
        )
        self.messages_frame.pack(
            pady=(0, 15),
            padx=20,
            fill="both",
            expand=True
        )
        
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(pady=(0, 15), padx=20, fill="x")
        
        self.input_field = ctk.CTkTextbox(
            input_frame,
            height=DIMENSIONS["input_height"],
            corner_radius=DIMENSIONS["corner_radius"],
            font=FONTS["body"]
        )
        self.input_field.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.input_field.bind("<Return>", self._on_enter_press)
        self.input_field.bind("<Shift-Return>", lambda e: None)
        
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Wyślij",
            command=self._send_message,
            width=100,
            height=DIMENSIONS["input_height"],
            corner_radius=DIMENSIONS["corner_radius"],
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_dark"],
            text_color_disabled=COLORS["text_disabled"]
        )
        self.send_btn.pack(side="right")
    
    def _on_enter_press(self, event: Any) -> None:
        """! @brief Obsługuje klawisz Enter (wysyłka bez Shift)."""
        if not event.state & 0x0001:
            self._send_message()
            return "break"
    
    def _send_message(self) -> None:
        """! @brief Czyta pole wejścia i wywołuje callback on_send_message."""
        message = self.input_field.get("1.0", "end-1c").strip()
        
        if message:
            self.on_send_message(message)
            inner = self.input_field._textbox
            prev_state = inner.cget("state")
            self.input_field.configure(state="normal")
            self.input_field.delete("1.0", "end")
            self.input_field.configure(state=prev_state)
    
    def add_user_message(self, message: str) -> None:
        """!
        @brief Dodaje wiadomość użytkownika do czatu.
        
        @param message Treść wiadomości.
        """
        self._add_message(message, "user")
    
    def add_assistant_message(self, message: str, model: str = "") -> None:
        """!
        @brief Dodaje odpowiedź asystenta AI.
        
        @param message Treść odpowiedzi.
        @param model Nazwa użytego modelu (opcjonalnie).
        """
        self._add_message(message, "assistant", model)
    
    def add_system_message(self, message: str) -> None:
        """!
        @brief Dodaje wiadomość systemową.
        
        @param message Treść wiadomości systemowej.
        """
        self._add_message(message, "system")

    def add_show_code_button(
        self,
        generated_code: str,
        execution_output: Optional[str] = None,
    ):
        """!
        @brief Dodaje przycisk „Wyświetl wygenerowany kod” pod ostatnią wiadomością.

        @param generated_code Kod źródłowy (opcjonalny).
        @param execution_output Wynik wykonania (opcjonalny).
        @details Po kliknięciu przełącza widoczność bloku z kodem i wynikiem.
        """
        if not generated_code and not execution_output:
            return
        msg_frame = ctk.CTkFrame(
            self.messages_frame,
            fg_color="transparent",
        )
        msg_frame._is_chat_message = True
        msg_frame.pack(pady=2, padx=10, fill="x")

        content_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
        is_visible = [False]

        def toggle():
            if is_visible[0]:
                content_frame.pack_forget()
                btn.configure(text="Wyświetl wygenerowany kod")
                is_visible[0] = False
            else:
                content_frame.pack(fill="x", pady=(5, 0), padx=5)
                btn.configure(text="Ukryj wygenerowany kod")
                is_visible[0] = True
            self._scroll_chat_to_end()

        btn = ctk.CTkButton(
            msg_frame,
            text="Wyświetl wygenerowany kod",
            command=toggle,
            width=220,
            height=32,
            fg_color=COLORS["primary_dark"],
            hover_color=COLORS["primary"],
            corner_radius=DIMENSIONS["corner_radius"],
            font=FONTS["body"],
            text_color_disabled=COLORS["text_disabled"],
        )
        btn.pack(anchor="w", padx=5, pady=2)

        if generated_code:
            code_label = ctk.CTkLabel(
                content_frame,
                text="Wygenerowany kod:",
                font=("Arial", 12, "bold"),
                anchor="w",
                text_color=COLORS["text_secondary"],
            )
            code_label.pack(anchor="w", pady=(8, 2))
            code_box = ctk.CTkTextbox(
                content_frame,
                fg_color=COLORS["primary_dark"],
                text_color="#FFFFFF",
                corner_radius=8,
                font=("Consolas", 11),
                wrap="word",
                height=min(300, max(120, generated_code.count("\n") * 18 + 40)),
            )
            code_box.pack(fill="x", pady=(0, 8))
            code_box.insert("1.0", generated_code)
            code_box.configure(state="disabled")

        if execution_output:
            out_label = ctk.CTkLabel(
                content_frame,
                text="Wynik wykonania:",
                font=("Arial", 12, "bold"),
                anchor="w",
                text_color=COLORS["text_secondary"],
            )
            out_label.pack(anchor="w", pady=(4, 2))
            out_box = ctk.CTkTextbox(
                content_frame,
                fg_color=COLORS["primary_dark"],
                text_color="#FFFFFF",
                corner_radius=8,
                font=("Consolas", 11),
                wrap="word",
                height=min(250, max(80, execution_output.count("\n") * 18 + 40)),
            )
            out_box.pack(fill="x", pady=(0, 8))
            out_box.insert("1.0", execution_output)
            out_box.configure(state="disabled")

        self._trim_message_history()
        self._scroll_chat_to_end()

    def add_show_reasoning_button(self, reasoning_text: str) -> None:
        """!
        @brief Dodaje przycisk „Pokaż proces rozumowania” (np. deepseek-reasoner).

        @param reasoning_text Treść rozumowania (chain-of-thought).
        @details Po kliknięciu przełącza widoczność bloku z reasoning_content.
        """
        reasoning_text = (reasoning_text or "").strip()
        if not reasoning_text:
            return
        msg_frame = ctk.CTkFrame(
            self.messages_frame,
            fg_color="transparent",
        )
        msg_frame._is_chat_message = True
        msg_frame.pack(pady=2, padx=10, fill="x")

        content_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
        is_visible = [False]

        def toggle():
            if is_visible[0]:
                content_frame.pack_forget()
                btn.configure(text="Pokaż proces rozumowania")
                is_visible[0] = False
            else:
                content_frame.pack(fill="x", pady=(5, 0), padx=5)
                btn.configure(text="Ukryj proces rozumowania")
                is_visible[0] = True
            self._scroll_chat_to_end()

        btn = ctk.CTkButton(
            msg_frame,
            text="Pokaż proces rozumowania",
            command=toggle,
            width=220,
            height=32,
            fg_color=COLORS["primary_dark"],
            hover_color=COLORS["primary"],
            corner_radius=DIMENSIONS["corner_radius"],
            font=FONTS["body"],
            text_color_disabled=COLORS["text_disabled"],
        )
        btn.pack(anchor="w", padx=5, pady=2)

        label = ctk.CTkLabel(
            content_frame,
            text="Proces rozumowania modelu (chain of thought):",
            font=("Arial", 12, "bold"),
            anchor="w",
            text_color=COLORS["text_secondary"],
        )
        label.pack(anchor="w", pady=(8, 2))
        reasoning_box = ctk.CTkTextbox(
            content_frame,
            fg_color=COLORS["primary_dark"],
            text_color="#E0E0E0",
            corner_radius=8,
            font=("Arial", 11),
            wrap="word",
            height=min(400, max(120, reasoning_text.count("\n") * 18 + 40)),
        )
        reasoning_box.pack(fill="x", pady=(0, 8))
        reasoning_box.insert("1.0", reasoning_text)
        reasoning_box.configure(state="disabled")

        self._trim_message_history()
        self._scroll_chat_to_end()

    def add_hybrid_detail_buttons_row(
        self,
        plan_text: str,
        numerical_results_text: str,
        reasoning_text: str = "",
    ) -> None:
        """!
        @brief Jedna linia przycisków: plan hybrydy, wyniki lokalne, opcjonalnie rozumowanie.

        @param plan_text Tekst planu analizy (etap 1).
        @param numerical_results_text Sformatowane wyniki obliczeń lokalnych (etap 2).
        @param reasoning_text Opcjonalne reasoning_content ze streamu / API.
        """
        plan_text = (plan_text or "").strip()
        numerical_results_text = (numerical_results_text or "").strip()
        reasoning_text = (reasoning_text or "").strip()

        specs: List[Tuple[str, str, str, str, bool]] = []
        if plan_text:
            specs.append(
                (
                    "Pokaż proces analizy",
                    "Ukryj proces analizy",
                    "Plan analizy:",
                    plan_text,
                    False,
                )
            )
        if numerical_results_text:
            specs.append(
                (
                    "Pokaż wyniki obliczeń lokalnych",
                    "Ukryj wyniki obliczeń lokalnych",
                    "Wyniki obliczeń lokalnych:",
                    numerical_results_text,
                    True,
                )
            )
        if reasoning_text:
            specs.append(
                (
                    "Pokaż proces rozumowania",
                    "Ukryj proces rozumowania",
                    "Proces rozumowania modelu (chain of thought):",
                    reasoning_text,
                    False,
                )
            )
        if not specs:
            return

        msg_frame = ctk.CTkFrame(
            self.messages_frame,
            fg_color="transparent",
        )
        msg_frame._is_chat_message = True
        msg_frame.pack(pady=2, padx=10, fill="x")

        btn_row = ctk.CTkFrame(msg_frame, fg_color="transparent")
        btn_row.pack(anchor="w", padx=5, pady=2, fill="x")

        stack = ctk.CTkFrame(msg_frame, fg_color="transparent")
        stack.pack(fill="x")

        widths = {
            "Pokaż proces analizy": 200,
            "Pokaż wyniki obliczeń lokalnych": 268,
            "Pokaż proces rozumowania": 220,
        }

        for show_lbl, hide_lbl, title_lbl, body, monospace in specs:
            content_frame = ctk.CTkFrame(stack, fg_color="transparent")
            is_visible = [False]

            section_label = ctk.CTkLabel(
                content_frame,
                text=title_lbl,
                font=("Arial", 12, "bold"),
                anchor="w",
                text_color=COLORS["text_secondary"],
            )
            section_label.pack(anchor="w", pady=(8, 2))
            font_cb = ("Consolas", 11) if monospace else ("Arial", 11)
            body_box = ctk.CTkTextbox(
                content_frame,
                fg_color=COLORS["primary_dark"],
                text_color="#E0E0E0" if not monospace else "#FFFFFF",
                corner_radius=8,
                font=font_cb,
                wrap="word",
                height=min(400, max(120, body.count("\n") * 18 + 40)),
            )
            body_box.pack(fill="x", pady=(0, 8))
            body_box.insert("1.0", body)
            body_box.configure(state="disabled")

            btn = ctk.CTkButton(
                btn_row,
                text=show_lbl,
                width=widths.get(show_lbl, 220),
                height=32,
                fg_color=COLORS["primary_dark"],
                hover_color=COLORS["primary"],
                corner_radius=DIMENSIONS["corner_radius"],
                font=FONTS["body"],
                text_color_disabled=COLORS["text_disabled"],
            )

            def make_toggle(
                cf: ctk.CTkFrame,
                b: ctk.CTkButton,
                show_s: str,
                hide_s: str,
                vis: List[bool],
            ) -> Callable[[], None]:
                def toggle() -> None:
                    if vis[0]:
                        cf.pack_forget()
                        b.configure(text=show_s)
                        vis[0] = False
                    else:
                        cf.pack(fill="x", pady=(6, 4), padx=5)
                        b.configure(text=hide_s)
                        vis[0] = True
                    self._scroll_chat_to_end()

                return toggle

            btn.configure(
                command=make_toggle(
                    content_frame, btn, show_lbl, hide_lbl, is_visible
                )
            )
            btn.pack(side="left", padx=(0, 8))

        self._trim_message_history()
        self._scroll_chat_to_end()

    def _add_message(self, message: str, role: str, model: str = "") -> None:
        """!
        @brief Dodaje wiadomość do panelu czatu.
        
        @param message Treść wiadomości.
        @param role Rola nadawcy ('user', 'assistant', 'system').
        @param model Nazwa modelu AI (tylko dla roli 'assistant').
        
        @details Tworzy wizualny "dymek" z wiadomością, stosując
                 odpowiednie kolory i ikony zależnie od roli.
        """
        msg_frame = ctk.CTkFrame(
            self.messages_frame,
            fg_color="transparent"
        )
        msg_frame._is_chat_message = True
        msg_frame.pack(pady=5, padx=10, fill="x")
        
        if role == "user":
            bg_color = COLORS["primary"]
            text_color = "#FFFFFF"
            icon = "👤"
            anchor = "e"
        elif role == "assistant":
            bg_color = COLORS["primary_dark"]
            text_color = "#FFFFFF"
            icon = "🧠"
            anchor = "w"
        else:
            bg_color = COLORS["primary_dark"]
            text_color = "#FFFFFF"
            icon = "⚠️"
            anchor = "w"
        
        header_text = f"{icon} "
        if role == "user":
            header_text += "Ty"
        elif role == "assistant":
            header_text += f"AI{' (' + model + ')' if model else ''}"
        else:
            header_text += "System"
        
        header_label = ctk.CTkLabel(
            msg_frame,
            text=header_text,
            font=("Arial", 13, "bold"),
            anchor=anchor,
            text_color=COLORS["text_secondary"]
        )
        header_label.pack(anchor=anchor, padx=5)
        
        bubble = ctk.CTkTextbox(
            msg_frame,
            fg_color=bg_color,
            text_color=text_color,
            corner_radius=10,
            font=FONTS["body"],
            wrap="word",
        )
        bubble_width, bubble_height = self._estimate_bubble_size(
            message, force_max_width=(role in {"assistant", "system"})
        )
        bubble.configure(width=bubble_width, height=bubble_height)
        bubble.pack(anchor=anchor, padx=5, pady=5)
        
        self._insert_message_with_math(bubble, message)
        bubble.configure(state="disabled")
        self._fit_bubble_height_to_content(bubble, message)

        self._trim_message_history()

        def _bubble_top_left_after_layout() -> None:
            self._scroll_bubble_text_viewport(bubble, x_at_start=True, y_at_start=True)

        self.after_idle(lambda: self.after_idle(_bubble_top_left_after_layout))

        self._scroll_chat_to_end()
        
        logger.debug(f"Dodano wiadomość: role={role}, length={len(message)}")
    
    def clear_messages(self) -> None:
        """! @brief Usuwa wszystkie widgety z obszaru wiadomości."""
        for widget in self.messages_frame.winfo_children():
            widget.destroy()
        
        logger.info("Wyczyszczono historię czatu")
    
    def set_input_text(self, text: str) -> None:
        """!
        @brief Ustawia tekst w polu wejściowym.

        @param text Treść do wstawienia.
        """
        self.input_field.delete("1.0", "end")
        self.input_field.insert("1.0", text)
    
    def set_input_enabled(self, enabled: bool) -> None:
        """!
        @brief Włącza lub wyłącza pole wejściowe i przycisk Wyślij.

        @param enabled True - edycja dozwolona; False - zablokowane.
        """
        state = "normal" if enabled else "disabled"
        self.input_field.configure(state=state)
        self.send_btn.configure(state=state)
    
    def show_typing_indicator(
        self,
        model: str | None = None,
        status_text: str | None = None,
    ) -> None:
        """!
        @brief Pokazuje animowany wskaźnik pisania AI w bąbelku.
        
        @param model Nazwa modelu AI (opcjonalnie, w nagłówku).
        @param status_text Tekst statusu; gdy None - „Generowanie odpowiedzi…”.
        @details Tworzy wiadomość z nagłówkiem „🤖 AI” i bąbelkiem z etykietą oraz paskiem postępu.
        """
        if self.typing_indicator is not None:
            return

        status = (status_text or "Generowanie odpowiedzi…").strip()

        bg_color = COLORS["primary_dark"]
        msg_frame = ctk.CTkFrame(
            self.messages_frame,
            fg_color="transparent"
        )
        msg_frame._is_chat_message = True  # type: ignore[attr-defined]
        msg_frame.pack(pady=5, padx=10, fill="x")
        self.typing_indicator = msg_frame

        header_text = f"🤖 AI{' (' + model + ')' if model else ''}"
        ctk.CTkLabel(
            msg_frame,
            text=header_text,
            font=("Arial", 13, "bold"),
            anchor="w",
            text_color=COLORS["text_secondary"],
        ).pack(anchor="w", padx=5)

        bubble = ctk.CTkFrame(
            msg_frame,
            fg_color=bg_color,
            corner_radius=10,
        )
        bubble.pack(anchor="w", padx=5, pady=5, fill="x")

        self.typing_label = ctk.CTkLabel(
            bubble,
            text=status,
            font=("Arial", 13, "bold"),
            text_color=COLORS["accent"],
            anchor="w",
        )
        self.typing_label.pack(anchor="w", padx=10, pady=(10, 4))

        self.typing_progress = ctk.CTkProgressBar(
            bubble,
            mode="indeterminate",
            height=8,
            corner_radius=4,
            progress_color=COLORS["accent"],
        )
        self.typing_progress.pack(fill="x", padx=10, pady=(0, 10))
        self.typing_progress.start()

        self._typing_base_text = status.rstrip(".").rstrip("…")
        self.typing_dots = 0
        if self._status_uses_animated_dots(status):
            self._animate_typing()
        self._scroll_chat_to_end()
        logger.debug("Pokazano wskaźnik pisania")

    @staticmethod
    def _status_uses_animated_dots(status: str) -> bool:
        """! @brief Etapy strategii i wykonanie kodu mają stały tekst bez animacji kropek."""
        return not (status.startswith("Etap ") or status.startswith("⏳"))
    
    def _animate_typing(self) -> None:
        """! @brief Animuje kropki w etykiecie wskaźnika pisania (callback after)."""
        if self.typing_indicator is None:
            return
        
        self.typing_dots = (self.typing_dots + 1) % 4
        dots = "." * self.typing_dots
        base = getattr(self, '_typing_base_text', "Generowanie odpowiedzi")
        self.typing_label.configure(text=f"{base}{dots}")
        
        self.typing_animation_id = self.after(500, self._animate_typing)
    
    def update_typing_text(self, text: str) -> None:
        """!
        @brief Aktualizuje tekst wskaźnika pisania i bazę animacji kropek.

        @param text Nowy tekst do wyświetlenia (np. nazwa etapu).
        """
        if self.typing_indicator is not None and hasattr(self, 'typing_label'):
            self._typing_base_text = text.rstrip(".").rstrip("…")
            self.typing_label.configure(text=text)
            if self.typing_animation_id is not None:
                self.after_cancel(self.typing_animation_id)
                self.typing_animation_id = None
            if self._status_uses_animated_dots(text):
                self.typing_dots = 0
                self._animate_typing()

    def hide_typing_indicator(self) -> None:
        """! @brief Zatrzymuje animację i niszczy widget wskaźnika pisania."""
        if self.typing_animation_id is not None:
            self.after_cancel(self.typing_animation_id)
            self.typing_animation_id = None
        
        if self.typing_indicator is not None:
            self.typing_progress.stop()
            self.typing_indicator.destroy()
            self.typing_indicator = None
            logger.debug("Ukryto wskaźnik pisania")
    
    def start_streaming_message(self, model: str | None = None, placeholder_text: str | None = None) -> None:
        """!
        @brief Rozpoczyna streaming wiadomości AI.
        
        @param model Nazwa modelu AI (opcjonalne).
        @param placeholder_text Tekst paska do pierwszego fragmentu treści (np. „Trwa proces myślenia…”
                 lub „Generowanie odpowiedzi…”). Gdy None – bez paska, od razu pusta bąbelka.
        @return Referencja do textbox'a streaming.
        
        @details Tekst placeholdera ustawia MainWindow (_streaming_placeholder_text) - ten sam co przy braku streamingu.
        """
        self.hide_typing_indicator()
        self._streaming_thinking_placeholder = None

        msg_frame = ctk.CTkFrame(
            self.messages_frame,
            fg_color="transparent"
        )
        msg_frame._is_chat_message = True
        msg_frame.pack(pady=5, padx=10, fill="x")

        bg_color = COLORS["primary_dark"]
        text_color = "#FFFFFF"

        header_text = f"🤖 AI{' (' + model + ')' if model else ''}"
        header_label = ctk.CTkLabel(
            msg_frame,
            text=header_text,
            font=("Arial", 13, "bold"),
            anchor="w",
            text_color=COLORS["text_secondary"]
        )
        header_label.pack(anchor="w", padx=5)

        content_area = ctk.CTkFrame(msg_frame, fg_color="transparent")
        content_area.pack(anchor="w", padx=5, pady=5, fill="x")
        self._streaming_content_area = content_area

        show_placeholder = bool(placeholder_text and placeholder_text.strip())
        if show_placeholder:
            placeholder_frame = ctk.CTkFrame(content_area, fg_color=bg_color, corner_radius=10)
            placeholder_frame.pack(anchor="w", fill="x")
            ctk.CTkLabel(
                placeholder_frame,
                text=placeholder_text.strip(),
                font=("Arial", 13, "bold"),
                text_color=COLORS["accent"],
                anchor="w",
            ).pack(anchor="w", padx=10, pady=(10, 4))
            placeholder_progress = ctk.CTkProgressBar(
                placeholder_frame,
                mode="indeterminate",
                height=8,
                corner_radius=4,
                progress_color=COLORS["accent"],
            )
            placeholder_progress.pack(fill="x", padx=10, pady=(0, 10))
            placeholder_progress.start()
            self._streaming_thinking_placeholder = (placeholder_frame, placeholder_progress)
        else:
            self._streaming_thinking_placeholder = None

        self.streaming_bubble = ctk.CTkTextbox(
            content_area,
            fg_color=bg_color,
            text_color=text_color,
            corner_radius=10,
            font=FONTS["body"],
            wrap="word",
        )
        self.streaming_bubble.configure(width=self._compute_wraplength(), height=60)
        if not show_placeholder:
            self.streaming_bubble.pack(anchor="w")
        self.streaming_frame = msg_frame
        self.streaming_content = ""
        self._streaming_update_count = 0

        self._scroll_chat_to_end()
        logger.debug("Rozpoczęto streaming wiadomości")
        return self.streaming_bubble
    
    def append_streaming_content(self, content: str) -> None:
        """!
        @brief Dodaje chunk tekstu do bąbelka streamingu (złożoność O(chunk), nie O(total)).

        @param content Fragment tekstu do dodania.

        @details Przy pierwszym fragmencie treści usuwa pasek „myślenia” i pokazuje bąbelek.
                 Wstawia surowy tekst do pola bez parsowania Markdown/LaTeX; pełne formatowanie
                 wykonuje finish_streaming_message() po zakończeniu streamingu.
                 Pełny układ wysokości i przewijanie odświeżane są co piątą aktualizację (mniejszy narzut UI).
        """
        if not hasattr(self, 'streaming_bubble') or self.streaming_bubble is None:
            return

        placeholder_data = getattr(self, '_streaming_thinking_placeholder', None)
        if placeholder_data is not None:
            placeholder, progress = placeholder_data
            progress.stop()
            placeholder.destroy()
            self._streaming_thinking_placeholder = None
            self.streaming_bubble.pack(anchor="w")

        self.streaming_content += content
        self._streaming_update_count += 1

        do_full_layout = (self._streaming_update_count % 5 == 0)

        self.streaming_bubble.configure(state="normal")
        self.streaming_bubble._textbox.insert("end", content)
        self.streaming_bubble.configure(state="disabled")

        self._scroll_bubble_text_viewport(
            self.streaming_bubble, x_at_start=True, y_at_start=False
        )

        if do_full_layout:
            self._fit_bubble_height_to_content(self.streaming_bubble, self.streaming_content, defer=False)
            self._scroll_chat_to_end()
    
    def finish_streaming_message(self) -> None:
        """!
        @brief Kończy streaming i finalizuje wiadomość.
        
        @details Dezaktywuje edycję bąbelki, resetuje stan streamingu.
        """
        if self._fit_bubble_after_id is not None:
            self.after_cancel(self._fit_bubble_after_id)
            self._fit_bubble_after_id = None
        if hasattr(self, 'streaming_bubble') and self.streaming_bubble is not None:
            logger.debug(f"Zakończono streaming, długość: {len(self.streaming_content)}")
            if self.streaming_content:
                self.streaming_bubble.configure(state="normal")
                self.streaming_bubble.delete("1.0", "end")
                self._insert_message_with_math(self.streaming_bubble, self.streaming_content)
                self.streaming_bubble.configure(state="disabled")
            self.update_idletasks()
            self._fit_bubble_height_to_content(self.streaming_bubble, self.streaming_content, defer=False)
            self._scroll_bubble_text_viewport(
                self.streaming_bubble, x_at_start=True, y_at_start=True
            )
            self._scroll_chat_to_end()
            self._trim_message_history()
        
        if getattr(self, '_streaming_thinking_placeholder', None) is not None:
            try:
                placeholder, progress = self._streaming_thinking_placeholder
                progress.stop()
                placeholder.destroy()
            except Exception:
                pass
            self._streaming_thinking_placeholder = None
        self.streaming_bubble = None
        self.streaming_frame = None
        self.streaming_content = ""
        self._streaming_content_area = None
