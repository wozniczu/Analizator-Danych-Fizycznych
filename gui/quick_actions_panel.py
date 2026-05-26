##
## @file quick_actions_panel.py
## @brief Stały panel boczny z szybkimi akcjami.
##
import customtkinter as ctk
from typing import Any, Callable

from gui.styles import COLORS, FONTS, DIMENSIONS


class QuickActionsPanel(ctk.CTkFrame):
    """!
    @brief Panel z przyciskami szybkich akcji (Statystyki, Analiza, Wizualizuj, Anomalie, Korelacja), callback on_quick_action(id).
    """

    WIDTH = 130

    def __init__(self, parent: Any, on_quick_action: Callable[[str], None], **kwargs: Any) -> None:
        """!
        @brief Konstruktor panelu szybkich akcji.

        @param parent Widget nadrzędny (grid).
        @param on_quick_action Callback z identyfikatorem akcji (np. statistics, visualize).
        """
        super().__init__(
            parent,
            width=self.WIDTH,
            fg_color=COLORS["surface"],
            corner_radius=DIMENSIONS["corner_radius"],
            border_width=0,
            **kwargs
        )
        self.on_quick_action = on_quick_action

        self.grid_propagate(False)
        self.configure(width=self.WIDTH)

        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=2, pady=2)

        label = ctk.CTkLabel(
            content_frame,
            text="Szybkie akcje",
            font=FONTS["subheading"],
            anchor="w",
            text_color=COLORS["text"]
        )
        label.pack(pady=(15, 10), padx=8, fill="x")

        actions = [
            ("Statystyki", "statistics"),
            ("Analiza", "analysis"),
            ("Wizualizuj", "visualize"),
            ("Anomalie", "anomalies"),
            ("Korelacja", "correlation")
        ]

        for text, action in actions:
            btn = ctk.CTkButton(
                content_frame,
                text=text,
                command=lambda a=action: self.on_quick_action(a),
                height=35,
                corner_radius=DIMENSIONS["corner_radius"],
                fg_color=COLORS["surface"],
                border_width=2,
                border_color=COLORS["primary"],
                font=FONTS["small"],
                text_color_disabled=COLORS["text_disabled"]
            )
            btn.pack(pady=5, padx=8, fill="x")
