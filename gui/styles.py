##
## @file styles.py
## @brief Definicje motywów (dark, light, blue, green, purple), COLORS, FONTS, DIMENSIONS, get/set theme.


THEMES = {
    "dark": {
        "name": "Ciemny",
        "primary": "#8ba5d3",
        "primary_dark": "#45536a",
        "primary_light": "#e2ecf8",
        "accent": "#f6a307",
        "success": "#00D26A",
        "warning": "#FFB84D",
        "error": "#FF4444",
        "text": "#FFFFFF",
        "text_secondary": "#FFFFFF",
        "text_disabled": "#1a1a1a",
        "background": "#2B2B2B",
        "surface": "#3B3B3B",
        "ctk_mode": "dark"
    },
    "light": {
        "name": "Jasny",
        "primary": "#5a7fb8",
        "primary_dark": "#3d5a80",
        "primary_light": "#8ba5d3",
        "accent": "#e59400",
        "success": "#00B359",
        "warning": "#E6A300",
        "error": "#CC0000",
        "text": "#1a1a1a",
        "text_secondary": "#1a1a1a",
        "text_disabled": "#4a4a4a",
        "background": "#f5f5f5",
        "surface": "#ffffff",
        "ctk_mode": "light"
    },
    "blue": {
        "name": "Niebieski",
        "primary": "#2196F3",
        "primary_dark": "#1565C0",
        "primary_light": "#64B5F6",
        "accent": "#FF9800",
        "success": "#4CAF50",
        "warning": "#FFC107",
        "error": "#f44336",
        "text": "#FFFFFF",
        "text_secondary": "#FFFFFF",
        "text_disabled": "#1a1a1a",
        "background": "#0D1B2A",
        "surface": "#1B263B",
        "ctk_mode": "dark"
    },
    "green": {
        "name": "Zielony",
        "primary": "#4CAF50",
        "primary_dark": "#2E7D32",
        "primary_light": "#81C784",
        "accent": "#FFC107",
        "success": "#8BC34A",
        "warning": "#FF9800",
        "error": "#E53935",
        "text": "#FFFFFF",
        "text_secondary": "#FFFFFF",
        "text_disabled": "#1a1a1a",
        "background": "#1B2E1B",
        "surface": "#2E3D2E",
        "ctk_mode": "dark"
    },
    "purple": {
        "name": "Fioletowy",
        "primary": "#9C27B0",
        "primary_dark": "#6A1B9A",
        "primary_light": "#CE93D8",
        "accent": "#FF4081",
        "success": "#00E676",
        "warning": "#FFAB00",
        "error": "#FF1744",
        "text": "#FFFFFF",
        "text_secondary": "#FFFFFF",
        "text_disabled": "#1a1a1a",
        "background": "#1A1A2E",
        "surface": "#2D2D44",
        "ctk_mode": "dark"
    }
}

_current_theme = "dark"


def get_current_theme() -> str:
    """!
    @brief Zwraca nazwę aktualnie ustawionego motywu.

    @return Klucz z THEMES (np. 'dark', 'light').
    """
    return _current_theme


def set_theme(theme_name: str) -> bool:
    """!
    @brief Ustawia motyw, aktualizuje globalny COLORS i ctk_mode.

    @param theme_name Nazwa motywu (dark, light, blue, green, purple).
    @return True przy sukcesie, False gdy theme_name nie istnieje w THEMES.
    """
    global _current_theme, COLORS

    if theme_name not in THEMES:
        return False

    _current_theme = theme_name
    theme = THEMES[theme_name]

    COLORS.update({
        "primary": theme["primary"],
        "primary_dark": theme["primary_dark"],
        "primary_light": theme["primary_light"],
        "accent": theme["accent"],
        "success": theme["success"],
        "warning": theme["warning"],
        "error": theme["error"],
        "text": theme["text"],
        "text_secondary": theme["text_secondary"],
        "text_disabled": theme["text_disabled"],
        "background": theme["background"],
        "surface": theme["surface"],
    })
    
    return True


def get_theme_names() -> list:
    """!
    @brief Zwraca listę par (klucz, nazwa wyświetlana) dla wszystkich motywów.

    @return Lista krotek (str, str).
    """
    return [(key, theme["name"]) for key, theme in THEMES.items()]


COLORS = {
    "primary": "#8ba5d3",
    "primary_dark": "#45536a",
    "primary_light": "#e2ecf8",
    "accent": "#f6a307",
    "success": "#00D26A",
    "warning": "#FFB84D",
    "error": "#FF4444",
    "text": "#FFFFFF",
    "text_secondary": "#FFFFFF",
    "text_disabled": "#1a1a1a",
    "background": "#2B2B2B",
    "surface": "#3B3B3B",
}

FONTS = {
    "title": ("Arial", 28, "bold"),
    "heading": ("Arial", 18, "bold"),
    "subheading": ("Arial", 16, "bold"),
    "body": ("Arial", 14),
    "small": ("Arial", 13),
    "code": ("Courier New", 13),
}

DIMENSIONS = {
    "sidebar_width": 340,
    "button_height": 45,
    "input_height": 50,
    "padding_small": 12,
    "padding_medium": 18,
    "padding_large": 24,
    "corner_radius": 10,
}

FIGURE_DEFAULT_SIZE = (8, 4)
FIGURE_DPI = 100
FIGURE_EXPORT_DPI = 300
CHART_MAX_COLUMNS_LINE = 5
CHART_MAX_COLUMNS_HIST = 5
CHART_MAX_COLUMNS_BOX = 6
CHART_MAX_COLUMNS_BAR = 8
CHART_LABEL_MAX_LEN = 10
FIGURE_ASPECT_WIDTH_MIN = 6
FIGURE_ASPECT_WIDTH_MAX = 12
FIGURE_ASPECT_FACTOR = 5

PREVIEW_MAX_COLUMNS = 5
