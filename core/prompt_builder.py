##
## @file prompt_builder.py
## @brief Budowanie promptów (direct, code) i formatowanie podsumowań DataFrame, instrukcja formatowania dla UI (Markdown, LaTeX).

from typing import Optional
import pandas as pd

FORMATTING_INSTRUCTION = (
    "W odpowiedziach używaj wyłącznie formatowania obsługiwanego przez aplikację:\n"
    "- Pogrubienie: **tekst**\n"
    "- Kursywa: *tekst* (pojedyncza gwiazdka)\n"
    "- Nagłówki: na początku linii #, ## lub ### (ze spacją po nich)\n"
    "- Wzory LaTeX: inline \\( wzór \\), blokowe \\[ wzór \\]. Nie używaj $$ ani innych delimiterów.\n"
    "- Każda tabela z danymi (statystyki, porównania, zestawienia) MUSI być w LaTeXie wewnątrz jednego bloku "
    "\\[ ... \\], ze środowiskiem \\begin{array}{...} ... \\end{array} albo \\begin{tabular}{...} ... \\end{tabular}. "
    "Aplikacja nie renderuje tabel Markdown.\n"
    "- ZAKAZ: tabel w stylu Markdown z pionowymi kreskami (np. wiersze | kolumna | wartość | lub linie :---). "
    "Nie zastępuj LaTeXu takimi tabelami ani ASCII-artem z + i -.\n"
    "- W tabelach LaTeX używaj '&' między komórkami w wierszu i '\\\\' na końcu wiersza (nie jednego backslasha).\n"
    "- Przykład poprawnej tabeli: \\[ \\begin{array}{rrrr} nag1 & nag2 \\\\ a & b \\\\ c & d \\end{array} \\]\n"
    "- W nagłówkach i komórkach tabel preferuj krótki zwykły tekst, np. speed_mps, acc_mps2. Ogranicz się do prostych zapisów.\n"
    "- Unikaj w tabelach komendy \\text{...}, jeśli nie jest konieczna; lepsze są krótkie etykiety bez dodatkowych makr.\n"
    "- Unikaj w tabelach LaTeX komend złożonych (\\multicolumn, \\multirow, booktabs), zagnieżdżonych środowisk i własnych makr.\n"
    "Unikaj HTML i innego znacznikowania poza wymienionym Markdownem i LaTeXem."
)


def system_prompt_with_formatting(base: str) -> str:
    """!
    @brief Dołącza FORMATTING_INSTRUCTION do promptu systemowego (Markdown, LaTeX zgodne z UI).

    @param base Treść promptu systemowego.
    @return base + instrukcja formatowania.
    """
    return base.rstrip() + "\n\n" + FORMATTING_INSTRUCTION


CODE_EXECUTION_SYSTEM_BASE = (
    "Jesteś ekspertem fizykiem analizującym dane eksperymentalne.\n"
    "Używaj narzędzia do wykonywania kodu Pythona w sandboxie do obliczeń i generowania wykresów.\n"
    "Dane eksperymentalne są w pliku CSV w środowisku wykonania - wczytaj je przez pd.read_csv.\n"
    "Zapisując wykresy, używaj w plt.savefig czytelnej nazwy pliku opisującej treść "
    "(np. histogram_masy.png, zaleznosc_t_od_czasu.png) - aplikacja wyświetli tę nazwę "
    "zamiast identyfikatora technicznego.\n"
    "Po wykonaniu obliczeń przedstaw wyniki z interpretacją fizyczną. Odpowiadaj po polsku."
)


def code_execution_system_prompt() -> str:
    """! @brief Wspólny prompt systemowy dla OpenAI Code Interpreter i Anthropic Code execution."""
    return system_prompt_with_formatting(CODE_EXECUTION_SYSTEM_BASE)


def format_code_execution_data_info(df: pd.DataFrame) -> str:
    """! @brief Metadane DataFrame do promptu wykonywania kodu (bez pełnej tabeli)."""
    return (
        f"Kolumny: {list(df.columns)}\n"
        f"Liczba wierszy: {len(df)}\n"
        f"Typy danych: {df.dtypes.to_dict()}"
    )


def build_code_execution_user_prompt(
    data_loading_instructions: str,
    data_info: str,
    question: str,
) -> str:
    """!
    @brief Wspólny prompt użytkownika dla strategii wykonywania kodu.

    @param data_loading_instructions Provider-specific: ścieżka w kontenerze lub nazwa załącznika.
    @param data_info Tekst z format_code_execution_data_info.
    @param question Pytanie użytkownika.
    """
    return (
        f"{data_loading_instructions.rstrip()}\n\n"
        f"Informacje o danych:\n{data_info}\n\n"
        f"Pytanie: {question}\n\n"
        "Użyj narzędzia do wykonywania kodu Pythona do analizy tych danych (obliczenia, wykresy)."
    )


def openai_code_execution_data_loading(data_path: str) -> str:
    """! @brief Instrukcja wczytania CSV z kontenera OpenAI Code Interpreter."""
    return (
        f"Dane eksperymentalne są w pliku CSV w kontenerze pod ścieżką:\n{data_path}\n"
        f"Wczytaj je w kodzie Pythona, np.: df = pd.read_csv({repr(data_path)})"
    )


def anthropic_code_execution_data_loading(csv_filename: str) -> str:
    """! @brief Instrukcja wczytania CSV z workspace Anthropic Code execution."""
    return (
        f"Dane eksperymentalne są w załączonym pliku CSV (nazwa: {csv_filename}).\n"
        f"Wczytaj je w kodzie Pythona, np.: df = pd.read_csv({repr(csv_filename)}) "
        "lub sprawdź listę plików w katalogu roboczym."
    )


class PromptBuilder:
    """!
    @brief Tworzy prompty do analizy bezpośredniej, wykonywania kodu oraz formatuje podsumowania DataFrame.
    """

    @staticmethod
    def build_direct_analysis_prompt(
        data_summary: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """!
        @brief Skleja opcjonalny system prompt, podsumowanie danych i pytanie w jeden prompt.

        @param data_summary Tekstowe podsumowanie danych (np. format_data_summary).
        @param question Pytanie użytkownika.
        @param system_prompt Opcjonalna dodatkowa instrukcja na początku.
        @return Pełny prompt do wysłania do modelu.
        """
        if system_prompt:
            prompt = f"{system_prompt}\n\n"
        else:
            prompt = ""
        
        prompt += f"""Dane eksperymentalne:
{data_summary}

Pytanie: {question}"""
        
        return prompt
    
    @staticmethod
    def build_code_generation_prompt(
        data_description: str,
        question: str
    ) -> str:
        """!
        @brief Buduje prompt z opisem struktury danych i zadaniem, wymaga kodu bez wczytywania plików.

        @param data_description Opis kolumn/typow (np. format_data_description).
        @param question Zadanie do wykonania na DataFrame 'df'.
        @return Prompt do wykonywania kodu Python.
        """
        prompt = f"""Wygeneruj kod Python do analizy danych.

Struktura danych (DataFrame 'df'):
{data_description}

Zadanie: {question}

Wymagania:
- Kod powinien zakładać, że DataFrame jest dostępny jako zmienna 'df'
- Użyj pandas, numpy do obliczeń
- Wypisz wyniki używając print()
- NIE generuj kodu do wczytywania plików
- Generuj TYLKO kod Python, bez markdown ani wyjaśnień
"""
        return prompt
    
    @staticmethod
    def format_data_summary(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
        """!
        @brief Zwraca tekst: rozmiar, kolumny z typami, oraz head(max_rows) lub całą tabelę.

        @param df DataFrame do podsumowania.
        @param max_rows Maks. liczba wierszy (None = wszystkie).
        @return Tekstowe podsumowanie do promptu.
        """
        summary = f"Rozmiar: {len(df)} wierszy × {len(df.columns)} kolumn\n\n"
        summary += "Kolumny:\n"
        
        for col in df.columns:
            dtype = df[col].dtype
            summary += f"  • {col} ({dtype})\n"
        
        if max_rows is not None:
            n_show = min(max_rows, len(df))
            summary += f"\nPierwsze {n_show} wierszy:\n"
            summary += df.head(max_rows).to_string(index=False)
        else:
            summary += f"\nWszystkie {len(df)} wierszy:\n"
            summary += df.to_string(index=False)
        
        return summary
    
    @staticmethod
    def format_data_description(df: pd.DataFrame) -> str:
        """!
        @brief Zwraca opis struktury: kolumny, typy, shape, head(3) - bez pełnych danych.

        @param df DataFrame.
        @return Opis do promptu wykonywania kodu.
        """
        desc = f"Kolumny: {list(df.columns)}\n"
        desc += f"Typy danych: {df.dtypes.to_dict()}\n"
        desc += f"Kształt: {df.shape}\n"
        desc += f"\nPrzykładowe wiersze:\n"
        desc += df.head(3).to_string()
        
        return desc