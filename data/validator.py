##
## @file validator.py
## @brief Walidacja i czyszczenie DataFrame: puste, duplikaty kolumn, NaN, kolumny numeryczne, rozmiar.

import pandas as pd
from typing import List
from utils.logger import logger


class DataValidator:
    """!
    @brief Sprawdza DataFrame pod kątem typowych problemów, zwraca listę ostrzeżeń. clean() usuwa puste wiersze/kolumny i strip nazw.
    """

    def validate(self, df: pd.DataFrame) -> List[str]:
        """!
        @brief Zbiera ostrzeżenia: pusty DataFrame, zduplikowane nazwy kolumn, NaN, brak kolumn numerycznych, < 3 wierszy.

        @param df DataFrame do walidacji.
        @return Lista komunikatów (pusta przy braku problemów).
        """
        issues = []

        logger.info("Walidacja danych...")

        if df.empty:
            issues.append("DataFrame jest pusty")
            return issues

        if df.columns.duplicated().any():
            duplicated = df.columns[df.columns.duplicated()].tolist()
            issues.append(f"Zduplikowane nazwy kolumn: {duplicated}")

        nan_counts = df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if not cols_with_nan.empty:
            for col, count in cols_with_nan.items():
                percent = (count / len(df)) * 100
                issues.append(
                    f"Kolumna '{col}': {count} wartości NaN ({percent:.1f}%)"
                )

        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            issues.append("Brak kolumn numerycznych")

        if len(df) < 3:
            issues.append(f"Bardzo mały zbiór danych ({len(df)} wierszy)")
        
        if issues:
            logger.warning(f"Znaleziono {len(issues)} problemów w danych")
        else:
            logger.info("Dane prawidłowe")
        
        return issues
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """!
        @brief Kopiuje DataFrame, usuwa wiersze/kolumny w całości puste (dropna), stripuje nazwy kolumn.

        @param df DataFrame do wyczyszczenia.
        @return Wyczyszczona kopia.
        """
        logger.info("Czyszczenie danych...")

        df_clean = df.copy()

        df_clean = df_clean.dropna(how='all')
        df_clean = df_clean.dropna(axis=1, how='all')
        df_clean.columns = df_clean.columns.str.strip()
        
        logger.info(f"Po czyszczeniu: {len(df_clean)} wierszy, {len(df_clean.columns)} kolumn")
        
        return df_clean