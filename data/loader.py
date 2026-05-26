##
## @file loader.py
## @brief Wczytywanie danych z plików CSV, Excel, JSON z auto-detekcją kodowania (CSV).

import pandas as pd
from pathlib import Path
from typing import Union
from utils.logger import logger


class DataLoader:
    """!
    @brief Ładuje pliki danych w obsługiwanych formatach (SUPPORTED_FORMATS).
    """

    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json']

    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """!
        @brief Wczytuje plik i zwraca DataFrame, wybiera _load_csv/_load_excel/_load_json według rozszerzenia.

        @param filepath Ścieżka do pliku.
        @return DataFrame z danymi.
        @exception FileNotFoundError Plik nie istnieje.
        @exception ValueError Nieobsługiwany format.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Plik nie istnieje: {filepath}")
        
        extension = filepath.suffix.lower()
        
        logger.info(f"Wczytywanie pliku: {filepath}")
        
        try:
            if extension == '.csv':
                df = self._load_csv(filepath)
            elif extension in ['.xlsx', '.xls']:
                df = self._load_excel(filepath)
            elif extension == '.json':
                df = self._load_json(filepath)
            else:
                raise ValueError(
                    f"Nieobsługiwany format: {extension}. "
                    f"Obsługiwane: {self.SUPPORTED_FORMATS}"
                )
            
            logger.info(f"Wczytano {len(df)} wierszy, {len(df.columns)} kolumn")
            return df
            
        except Exception as e:
            logger.error(f"Błąd wczytywania pliku: {e}", exc_info=True)
            raise
    
    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """!
        @brief Wczytuje CSV z kolejną próbą kodowań (utf-8, utf-8-sig, latin1, cp1250), sep=None, engine='python'.

        @param filepath Ścieżka do pliku CSV.
        @return DataFrame.
        @exception ValueError Żadne kodowanie nie zadziałało.
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1250']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    filepath,
                    encoding=encoding,
                    sep=None,
                    engine='python'
                )
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        raise ValueError(f"Nie udało się wczytać CSV: {filepath}")
    
    def _load_excel(self, filepath: Path) -> pd.DataFrame:
        """!
        @brief Wczytuje pierwszą arkusz Excela (sheet_name=0).

        @param filepath Ścieżka do .xlsx/.xls.
        @return DataFrame.
        """
        return pd.read_excel(filepath, sheet_name=0)

    def _load_json(self, filepath: Path) -> pd.DataFrame:
        """!
        @brief Wczytuje JSON do DataFrame (pd.read_json).

        @param filepath Ścieżka do .json.
        @return DataFrame.
        """
        return pd.read_json(filepath)