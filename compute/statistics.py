##
## @file statistics.py
## @brief Statystyki opisowe, korelacja, regresja liniowa, dopasowanie kwadratowe,
##        wykrywanie wartości odstających, pochodna ilorazowa i pochodna numeryczna.

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from scipy import stats
from utils.logger import logger

class StatisticsCalculator:
    """!
    @brief Metody statyczne: statystyki opisowe, korelacja, regresja, outliery, pochodna.
    """

    @staticmethod
    def basic_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """!
        @brief Oblicza mean, median, std, min, max, q25, q75 dla każdej kolumny numerycznej.

        @param df DataFrame z danymi.
        @return Słownik {nazwa_kolumny: {stat_name: wartość}}.
        """
        logger.debug("Obliczanie statystyk podstawowych")
        
        stats_dict = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            stats_dict[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }
        
        return stats_dict
    
    @staticmethod
    def correlation_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """!
        @brief Zwraca macierz korelacji kolumn numerycznych (corr().to_dict()).

        @param df DataFrame z danymi.
        @return Zagnieżdżony słownik.
        """
        logger.debug("Obliczanie macierzy korelacji")
        
        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()
        
        return corr_matrix.to_dict()
    
    @staticmethod
    def linear_regression(
        x: pd.Series,
        y: pd.Series
    ) -> Dict[str, float]:
        """!
        @brief Dopasowanie prostej (scipy.stats.linregress), pomija pary z NaN.

        @param x Zmienna niezależna.
        @param y Zmienna zależna.
        @return Słownik slope, intercept, r_squared, p_value, std_err.
        """
        logger.debug("Dopasowanie regresji liniowej")

        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x_clean, y_clean
        )
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err)
        }
    
    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> List[int]:
        """!
        @brief Zwraca indeksy wierszy uznanych za wartości odstające (IQR lub z-score).

        @param df DataFrame z danymi.
        @param method 'iqr' lub 'zscore'.
        @param threshold Mnożnik IQR lub próg |z|.
        @return Posortowana lista indeksów.
        """
        logger.debug(f"Detekcja wartości odstających metodą {method}")
        
        outliers_indices = set()
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df[
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ]
                outliers_indices.update(outliers.index.tolist())
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df.iloc[np.where(z_scores > threshold)[0]]
                outliers_indices.update(outliers.index.tolist())
        
        return sorted(list(outliers_indices))
    
    @staticmethod
    def numerical_derivative(
        x: pd.Series,
        y: pd.Series
    ) -> Dict[str, Any]:
        """!
        @brief Pochodna numeryczna (różnice centralne), zwraca x (środki), dy_dx, mean, std.

        @param x Zmienna niezależna.
        @param y Zmienna zależna.
        @return Słownik x, dy_dx, mean, std.
        """
        logger.debug("Obliczanie pochodnej numerycznej")

        dx = np.diff(x.values)
        dy = np.diff(y.values)
        derivative = dy / dx
        x_derivative = (x.values[:-1] + x.values[1:]) / 2
        
        return {
            'x': x_derivative.tolist(),
            'dy_dx': derivative.tolist(),
            'mean': float(np.mean(derivative)),
            'std': float(np.std(derivative))
        }

    @staticmethod
    def polynomial_fit_degree2(
        x: pd.Series,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """!
        @brief Wielomian stopnia 2: ``np.polyfit``, \\(y \\approx a x^2 + b x + c\\).

        @return Słownik a, b, c (współczynniki przy \\(x^2, x^1, x^0\\)), r_squared, lub error.
        """
        logger.debug("Dopasowanie wielomianu stopnia 2")
        mask = ~(x.isna() | y.isna())
        t_clean = x[mask].to_numpy(dtype=float)
        v_clean = y[mask].to_numpy(dtype=float)
        if len(t_clean) < 3:
            return {'error': 'Za mało punktów (wymagane co najmniej 3 po usunięciu NaN)'}
        coeffs = np.polyfit(t_clean, v_clean, 2)
        a_f, b_f, c_f = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        y_pred = np.polyval(coeffs, t_clean)
        ss_res = float(np.sum((v_clean - y_pred) ** 2))
        ss_tot = float(np.sum((v_clean - float(np.mean(v_clean))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
        return {
            'a': a_f,
            'b': b_f,
            'c': c_f,
            'r_squared': float(r2),
            'coeffs_descending': [a_f, b_f, c_f],
        }

    @staticmethod
    def numerical_gradient_at_row_nearest_t(
        x: pd.Series,
        y: pd.Series,
        t_target: float = 5.0,
    ) -> Dict[str, Any]:
        """!
        @brief Pochodna ``np.gradient(y, x)`` w wierszu z minimalnym \\(|x - t_{target}|\\).

        @param t_target Szukany punkt na osi x (domyślnie 5.0).
        @return pochodna_t, t_row, t_target lub error.
        """
        logger.debug(
            f"Pochodna numpy.gradient w punkcie najbliższym t_target={t_target}",
        )
        mask = ~(x.isna() | y.isna())
        t_arr = x[mask].to_numpy(dtype=float)
        v_arr = y[mask].to_numpy(dtype=float)
        if len(t_arr) < 2:
            return {'error': 'Za mało punktów (wymagane co najmniej 2 po usunięciu NaN)'}
        grad = np.gradient(v_arr, t_arr)
        idx = int(np.argmin(np.abs(t_arr - t_target)))
        return {
            'pochodna_t': float(grad[idx]),
            't_row': float(t_arr[idx]),
            't_target': float(t_target),
        }
