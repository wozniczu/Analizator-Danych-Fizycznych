# Analizator Danych Fizycznych (ADF)

Desktopowa aplikacja do analizy danych pomiarowych z wykorzystaniem modeli AI (OpenAI, Anthropic, DeepSeek): wczytanie pliku, analiza w czacie, wizualizacje i podsumowania zużycia API.

## Instalacja i uruchomienie

Wymagania: **Python 3.9+**

W katalogu programu skopiuj **`.env.example`** do **`.env`** i uzupełnij co najmniej jeden klucz. Szablon zawiera puste pola:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
```

Zainstaluj biblioteki z pliku **`requirements.txt`** i uruchom aplikację:

```bash
pip install -r requirements.txt
python main_app.py
```

## Użycie

Wczytaj dane (**CSV, Excel, JSON**) z paska bocznego, wybierz dostawcę i model, strategię analizy, a następnie zadaj pytanie na czacie i obserwuj wyniki.

## Dokumentacja API

Aby wygenerować dokumentację w HTML:

1. Zainstaluj [Doxygen](https://www.doxygen.nl/download.html).
2. W katalogu z plikiem **`Doxyfile`** uruchom:

```bash
doxygen Doxyfile
```

3. Otwórz w przeglądarce plik **`docs/doxygen/html/index.html`**.