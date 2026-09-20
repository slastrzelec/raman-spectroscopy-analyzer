# Raman Spectroscopy Analyzer

Interaktywna aplikacja Streamlit do analizy widm Ramana: korekta linii bazowej,
normalizacja, wygładzanie, detekcja i dopasowanie pików oraz eksport wyników.

Pełna specyfikacja projektu (architektura, uzasadnienie wersji zależności,
bezpieczeństwo danych): zobacz [`SPEC.md`](SPEC.md).

## Funkcje

- Wczytanie widma: jedno z 3 przykładowych widm CNT-COOH albo własny plik `.txt`
  (dwie kolumny: Wavenumber, Intensity) — upload wspiera też wiele plików naraz
- Preprocessing: korekta linii bazowej (Linear / ALS), normalizacja, wygładzanie
- Detekcja i dopasowanie pików (Lorentzian / Gaussian / Pseudo-Voigt)
- Interaktywne wykresy Plotly (z fallbackiem do Matplotlib)
- Eksport wyników: CSV (widmo/piki/dopasowanie), XLSX (wszystko w jednym pliku),
  raport TXT — oraz **eksport wsadowy**: te same ustawienia zastosowane do
  wszystkich wgranych plików naraz, spakowane do jednego ZIP

## Uruchomienie lokalne

```bash
pip install -r requirements.txt
streamlit run app.py
```

Wymaga Pythona 3.11 (patrz `runtime.txt`).

## Testy

```bash
pytest
```

## Struktura projektu

```
app.py              # UI Streamlit (orkiestracja, bez logiki biznesowej)
config/settings.py  # stałe i wartości domyślne
utils/               # logika: wczytywanie danych, preprocessing, detekcja/dopasowanie
                     # pików, wizualizacja, eksport, przetwarzanie wsadowe (batch)
data/raw/            # przykładowe widma
tests/               # testy pytest dla każdego modułu w utils/
```
