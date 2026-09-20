# Specyfikacja — Raman Spectroscopy Analyzer (przepisanie od zera)

Status: **DO ZATWIERDZENIA** — bez implementacji, dopóki nie zaakceptujesz tej specyfikacji.

## 1. Cel

Aplikacja Streamlit do analizy widm Ramana: wczytanie widma (przykładowego lub własnego),
korekta linii bazowej, normalizacja, wygładzanie, detekcja i dopasowanie pików, eksport wyników.
Funkcjonalnie odtwarza poprzednią wersję (patrz sekcja 4), ale z czystą architekturą modułową,
zweryfikowanymi wersjami zależności i testami od pierwszego commita.

## 2. Wersje zależności — i dlaczego akurat te

Poprzedni deploy padał dwa razy z dwóch różnych, konkretnych przyczyn: (a) `pyarrow==25.0.1` ma
udokumentowany, znany bug powodujący segfault na Streamlit Cloud, (b) `streamlit==1.28.0` obsługiwał
`st.rerun()` rekurencyjnie zamiast w pętli, więc stos wywołań rósł z każdym rerunem aż do
`RecursionError` (naprawione w PR #8100, Streamlit 1.32.0). Dlatego każda wersja poniżej jest
świadomie sprawdzona, nie tylko "najnowsza z PyPI":

| Pakiet | Wersja | Uzasadnienie |
|---|---|---|
| Python | 3.11 | Spójne z resztą Twoich projektów portfolio (`runtime.txt`) |
| streamlit | 1.64.0 | Aktualna stabilna; zawiera fix #8100 (rerun w pętli, nie rekurencyjnie) |
| pandas | 3.0.6 | Aktualna stabilna. Breaking changes (Copy-on-Write, domyślny `str` dtype) nie dotyczą nas — piszemy kod od zera, bez starych wzorców typu chained assignment |
| numpy | 2.4.6 | Najnowsza wersja *wspierająca Python 3.11* (2.5.x wymaga już Pythona 3.12) |
| scipy | 1.17.1 | Najnowsza wersja *wspierająca Python 3.11* (1.18.x wymaga już Pythona 3.12) |
| matplotlib | 3.11.2 | Aktualna stabilna (fallback dla trybu non-interactive) |
| plotly | 7.1.0 | Aktualna stabilna; podstawowe API (`go.Figure`, `update_layout`, `add_trace`) niezmienione względem 5.x, brak zgłoszonych krytycznych regresji |
| **pyarrow** | **24.0.0** | **Świadomie NIE najnowsza.** 25.0.1 ma potwierdzony known-segfault (to on stał za dzisiejszym crashem pośrednio wykrytym przez auto-resolver Streamlit Cloud). 24.0.0 to najnowsza wersja *przed* tym problemem (kwiecień 2026), wystarczająco świeża pod pandas 3.0 |
| openpyxl | 3.1.5 | Aktualna stabilna (eksport do .xlsx) |
| pytest | 9.1.1 | Aktualna stabilna |

Wszystkie wersje **pinowane na sztywno** (`==`) w `requirements.txt`, plus `runtime.txt` z `python-3.11`
ustawiane jawnie (Streamlit Cloud domyślnie proponuje najnowszy Python, obecnie 3.14 — nieprzetestowany
z tymi pinami).

## 3. Struktura modułów

```
raman-spectroscopy-analyzer/
├── app.py                  # tylko UI/orkiestracja Streamlit — bez logiki biznesowej
├── config/
│   └── settings.py         # stałe, wartości domyślne (progi, kolory, etykiety)
├── utils/
│   ├── data_loading.py     # wczytywanie plików .txt (upload + przykładowe), walidacja formatu
│   ├── preprocessing.py    # baseline correction (Linear, ALS), normalizacja, wygładzanie
│   ├── peak_detection.py   # wykrywanie pików
│   ├── peak_fitting.py     # dopasowanie pików (osobny moduł — w starej wersji było zlepione z detekcją)
│   ├── visualization.py    # wykresy Plotly/Matplotlib
│   └── export.py           # eksport CSV/XLSX/raport (w starej wersji leżało wprost w app.py)
├── data/
│   └── raw/                # 3 przykładowe widma (już są w folderze)
├── tests/
│   ├── conftest.py         # fixtures korzystające z prawdziwych plików w data/raw
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   ├── test_peak_detection.py
│   ├── test_peak_fitting.py
│   └── test_export.py
├── .github/workflows/
│   └── tests.yml           # pytest na push/PR, Python 3.11
├── .streamlit/
│   └── config.toml         # theme (patrz sekcja 6)
├── requirements.txt
├── runtime.txt
├── pytest.ini
├── .gitignore
├── LICENSE
├── README.md
└── SPEC.md                 # ten dokument
```

Zasada: `app.py` nie zawiera obliczeń ani logiki eksportu — tylko wywołania funkcji z `utils/*` i
renderowanie wyników. Każdy moduł w `utils/` jest testowalny w izolacji (funkcje przyjmują/zwracają
`DataFrame`/`dict`, żadnych bezpośrednich odwołań do `st.session_state` wewnątrz `utils/`).

## 4. Funkcje aplikacji (zakres — jak w poprzedniej wersji)

**Sidebar:**
- Wybór źródła danych: (a) jedno z 3 przykładowych widm z `data/raw`, (b) upload własnego pliku(ów) `.txt`
  (tryb pojedynczy/wielokrotny, limit 200MB/plik, format: dwie kolumny Wavenumber/Intensity,
  separator spacja lub tab)
- Ustawienia: interaktywne wykresy (Plotly) wł/wył, siatka wł/wył

**Zakładki:**
1. **Data Overview** — podgląd surowego widma, statystyki (liczba punktów, zakres wavenumber,
   max intensity), tabela danych
2. **Preprocessing**:
   - Baseline Correction: Linear (Endpoints, liczba punktów końcowych) lub ALS (Asymmetric Least
     Squares, parametry λ i p)
   - Normalizacja (odblokowana po korekcie baseline)
   - Wygładzanie (odblokowane po normalizacji)
3. **Peak Detection** — wykrywanie pików na przetworzonym widmie
4. **Peak Fitting** — dopasowanie profili pików, niepewności dopasowania
5. **Analysis & Results** — zestawienie wyników, metryki
6. **Export** — eksport pików/dopasowania/raportu (CSV/XLSX), tryb wsadowy dla wielu plików

## 5. Wybór widma przykładowego — nowa funkcja

Selector nad/obok uploadera: "Użyj przykładowego widma" (domyślnie pokazuje 3 pliki z `data/raw` pod
czytelnymi etykietami, np. "Przykład 1 — CNT-COOH (a)") **albo** "Wgraj własny plik". Wybór
przykładu nie różni się dalej niczym w przetwarzaniu od pliku wgranego — trafia do tej samej ścieżki
`data_loading.py`, więc nie ma dwóch osobnych ścieżek kodu do utrzymania.

## 6. Szata wizualna

Spójna z Twoim portfolio (MkDocs Material): **primary = deep orange, accent = orange**, czcionka
tekstu Roboto, czcionka kodu/danych JetBrains Mono. W `.streamlit/config.toml`:
`primaryColor` w odcieniu deep orange (`#FF5722`), `font = "sans serif"`. Wykresy Plotly:
paleta kategoryczna i sekwencyjna dobrana pod kątem dostępności (WCAG, colorblind-safe) z akcentem
pomarańczowym jako kolorem głównym serii, zamiast domyślnego niebieskiego z poprzedniej wersji.

## 7. Bezpieczeństwo danych

- Żadnych zewnętrznych wywołań sieciowych ani API — appka działa w pełni lokalnie/na Streamlit Cloud,
  bez kluczy/sekretów do zarządzania
- Wgrane pliki użytkownika przetwarzane wyłącznie w pamięci (`st.session_state` / DataFrame) —
  **nigdy nie zapisywane na dysk appki**, więc nie ma ryzyka wycieku między sesjami/użytkownikami
- Walidacja formatu wgrywanego pliku przed parsowaniem (rozszerzenie `.txt`, sprawdzenie że da się
  sparsować jako dwie kolumny liczbowe) — błędny plik zwraca czytelny komunikat, nie stack trace
- Brak `eval`/`exec`/`pickle.load` na danych pochodzących od użytkownika
- Przykładowe dane w `data/raw` są plikami repo (read-only w runtime), nie mieszają się z danymi
  wgrywanymi przez użytkownika
- `.gitignore` wyklucza `__pycache__/`, `.pytest_cache/`, lokalne środowiska wirtualne — nic z Twojego
  komputera poza kodem nie trafia do repo
- Zależności pinowane na sztywno (sekcja 2) — to też kwestia bezpieczeństwa/stabilności łańcucha
  dostaw, nie tylko wygody

## 8. Testy i CI

Pytest dla każdego modułu w `utils/`, fixtures oparte o prawdziwe pliki z `data/raw` (nie trzeba
syntetycznych danych). GitHub Actions (`.github/workflows/tests.yml`) uruchamia `pytest` na Python
3.11 przy każdym push/PR do `main`. Piszę testy równolegle z każdym modułem, nie na końcu.

## 9. Deployment

Zgodnie z ustaleniem: cały kod, testy i konfiguracja powstają lokalnie w tym folderze; kiedy appka
będzie kompletna i przetestowana lokalnie, robimy **jeden commit**, Ty zakładasz puste repo
`raman-spectroscopy-analyzer` na GitHubie, ja przygotowuję `git remote add` + commit gotowy do
pushnięcia (push wykonujesz Ty ręcznie — to środowisko nie ma Twoich danych logowania GitHub).
Deploy na Streamlit Community Cloud robimy na końcu, raz, świadomie ustawiając Python 3.11 w
Advanced Settings.

## 10. Założenia przyjęte bez pytania (możesz je zakwestionować)

- Format przykładowych i wgrywanych plików: identyczny jak poprzednio (dwie kolumny, spacja/tab)
- Wybór przykładowego widma: pojedynczy wybór (nie multi-select) — upload nadal wspiera tryb wielokrotny
- Nazwa repo od razu poprawna (`raman-spectroscopy-analyzer`), bez etapu zmiany nazwy jak poprzednio
