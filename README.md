# Metaheuristic Neural Network Tuning

Projekt demonstruje wykorzystanie metaheurystyk do optymalizacji wag sieci MLP. Implementuje dwie metody inspirowane naturą:

- **PSO** – Particle Swarm Optimization
- **Firefly Algorithm**

Repozytorium pozwala uruchomić pojedynczy algorytm lub oba jednocześnie w celu późniejszego porównania wyników.

## Struktura repozytorium

```
algorithms/   # implementacje PSO i Firefly
models/       # prosty model MLP wraz z funkcją nadpisywania wag
data/         # generowanie przykładowego zbioru danych (make_regression)
utils/        # funkcje zapisu wyników i rysowania wykresów
main.py       # punkt wejścia, obsługa CLI
```

## Instalacja

Wymagany jest Python >=3.8. Zainstaluj zależności poleceniem:

```bash
pip install -r requirements.txt
```

## Uruchomienie

Domyślnie uruchamiany jest PSO:

```bash
python main.py --algorithm pso
```

Możliwe wartości argumentu `--algorithm`:

- `pso` – tylko Particle Swarm Optimization
- `firefly` – tylko algorytm świetlików
- `all` – oba algorytmy i automatyczne porównanie

## Wyniki

Dla każdego algorytmu tworzony jest podkatalog w `results/` zawierający m.in.:

- `summary.txt` – krótkie podsumowanie przebiegu
- `history.csv` – MSE w kolejnych iteracjach
- `metrics.json` – zbiorcze metryki (MSE, MAE, RMSE, R²)
- `predictions.png` – wykres przewidywań
- `convergence.png` – zbieżność w czasie

Przy uruchomieniu obu metod tworzony jest dodatkowo folder `results/compare/` z porównaniem metryk i wspólnym wykresem zbieżności.

## Dalsze modyfikacje

Parametry takie jak liczba cząstek/świetlików czy liczba iteracji można zmienić bezpośrednio w plikach `algorithms/pso.py` i `algorithms/firefly.py`. Dane treningowe generowane są syntetycznie w `data/dataset.py` i można je zastąpić własnym zbiorem.

