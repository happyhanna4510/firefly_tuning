# Metaheuristic-tuning
Temat projektu: Optymalizacja parametrów sieci neuronowej z użyciem metaheurystyk inspirowanych naturą.

Opis zadania: Zadanie polegało na opracowaniu i przebadaniu algorytmu dobierania wag sieci neuronowej MLP wybraną techniką inspirowaną naturą,
np. z listy Evolutionary Computation Bestiary (https://github.com/fcampelo/EC-Bestiary). 

## Zaimplementowane algorytmy

- **PSO** – Particle Swarm Optimization  
- **Firefly Algorithm**

Projekt wspiera zarówno osobne uruchamianie każdego algorytmu, jak i uruchomienie zbiorcze z porównaniem wyników.

---

## Generowane wyniki

Po zakończeniu działania algorytmu zapisywane są automatycznie:

- `best_weights.csv` – najlepsze znalezione wagi
- `config.txt` – konfiguracja eksperymentu
- `history.csv` – przebieg MSE w czasie
- `metrics.json` – dokładne metryki (`MSE`, `MAE`, `RMSE`, `R²`)
- `predictions.png` – wykres przewidywań vs rzeczywistość
- `convergence.png` – wykres zbieżności
- `summary.txt` – podsumowanie wyników
- `y_pred.csv` i `y_test.csv` – dane wyjściowe i rzeczywiste etykiety

Jeśli użyto `--algorithm all`, dodatkowo generowany jest folder `results/compare/` z:

- porównaniem metryk (`comparison_metrics.csv`)
- rankingiem algorytmów (`comparison_ranks.csv`)
- wspólnym wykresem zbieżności i błędów

---

## Instalacja zależności

Upewnij się, że masz zainstalowanego Pythona 3.8+.  
Zainstaluj wszystkie wymagane biblioteki:

```bash
pip install -r requirements.txt
```

## Uruchomienie
Aby uruchomić projekt:

```bash
python main.py --algorithm pso
```
Możliwe wartości --algorithm:
 - pso – uruchomienie tylko PSO
 - firefly – uruchomienie tylko Firefly
 - all – uruchomienie obu i automatyczne porównanie

