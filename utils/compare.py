import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_results(results_all):
    print("\nPORÓWNANIE ALGORYTMÓW:")

    os.makedirs("results/compare", exist_ok=True)
    comparison_data = {}
    rankings = {}

    # --- METRYKI + TIME ---
    for algo, result in results_all.items():
        metrics = result['metrics']
        comparison_data[algo] = {
            'MSE': metrics['mse'],
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R2': metrics['r2'],
            'Czas [s]': result.get('duration', 0)
        }

    df = pd.DataFrame(comparison_data).T
    df.to_csv("results/compare/comparison_metrics.csv", float_format="%.5f", index_label="Algorytm")
    print("Zapisano: results/compare/comparison_metrics.csv")

    # --- RANKING (!!!!!!!! ale jeżeli mamy tylko 2 to nie ma sensu chyba i można delete to) ---
    df_rank = df.copy()
    df_rank['RANK_MSE'] = df_rank['MSE'].rank(method='min')
    df_rank['RANK_MAE'] = df_rank['MAE'].rank(method='min')
    df_rank['RANK_RMSE'] = df_rank['RMSE'].rank(method='min')
    df_rank['RANK_R2'] = df_rank['R2'].rank(ascending=False, method='min')
    df_rank['RANK_CZAS'] = df_rank['Czas [s]'].rank(method='min')
    df_rank.to_csv("results/compare/comparison_ranks.csv", float_format="%.2f", index_label="Algorytm")
    print("Zapisano: results/compare/comparison_ranks.csv")

    # --- WYKRES różnic metryk ---
    df.T.plot(kind='bar', figsize=(10, 6))
    plt.title("Porównanie metryk (niższe = lepsze, R² wyższy)")
    plt.ylabel("Wartość")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/compare/metric_comparison_plot.png")
    plt.close()
    print("Zapisano: results/compare/metric_comparison_plot.png")

    # --- ZBIEŻNOŚĆ ---
    plt.figure()
    for algo, result in results_all.items():
        plt.plot(result['history'], label=algo.upper())
    plt.xlabel("Iteracja")
    plt.ylabel("MSE")
    plt.title("Porównanie zbieżności PSO vs Firefly")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/compare/convergence_comparison.png")
    plt.close()
    print("Zapisano: results/compare/convergence_comparison.png")
