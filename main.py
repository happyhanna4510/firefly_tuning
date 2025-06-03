import argparse
import warnings
import time

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from data.dataset import load_and_preprocess_data
from models.mlp import create_model
from utils.io import save_results
from utils.plotting import plot_predictions, plot_convergence
from algorithms import pso

from algorithms.firefly import run_firefly
from utils.compare import compare_results

from utils.save import save_all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='pso', help='Wybierz algorytm: pso, ga, de')
    args = parser.parse_args()

    # Dane
    X_train, X_test, y_train, y_test, scaler_y = load_and_preprocess_data()

    # Model
    model = create_model(X_train.shape[1])


    results_all = {}
    selected_algorithms = []
    # tu wybiera sie algorytm w zależnosci od tego co napiszemy w terminale pso|firefly lub all (both)
    if args.algorithm in ["pso", "all"]:
        selected_algorithms.append("pso")

    if args.algorithm in ["firefly", "all"]:
        selected_algorithms.append("firefly")

    for algo in selected_algorithms:
        print(f"\nUruchamianie {algo.upper()}...")
        start = time.time()
        
        if algo == "pso":
            result = pso.run(model, X_train, y_train, X_test, y_test, scaler_y)
        elif algo == "firefly":
            result = run_firefly(model, X_train, y_train, X_test, y_test, scaler_y)
        
        duration = time.time() - start
        result['duration'] = duration
        results_all[algo] = result

        save_results(result, algorithm=algo)
        plot_convergence(result['history'], algorithm=algo)
        plot_predictions(result, algorithm=algo)

    # porównanie tylko gdy uruchomione oba
    if args.algorithm == "all":
        compare_results(results_all)


if __name__ == "__main__":
    main()
