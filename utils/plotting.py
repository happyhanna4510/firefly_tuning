import matplotlib.pyplot as plt
import os

def plot_predictions(result, algorithm='pso'):
    plt.figure()
    plt.scatter(result['y_true'], result['predictions'], label='Predykcja vs Rzeczywistość')
    plt.plot([min(result['y_true']), max(result['y_true'])],
             [min(result['y_true']), max(result['y_true'])], 'r--')
    plt.xlabel('Rzeczywista')
    plt.ylabel('Predykowana')
    plt.title('Predykcja vs Rzeczywista')
    plt.legend()
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)

    # do folderu pso lub firefly
    outdir = f"results/{algorithm}"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/predictions.png")
    plt.close()

def plot_convergence(history, algorithm='pso'):
    plt.figure()
    plt.plot(history)
    plt.xlabel('Iteracja')
    plt.ylabel('MSE')
    plt.title('Zbieżność algorytmu')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    
    outdir = f"results/{algorithm}"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/_convergence.png")
    plt.close()
