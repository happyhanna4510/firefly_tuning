import os
import pandas as pd

def save_results(result, algorithm='pso'):
    os.makedirs('results', exist_ok=True)
    
    metrics = result['metrics']

    text = f"""GŁÓWNY {algorithm.upper()}:
Najlepszy MSE (trening): {result['history'][-1]:.4f}
MSE (test): {metrics['mse']:.4f}
MAE : {metrics['mae']:.4f} | RMSE : {metrics['rmse']:.4f} | R² : {metrics['r2']:.4f}"""
    
    print(text)
    
    # folder robimy po nazwie algorytmu jesli nie ma go
    output_dir = f"results/{algorithm}"
    os.makedirs(output_dir, exist_ok=True)

    # save in results/pso/summary.txt, results/firefly/summary.txt ...
    with open(os.path.join(output_dir, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write(text)

