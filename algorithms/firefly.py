import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.mlp import apply_weights

from utils.save import save_all_outputs


def firefly_algorithm(fitness_func, dim, n_fireflies=30, max_iter=100, alpha=0.3, beta0=1.0, gamma=0.1):
    fireflies = np.random.uniform(-1, 1, (n_fireflies, dim))
    scores = np.array([fitness_func(f) for f in fireflies])
    best_idx = np.argmin(scores)
    best_solution = fireflies[best_idx].copy()
    best_score = scores[best_idx]
    history = [best_score]

    for _ in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if scores[j] < scores[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(dim) - 0.5)
                    scores[i] = fitness_func(fireflies[i])
                    if scores[i] < best_score:
                        best_score = scores[i]
                        best_solution = fireflies[i].copy()
        history.append(best_score)
    return best_solution, best_score, history

def run_firefly(model, X_train, y_train, X_test, y_test, scaler_y, n_fireflies=30, max_iter=100):
    os.makedirs("results", exist_ok=True)
    
    dim = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)

    def fitness(position):
        temp_model = apply_weights(model, position.copy())
        pred = temp_model.predict(X_train)
        return mean_squared_error(y_train, pred)

    print(f"[INFO] Firefly Algorithm running for {max_iter} iterations...")
    best_weights, best_score, history = firefly_algorithm(
        fitness_func=fitness,
        dim=dim,
        n_fireflies=n_fireflies,
        max_iter=max_iter
    )

    final_model = apply_weights(model, best_weights)
    y_pred = final_model.predict(X_test)

    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }

    result = {
        'model': final_model,
        'history': history,
        'metrics': metrics,
        'predictions': y_pred,
        'y_true': y_test,
        'weights': best_weights
    }

    save_all_outputs(result, output_dir="results/firefly", algorithm_name="Firefly", extra_config={
        "Liczba świetlików": n_fireflies,
        "Maks. iteracji": max_iter,
        "Wymiar przestrzeni": dim
    })


    return {
        'model': final_model,
        'history': history,
        'metrics': metrics,
        'predictions': y_pred,
        'y_true': y_test
    }
