import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.mlp import apply_weights

from utils.save import save_all_outputs


class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-1, 1, dim)
        self.velocity = np.zeros(dim)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

def run(model, X_train, y_train, X_test, y_test, scaler_y, n_particles=30, max_iter=100):
    output_dir = "results/pso"
    os.makedirs(output_dir, exist_ok=True)

    dim = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    particles = [Particle(dim) for _ in range(n_particles)]
    global_best = particles[0].position.copy()
    global_score = float('inf')
    history = []

    def fitness(position):
        temp_model = apply_weights(model, position.copy())
        pred = temp_model.predict(X_train)
        return mean_squared_error(y_train, pred)

    print(f"[INFO] PSO running for {max_iter} iterations...")
    for i in range(max_iter):
        for p in particles:
            score = fitness(p.position)
            if score < p.best_score:
                p.best_score = score
                p.best_position = p.position.copy()
            if score < global_score:
                global_score = score
                global_best = p.position.copy()
        history.append(global_score)

        for p in particles:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            p.velocity = 0.7 * p.velocity + 1.5 * r1 * (p.best_position - p.position) + 1.5 * r2 * (global_best - p.position)
            p.position += p.velocity

        if i % 10 == 0:
            print(f"[INFO] Iteration {i}/{max_iter} - Best MSE: {global_score:.5f}")

    # Final model
    final_model = apply_weights(model, global_best)
    y_pred = final_model.predict(X_test)


    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }

    # prepare result
    result = {
        'model': final_model,
        'history': history,
        'metrics': metrics,
        'predictions': y_pred,
        'y_true': y_test,
        'weights': global_best
    }

    # save everything
    save_all_outputs(result, output_dir="results/pso", algorithm_name="PSO", extra_config={
        "Liczba cząstek": n_particles,
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
