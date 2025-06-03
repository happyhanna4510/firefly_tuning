import os
import json
import numpy as np
import pandas as pd

def save_all_outputs(result_dict, output_dir, algorithm_name, extra_config=None):
    os.makedirs(output_dir, exist_ok=True)

    #  predictions
    np.savetxt(os.path.join(output_dir, "y_test.csv"), result_dict['y_true'], delimiter=",")
    np.savetxt(os.path.join(output_dir, "y_pred.csv"), result_dict['predictions'], delimiter=",")

    # history
    pd.DataFrame(result_dict['history'], columns=["MSE"]).to_csv(
        os.path.join(output_dir, "history.csv"), index_label="iteration")

    # best weights 
    if 'weights' in result_dict:
        np.savetxt(os.path.join(output_dir, "best_weights.csv"), result_dict['weights'], delimiter=",")

    # metrics
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result_dict['metrics'], f, indent=4)

    # config
    with open(os.path.join(output_dir, "config.txt"), "w", encoding="utf-8") as f:
        if extra_config:
            for k, v in extra_config.items():
                f.write(f"{k}: {v}\n")
        f.write(f"Algorytm: {algorithm_name}\n")
        f.write(f"Najlepszy MSE (trening): {result_dict['history'][-1]:.5f}\n")

    return output_dir
