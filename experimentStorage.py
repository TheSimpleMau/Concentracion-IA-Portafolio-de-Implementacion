# ============================================================
# experimentStorage.py
# ============================================================

import os
import pickle
from datetime import datetime


RESULTS_DIRECTORY = "resultados"

def save_experiment(experiment_data, directory=RESULTS_DIRECTORY):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(directory, f"experimento_{timestamp}.pkl")
    experiment_data = {"timestamp": timestamp, **experiment_data}

    with open(filename,"wb") as file:
        pickle.dump(experiment_data,file,protocol=pickle.HIGHEST_PROTOCOL)

    print("\nExperimento guardado en:")
    print(filename)
    return filename


def load_experiment(filename):
    with open(filename,"rb") as file:
        return pickle.load(file)