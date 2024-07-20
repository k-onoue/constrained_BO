import numpy as np
import optuna
from objectives import WarcraftObjective


if __name__ == '__main__':

    weight_matrix = np.array([
        [0.1, 0.4, 0.9],
        [0.4, 0.1, 0.4],
        [0.9, 0.4, 0.1]
    ])

    objective = WarcraftObjective(weight_matrix)

    n_trials = 100
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)
    print(study.best_value)