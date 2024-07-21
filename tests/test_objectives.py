import sys
from constants import PATH_INFO
sys.path.append(PATH_INFO.get('project_dir'))

import numpy as np
import optuna
from src.objectives import WarcraftObjective
from src.objectives import AckleyObjective
from src.objectives import RosenbrockObjective
from src.objectives import DiscreteAckleyObjective
from src.objectives import DiscreteRosenbrockObjective


if __name__ == '__main__':
    # optuna.logging.set_verbosity(optuna.logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    weight_matrix = np.array([
        [0.1, 0.4, 0.9],
        [0.4, 0.1, 0.4],
        [0.9, 0.4, 0.1]
    ])

    objective = WarcraftObjective(weight_matrix)

    n_trials = 1000
    seed = 42 
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)
    print(study.best_value)


# if __name__ == '__main__':
#     # optuna.logging.set_verbosity(optuna.logging.ERROR)
#     optuna.logging.set_verbosity(optuna.logging.INFO)

#     # Define the objective function
#     # objective = AckleyObjective(dim=2)
#     # objective = DiscreteAckleyObjective(dim=2, n_split=30)
#     # objective = RosenbrockObjective(dim=2)
#     objective = DiscreteRosenbrockObjective(dim=2, n_split=30)

#     n_trials = 100
#     seed = 42 
#     sampler = optuna.samplers.TPESampler(seed=seed)

#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     study.optimize(objective, n_trials=n_trials)

#     print(study.best_params)
#     print(study.best_value)

#     # Plot optimization trajectory
#     objective.plot_optimization_trajectory(study)