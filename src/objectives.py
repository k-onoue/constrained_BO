import numpy as np
from utils_warcraft import navigate_through_matrix, manhattan_distance


class BaseObjective:
    def __init__(self):
        raise NotImplementedError("Subclasses should implement this method")

    def sample(self, trial):
        raise NotImplementedError("Subclasses should implement this method")

    def evaluate(self):
        raise NotImplementedError("Subclasses should implement this method")

    def __call__(self, trial):
        raise NotImplementedError("Subclasses should implement this method")
    

class WarcraftObjective(BaseObjective):
    def __init__(self, weight_matrix: np.ndarray):
        self.map_shape = weight_matrix.shape
        self.weights = weight_matrix / weight_matrix.sum() # normalize weights

    def sample(self, trial):
        directions = ['oo', 'ab', 'ac', 'ad', 'bc', 'bd', 'cd']  # search space
        direction_matrix = np.zeros(self.map_shape, dtype=object)
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                direction_matrix[i, j] = trial.suggest_categorical(f'({i},{j})', directions)
        self.direction_matrix = direction_matrix

    def evaluate(self):
        start = (0, 0)
        goal = (self.map_shape[0] - 1, self.map_shape[1] - 1)
        history = navigate_through_matrix(self.direction_matrix, start, goal)

        if history:
            path_weight = sum(self.weights[coord] for coord in history)
            norm_const = manhattan_distance(start, goal)
            loss1 = 1 - (1 - manhattan_distance(history[-1], goal) / norm_const) + path_weight
        else:
            loss1 = 1

        mask = self.direction_matrix != 'oo'
        loss2 = self.weights[mask].sum()
        
        return loss1 + loss2

    def __call__(self, trial):
        self.sample(trial)
        print(f'direction matrix:\n{self.direction_matrix}\n\n')
        return self.evaluate()
    



if __name__ == '__main__':
    import numpy as np
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    # optuna.logging.set_verbosity(optuna.logging.INFO)

    weight_matrix = np.array([
        [0.1, 0.4, 0.9],
        [0.4, 0.1, 0.4],
        [0.9, 0.4, 0.1]
    ])

    objective = WarcraftObjective(weight_matrix)

    n_trials = 10000
    seed = 42 
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)
    print(study.best_value)