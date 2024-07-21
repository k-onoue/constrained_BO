import numpy as np
import optuna
import plotly.graph_objects as go
from utils_warcraft import navigate_through_matrix, manhattan_distance
from utils_benchmark import ackley_function, rosenbrock_function, discretize_function


class BaseObjective:
    def __init__(self):
        raise NotImplementedError("Subclasses should implement this method")

    def sample(self, trial):
        raise NotImplementedError("Subclasses should implement this method")

    def evaluate(self, sample):
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
        return direction_matrix

    def evaluate(self, sample):
        direction_matrix = sample
        start = (0, 0)
        goal = (self.map_shape[0] - 1, self.map_shape[1] - 1)
        history = navigate_through_matrix(direction_matrix, start, goal)

        if history:
            path_weight = sum(self.weights[coord] for coord in history)
            norm_const = manhattan_distance(start, goal)
            loss1 = 1 - (1 - manhattan_distance(history[-1], goal) / norm_const) + path_weight
        else:
            loss1 = 1

        mask = direction_matrix != 'oo'
        loss2 = self.weights[mask].sum()
        
        return loss1 + loss2

    def __call__(self, trial):
        sample = self.sample(trial)
        # print(f'direction matrix:\n{sample}\n\n')
        return self.evaluate(sample)
    
class AckleyObjective(BaseObjective):
    def __init__(self, dim: int = 2, search_space: tuple[float, float] = (-1, 1)):
        self.dim = dim
        self.search_space = search_space  # homogeneous for now

    def sample(self, trial):
        return [
            trial.suggest_uniform(
                f'x{i}', 
                self.search_space[0], 
                self.search_space[1]
            ) for i in range(self.dim)
        ]

    def evaluate(self, sample):
        return ackley_function(x=sample)

    def __call__(self, trial):
        sample = self.sample(trial)
        # print(f'x: {sample}\n\n')
        return self.evaluate(sample)

    def plot_optimization_trajectory(self, study: optuna.Study):
        if self.dim != 2:
            print("Plotting is only supported for 2-dimensional problems.")
            return
        
        # Collect parameter history
        x_history = [trial.params['x0'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        y_history = [trial.params['x1'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        z_history = [ackley_function([x, y]) for x, y in zip(x_history, y_history)]

        # Create colors based on the number of trials
        colors = np.linspace(0, 1, len(x_history))
        
        # 2次元のAckley関数をプロット
        x_2d = np.linspace(self.search_space[0], self.search_space[1], 100)
        y_2d = np.linspace(self.search_space[0], self.search_space[1], 100)
        X, Y = np.meshgrid(x_2d, y_2d)
        Z = np.array([[ackley_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.add_trace(go.Scatter3d(x=x_history, y=y_history, z=z_history,
                                   mode='markers+lines', 
                                   marker=dict(size=5, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(x=0)), 
                                   line=dict(color=colors, colorscale='Viridis', showscale=True, colorbar=dict(x=0)),
                                   name='Optimization Trajectory'))
        fig.update_layout(title='2次元のAckley関数と最適化の軌跡',
                          scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'))
        fig.show()


class DiscreteAckleyObjective(AckleyObjective):
    def __init__(self, dim: int = 2, search_space: tuple[float, float] = (-1, 1), n_split: int = 50):
        super().__init__(dim, search_space)
        self.n_split = n_split
        self.discretization_step = (search_space[1] - search_space[0]) / n_split

    def sample(self, trial):
        return [
            trial.suggest_discrete_uniform(
                f'x{i}', 
                self.search_space[0], 
                self.search_space[1], 
                self.discretization_step
            ) for i in range(self.dim)
        ]

    def plot_optimization_trajectory(self, study: optuna.Study):
        if self.dim != 2:
            print("Plotting is only supported for 2-dimensional problems.")
            return
        
        # Collect parameter history
        x_history = [trial.params['x0'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        y_history = [trial.params['x1'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        z_history = [ackley_function([x, y]) for x, y in zip(x_history, y_history)]

        # Create colors based on the number of trials
        colors = np.linspace(0, 1, len(x_history))
        
        # 2次元のAckley関数を離散化してプロット
        x_2d = np.linspace(self.search_space[0], self.search_space[1], self.n_split * 10)
        y_2d = np.linspace(self.search_space[0], self.search_space[1], self.n_split * 10)
        X, Y = np.meshgrid(x_2d, y_2d)
        X_discrete = discretize_function(X, self.discretization_step)
        Y_discrete = discretize_function(Y, self.discretization_step)
        Z = np.array([[ackley_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X_discrete, Y_discrete)])

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.add_trace(go.Scatter3d(x=x_history, y=y_history, z=z_history,
                                   mode='markers+lines', 
                                   marker=dict(size=5, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(x=0)), 
                                   line=dict(color=colors, colorscale='Viridis', showscale=True, colorbar=dict(x=0)),
                                   name='Optimization Trajectory'))
        fig.update_layout(title='2次元の離散化されたAckley関数と最適化の軌跡',
                          scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'))
        fig.show()


class RosenbrockObjective(BaseObjective):
    def __init__(self, dim: int = 2, search_space: tuple[float, float] = (-1, 1)):
        self.dim = dim
        self.search_space = search_space  # homogeneous for now

    def sample(self, trial):
        return [
            trial.suggest_uniform(
                f'x{i}', 
                self.search_space[0], 
                self.search_space[1]
            ) for i in range(self.dim)
        ]

    def evaluate(self, sample):
        return rosenbrock_function(sample)

    def __call__(self, trial):
        sample = self.sample(trial)
        # print(f'x: {sample}\n\n')
        return self.evaluate(np.array(sample))

    def plot_optimization_trajectory(self, study: optuna.Study):
        if self.dim != 2:
            print("Plotting is only supported for 2-dimensional problems.")
            return
        
        # Collect parameter history
        x_history = [trial.params['x0'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        y_history = [trial.params['x1'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        z_history = [rosenbrock_function([x, y]) for x, y in zip(x_history, y_history)]

        # Create colors based on the number of trials
        colors = np.linspace(0, 1, len(x_history))
        
        # 2次元のRosenbrock関数をプロット
        x_2d = np.linspace(self.search_space[0], self.search_space[1], 100)
        y_2d = np.linspace(self.search_space[0], self.search_space[1], 100)
        X, Y = np.meshgrid(x_2d, y_2d)
        Z = np.array([[rosenbrock_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.add_trace(go.Scatter3d(x=x_history, y=y_history, z=z_history,
                                   mode='markers+lines', 
                                   marker=dict(size=5, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(x=0)), 
                                   line=dict(color=colors, colorscale='Viridis', showscale=True, colorbar=dict(x=0)),
                                   name='Optimization Trajectory'))
        fig.update_layout(title='2次元のRosenbrock関数と最適化の軌跡',
                          scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'))
        fig.show()


class DiscreteRosenbrockObjective(RosenbrockObjective):
    def __init__(self, dim: int = 2, search_space: tuple[float, float] = (-1.0, 1.0), n_split: int = 50):
        super().__init__(dim, search_space)
        self.n_split = n_split
        self.discretization_step = (search_space[1] - search_space[0]) / n_split

    def sample(self, trial):
        return [
            trial.suggest_discrete_uniform(
                f'x{i}', 
                self.search_space[0], 
                self.search_space[1], 
                self.discretization_step
            ) for i in range(self.dim)
        ]

    def plot_optimization_trajectory(self, study: optuna.Study):
        if self.dim != 2:
            print("Plotting is only supported for 2-dimensional problems.")
            return
        
        # Collect parameter history
        x_history = [trial.params['x0'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        y_history = [trial.params['x1'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        z_history = [rosenbrock_function([x, y]) for x, y in zip(x_history, y_history)]

        # Create colors based on the number of trials
        colors = np.linspace(0, 1, len(x_history))
        
        # 2次元のRosenbrock関数を離散化してプロット
        x_2d = np.linspace(self.search_space[0], self.search_space[1], self.n_split * 10)
        y_2d = np.linspace(self.search_space[0], self.search_space[1], self.n_split * 10)
        X, Y = np.meshgrid(x_2d, y_2d)
        X_discrete = discretize_function(X, self.discretization_step)
        Y_discrete = discretize_function(Y, self.discretization_step)
        Z = np.array([[rosenbrock_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X_discrete, Y_discrete)])

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.add_trace(go.Scatter3d(x=x_history, y=y_history, z=z_history,
                                   mode='markers+lines', 
                                   marker=dict(size=5, color=colors, colorscale='Viridis', showscale=True, colorbar=dict(x=0)), 
                                   line=dict(color=colors, colorscale='Viridis', showscale=True, colorbar=dict(x=0)),
                                   name='Optimization Trajectory'))
        fig.update_layout(title='2次元の離散化されたRosenbrock関数と最適化の軌跡',
                          scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'))
        fig.show()



# if __name__ == '__main__':
#     import numpy as np
#     import optuna
#     optuna.logging.set_verbosity(optuna.logging.ERROR)
#     # optuna.logging.set_verbosity(optuna.logging.INFO)

#     weight_matrix = np.array([
#         [0.1, 0.4, 0.9],
#         [0.4, 0.1, 0.4],
#         [0.9, 0.4, 0.1]
#     ])

#     objective = WarcraftObjective(weight_matrix)

#     n_trials = 10000
#     seed = 42 
#     sampler = optuna.samplers.TPESampler(seed=seed)

#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     study.optimize(objective, n_trials=n_trials)

#     print(study.best_params)
#     print(study.best_value)


if __name__ == '__main__':
    import numpy as np
    import optuna
    # optuna.logging.set_verbosity(optuna.logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Define the objective function
    # objective = AckleyObjective(dim=2)
    # objective = DiscreteAckleyObjective(dim=2, n_split=30)
    # objective = RosenbrockObjective(dim=2)
    objective = DiscreteRosenbrockObjective(dim=2, n_split=30)

    n_trials = 100
    seed = 42 
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)
    print(study.best_value)

    # Plot optimization trajectory
    objective.plot_optimization_trajectory(study)