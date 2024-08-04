import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from scipy.optimize import minimize
from bnn import BayesianMLPModel, fit_pytorch_model
from utils_experiment import generate_integer_samples


"""
BOループの終了条件を少し工夫しないと，ある（局所）最適解に達している場合に，
獲得関数のパラメータの更新を永遠と行って実験が終了しない可能性がある
"""



class Experiment:
    def __init__(self, config):
        self.config = config
        self.bounds = config["bounds"]
        self.objective_function = config["objective_function"]
        self.train_x, self.train_y = self.generate_initial_data(config["initial_points"])
        self.model = self.initialize_model(self.train_x, self.train_y)
        self.best_values = []  

        self.beta = config["algo_params"].get("beta", 2.0)
        self.beta_h = config["algo_params"].get("beta_h", 10.0)
        self.l_h = config["algo_params"].get("l_h", 2.0)

    def generate_initial_data(self, n):
        print(f'bounds: {self.bounds}')
        train_x = generate_integer_samples(self.bounds, n).float()
        train_y = self.objective_function(train_x).unsqueeze(-1)
        return train_x, train_y

    def initialize_model(self, train_x, train_y):
        model = BayesianMLPModel(train_x, train_y)
        return model

    def acquisition_function(self, beta):
        return UpperConfidenceBound(self.model, beta=beta)

    def optimize_acquisition(self, acq_function):
        try:
            candidates, _ = optimize_acqf(
                acq_function,
                bounds=self.bounds,
                q=1,
                num_restarts=self.config["num_restarts"],
                raw_samples=self.config["raw_samples"],
            )
        except RuntimeError as e:
            print(f"RuntimeError during acquisition optimization: {e}")
            return None

        if torch.isnan(candidates).any() or torch.isinf(candidates).any():
            print("Warning: Candidates contain NaN or Inf values")
            return None

        return candidates.detach()

    def adjust_beta(self):
        def objective(params):
            delta_beta = params[0]
            adjusted_beta = self.beta + delta_beta
            acq_function = self.acquisition_function(adjusted_beta)
            new_x = self.optimize_acquisition(acq_function)
            if new_x is None:
                return float('inf')
            rounded_new_x = torch.round(new_x)

            penalty = float('inf')
            if (rounded_new_x == self.train_x).all(dim=1).any():
                penalty = 1000

            return delta_beta + torch.norm(new_x - rounded_new_x).item() + penalty

        initial_guess = [0.0]
        bounds = [(0.0, self.beta_h)]

        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        delta_beta = result.x[0]
        self.beta += delta_beta

    def get_new_candidate(self):
        acq_function = self.acquisition_function(self.beta)
        new_x = self.optimize_acquisition(acq_function)
        if new_x is None:
            return None

        new_x = torch.round(new_x)

        while (new_x == self.train_x).all(dim=1).any():
            self.adjust_beta()
            acq_function = self.acquisition_function(self.beta)
            new_x = self.optimize_acquisition(acq_function)
            if new_x is None:
                return None
            new_x = torch.round(new_x)

        return new_x

    def optimize_acqf_and_get_observation(self):
        new_x = self.get_new_candidate()
        if new_x is None:
            return None, None

        new_y = self.objective_function(new_x).unsqueeze(-1)
        return new_x, new_y

    def run(self):
        for iteration in range(1, self.config["n_iterations"] + 1):

            fit_pytorch_model(self.model)
            
            new_x, new_y = self.optimize_acqf_and_get_observation()
            if new_x is None or new_y is None:
                print("Stopping optimization due to numerical issues.")
                break
            
            self.train_x = torch.cat([self.train_x, new_x])
            self.train_y = torch.cat([self.train_y, new_y])
            self.model = self.initialize_model(self.train_x, self.train_y)
            
            best_value = self.train_y.max().item()
            self.best_values.append(best_value)

            print(f"Iteration {iteration}/{self.config['n_iterations']}: Best value = {best_value}")

        print("All done.")




if __name__ == "__main__":
    import warnings
    import plotly.graph_objects as go

    warnings.filterwarnings('ignore')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    def objective_function(X):
        return -((X - 2) ** 2).sum(dim=-1)

    experiment_config = {
        "initial_points": 5,
        "bounds": torch.tensor([[0.0, 0.0], [4.0, 4.0]], device=device, dtype=dtype),
        "batch_size": 1,
        "num_restarts": 10,
        "raw_samples": 20,
        "n_iterations": 5,
        "objective_function": objective_function,
        "algo_params": {
            "beta": 2.0,
            "beta_h": 10.0,
            "l_h": 2.0
        }
    }

    experiment = Experiment(experiment_config)
    experiment.run()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(experiment.best_values) + 1)),
        y=experiment.best_values,
        mode='lines+markers',
        name='Best Objective Value'
    ))

    fig.update_layout(
        title='Optimization History Plot',
        xaxis_title='Iteration',
        yaxis_title='Best Objective Value'
    )

    fig.show()