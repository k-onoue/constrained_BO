import torch
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from bnn import BayesianMLPModel, fit_pytorch_model
    


class Experiment:
    def __init__(self, config):
        self.config = config
        self.bounds = config["bounds"]
        self.objective_function = config["objective_function"]
        self.train_x, self.train_y = self.generate_initial_data(config["initial_points"])
        self.model = self.initialize_model(self.train_x, self.train_y)
        self.best_values = []  # ベストな評価点リストを保持するためのリスト

    def generate_initial_data(self, n):
        train_x = (self.bounds[1] - self.bounds[0]) * torch.rand(n, self.bounds.size(1), device=device, dtype=dtype) + self.bounds[0]
        train_y = self.objective_function(train_x).unsqueeze(-1)
        return train_x, train_y

    def initialize_model(self, train_x, train_y):
        model = BayesianMLPModel(train_x, train_y)
        return model

    def optimize_acqf_and_get_observation(self, acq_func):
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=self.config["num_restarts"],
            raw_samples=self.config["raw_samples"],
        )
        new_x = candidates.detach()
        new_y = self.objective_function(new_x).unsqueeze(-1)
        return new_x, new_y

    def run(self):
        for iteration in range(1, self.config["n_iterations"] + 1):

            fit_pytorch_model(self.model)
            
            UCB = UpperConfidenceBound(model=self.model, beta=2.0)  # betaの値を設定
            new_x, new_y = self.optimize_acqf_and_get_observation(UCB)
            
            self.train_x = torch.cat([self.train_x, new_x])
            self.train_y = torch.cat([self.train_y, new_y])
            self.model = self.initialize_model(self.train_x, self.train_y)
            
            best_value = self.train_y.max().item()
            self.best_values.append(best_value)

            print(f"Iteration {iteration}/{self.config['n_iterations']}: Best value = {best_value}")

        print(f"All done.")




if __name__ == "__main__":
    import warnings
    import plotly.graph_objects as go

    warnings.filterwarnings('ignore')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    # 多変数の目的関数の定義
    def objective_function(X):
        return -((X - 2) ** 2).sum(dim=-1)

    # 実験の設定を辞書で管理
    experiment_config = {
        "initial_points": 5,
        "bounds": torch.tensor([[0.0, 0.0], [4.0, 4.0]], device=device, dtype=dtype),  # 2次元入力変数の範囲
        "batch_size": 1,
        "num_restarts": 10,
        "raw_samples": 20,
        "n_iterations": 20,
        "objective_function": objective_function
    }

    # 実験の実行
    experiment = Experiment(experiment_config)
    experiment.run()
    # train_x, train_y = experiment.train_x, experiment.train_y
    
    # history plot を plotly で作成
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(experiment.best_values) + 1)),
        y=experiment.best_values,
        mode='lines+markers',
        name='Best Objective Value'
    ))

    fig.update_layout(
        title='Bayesian Optimization History Plot',
        xaxis_title='Iteration',
        yaxis_title='Best Objective Value'
    )

    fig.show()