import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from bnn import BayesianMLPModel


class DiscreteBO:
    def __init__(self, bounds, beta=2.0, beta_h=10.0, l_h=2.0):
        self.bounds = bounds
        self.beta = beta
        self.beta_h = beta_h
        self.l_h = l_h
        self.l = 1.0

    def initialize(self, X_init, Y_init):
        self.train_X = X_init.float()
        self.train_Y = Y_init.float()
        self.model = self._fit_model()

    def _fit_model(self):
        # Define and fit the model
        model = BayesianMLPModel(input_dim=self.train_X.shape[1])
        model.set_train_data(self.train_X, self.train_Y)
        model = train_model(model, self.train_X, self.train_Y)
        return model

    def acquisition_function(self, beta):
        return UpperConfidenceBound(self.model, beta=beta)

    def optimize_acquisition(self, acq_function):
        try:
            candidates, _ = optimize_acqf(
                acq_function,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
            )
        except RuntimeError as e:
            print(f"RuntimeError during acquisition optimization: {e}")
            return None

        # 候補の中に inf や nan がないかチェック
        if torch.isnan(candidates).any() or torch.isinf(candidates).any():
            print("Warning: Candidates contain NaN or Inf values")
            return None
        
        return candidates

    def step(self):
        acq_function = self.acquisition_function(self.beta)
        new_X = self.optimize_acquisition(acq_function)
        if new_X is None:
            return None

        new_X = torch.round(new_X)

        # Check if new_X is in train_X and adjust beta and l if necessary
        while (new_X == self.train_X).all(dim=1).any():
            self.adjust_beta_and_l()
            acq_function = self.acquisition_function(self.beta)
            new_X = self.optimize_acquisition(acq_function)
            if new_X is None:
                return None
            new_X = torch.round(new_X)

        return new_X

    def update(self, new_X, new_Y):
        self.train_X = torch.cat([self.train_X, new_X.float()], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y.float()], dim=0)
        self.model = self._fit_model()

    def adjust_beta_and_l(self):
        # Define the optimization problem to adjust beta and l
        def objective(params):
            delta_beta, l = params
            adjusted_beta = self.beta + delta_beta
            acq_function = self.acquisition_function(adjusted_beta)
            new_X = self.optimize_acquisition(acq_function)
            if new_X is None:
                return float('inf')
            rounded_new_X = torch.round(new_X)

            penalty = float('inf')
            if (rounded_new_X == self.train_X).all(dim=1).any():
                penalty = 1000  # Arbitrary high value to avoid repetition

            return delta_beta + torch.norm(new_X - rounded_new_X).item() + penalty

        # Initial guess and bounds for delta_beta and l
        initial_guess = torch.tensor([0.0, self.l])
        bounds = [(0.0, self.beta_h), (1e-3, self.l_h)]

        # Optimize the objective function
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        delta_beta, l = result.x
        self.beta += delta_beta
        self.l = l


def train_model(model, train_X, train_Y, num_epochs=1000, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output_dist = model(train_X)
        loss = -output_dist.log_prob(train_Y).mean()  # Minimize negative log likelihood
        loss.backward()
        optimizer.step()
    return model


# Run the test
if __name__ == "__main__":
    # Griewank function with batch input
    def griewank_function(X):
        r"""
        f(x) = \sum_{i=1}^d \frac{x_i^2}{4000} - \prod_{i=1}^d cos \left( \frac{x_i}{\sqrt{i}} \right) + 1
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)  # Make it a 2D tensor with shape (1, dim)

        sum_term = torch.sum(X**2 / 4000, dim=1)
        prod_term = torch.prod(torch.cos(X / torch.sqrt(torch.arange(1, X.shape[1] + 1).float())), dim=1)
        return sum_term - prod_term + 1

    objective_function = griewank_function

    # Initialize bounds for the optimization
    bounds = torch.tensor([[-50.0, -50.0, -50.0], [600.0, 600.0, 600.0]])

    # Generate initial samples
    X_init = torch.randint(-50, 601, (5, 3)).float()  # float に変換
    Y_init = objective_function(X_init).unsqueeze(-1).float()  # float に変換

    # DiscreteBO のインスタンス化と初期化
    bo = DiscreteBO(bounds)
    bo.initialize(X_init, Y_init)

    # Optimization loop
    for i in range(1000):
        new_X = bo.step()
        if new_X is None:
            print("Stopping optimization due to numerical issues.")
            break
        new_Y = objective_function(new_X).unsqueeze(-1).float()  # float に変換

        bo.update(new_X, new_Y)
        print(f"Iteration {i+1}: Suggested point: {new_X.numpy()}, Function value: {new_Y.numpy()}")

    print()
    print(f"Best value found: {bo.train_Y.min().item()}")
    print(f"Best point found: {bo.train_X[bo.train_Y.argmin()].numpy()}")
    print()

    print(bo.train_X)
