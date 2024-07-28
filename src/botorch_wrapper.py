import torch
from abc import ABC, abstractmethod
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from scipy.optimize import minimize



class BaseBO(ABC):
    def __init__(self, bounds):
        self.bounds = bounds
        self.train_X = None
        self.train_Y = None
        self.model = None

    @abstractmethod
    def initialize(self, X_init, Y_init):
        pass

    @abstractmethod
    def _fit_model(self):
        pass

    @abstractmethod
    def acquisition_function(self):
        pass

    @abstractmethod
    def optimize_acquisition(self, acq_function):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def update(self, new_X, new_Y):
        pass


class BO(BaseBO):
    def __init__(self, bounds, beta=2.0):
        super().__init__(bounds)
        self.beta = beta

    def initialize(self, X_init, Y_init):
        self.train_X = X_init
        self.train_Y = Y_init
        self.model = self._fit_model()

    def _fit_model(self):
        # from botorch.models import SingleTaskGP
        # from gpytorch.mlls import ExactMarginalLogLikelihood
        # from botorch.fit import fit_gpytorch_mll
        
        model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def acquisition_function(self):
        # from botorch.acquisition import UpperConfidenceBound
        
        UCB = UpperConfidenceBound(self.model, beta=self.beta)
        return UCB

    def optimize_acquisition(self, acq_function):
        # from botorch.optim import optimize_acqf
        
        candidates, _ = optimize_acqf(
            acq_function,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=20,
        )
        return candidates

    def step(self):
        acq_function = self.acquisition_function()
        new_X = self.optimize_acquisition(acq_function)
        return new_X

    def update(self, new_X, new_Y):
        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)
        self.model = self._fit_model()


class DiscreteBO(BaseBO):
    def __init__(self, bounds, beta=2.0, beta_h=10.0, l_h=2.0):
        super().__init__(bounds)
        self.beta = beta
        self.beta_h = beta_h
        self.l_h = l_h
        self.l = 1.0

    def initialize(self, X_init, Y_init):
        self.train_X = X_init
        self.train_Y = Y_init
        self.model = self._fit_model()

    def _fit_model(self):
        model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def acquisition_function(self, beta):
        return UpperConfidenceBound(self.model, beta=beta)

    def optimize_acquisition(self, acq_function):
        candidates, _ = optimize_acqf(
            acq_function,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=20,
        )
        return candidates

    def step(self):
        acq_function = self.acquisition_function(self.beta)
        new_X = self.optimize_acquisition(acq_function)
        return new_X

    def update(self, new_X, new_Y):
        self.train_X = torch.cat([self.train_X, new_X], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)
        self.model = self._fit_model()

    def adjust_beta_and_l(self):
        # Define the optimization problem to adjust beta and l
        def objective(params):
            delta_beta, l = params
            adjusted_beta = self.beta + delta_beta
            acq_function = self.acquisition_function(adjusted_beta)
            new_X = self.optimize_acquisition(acq_function)
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



# Griewank function with batch input
def griewank_function(X):
    r"""
    f(x) = \sum_{i=1}^d \frac{x_i^2}{4000} - \prod_{i=1}^d \cos \left( \frac{x_i}{\sqrt{i}} \right) + 1
    """
    if X.dim() == 1:
        X = X.unsqueeze(0)  # Make it a 2D tensor with shape (1, dim)

    sum_term = torch.sum(X**2 / 4000, dim=1)
    prod_term = torch.prod(torch.cos(X / torch.sqrt(torch.arange(1, X.shape[1] + 1).float())), dim=1)
    return sum_term - prod_term + 1


# Run the test
if __name__ == "__main__":
    # Initialize bounds for the optimization
    bounds = torch.tensor([[-50.0, -50.0, -50.0], [600.0, 600.0, 600.0]])
    
    # Generate initial samples
    X_init = torch.randint(-50, 601, (5, 3)).double()
    Y_init = griewank_function(X_init).unsqueeze(-1)

    # Create an instance of DiscreteBO and initialize it
    bo = DiscreteBO(bounds=bounds)
    bo.initialize(X_init, Y_init)
    
    # Optimization loop
    for i in range(300):
        new_X = bo.step()
        if (new_X == bo.train_X).all(dim=1).any():
            bo.adjust_beta_and_l()
            new_X = bo.step()
        new_Y = griewank_function(new_X).unsqueeze(-1)
        
        bo.update(new_X, new_Y)
        print(f"Iteration {i+1}: Suggested point: {new_X.numpy()}, Function value: {new_Y.numpy()}")

# # Run the test
# if __name__ == "__main__":
#     # Initialize bounds for the optimization
#     bounds = torch.tensor([[-50.0, -50.0, -50.0], [600.0, 600.0, 600.0]])
    
#     # Generate initial samples
#     X_init = torch.randint(-50, 601, (5, 3)).double()
#     Y_init = griewank_function(X_init).unsqueeze(-1)

#     # Create an instance of BO and initialize it
#     bo = BO(bounds=bounds)
#     bo.initialize(X_init, Y_init)
    
#     # Optimization loop
#     for i in range(300):
#         new_X = bo.step()
#         new_Y = griewank_function(new_X).unsqueeze(-1)
        
#         bo.update(new_X, new_Y)
#         print(f"Iteration {i+1}: Suggested point: {new_X.numpy()}, Function value: {new_Y.numpy()}")