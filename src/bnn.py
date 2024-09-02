import torch
import torch.nn as nn
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from torch.distributions.normal import Normal


# Bayesian Linear Regression class
class BayesianLinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(BayesianLinearRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Parameters for prior distributions of weights and biases
        self.w_mu = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.w_log_sigma = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.b_mu = nn.Parameter(torch.zeros(output_dim))
        self.b_log_sigma = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_sigma = torch.exp(self.w_log_sigma)
        b_sigma = torch.exp(self.b_log_sigma)

        # Sample weights and biases
        w = self.w_mu + w_sigma * torch.randn_like(self.w_mu)
        b = self.b_mu + b_sigma * torch.randn_like(self.b_mu)

        return torch.matmul(x, w) + b

    def predict_dist(self, x: torch.Tensor) -> Normal:
        y = self.forward(x)

        # Compute uncertainty in the output
        w_sigma = torch.exp(self.w_log_sigma)
        b_sigma = torch.exp(self.b_log_sigma)

        # Calculate the standard deviation considering the uncertainty in weights and biases
        output_sigma = torch.sqrt(torch.matmul(x**2, w_sigma**2) + b_sigma**2)

        return Normal(y, output_sigma)


# Bayesian MLP class with adjustable hidden units and layers
class BayesianMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_unit_size: int = 64,
        num_hidden_layers: int = 3,
        activation_fn: nn.Module = nn.ReLU(),
        min_val: float = None,
        max_val: float = None,
        clipping: bool = False,
    ) -> None:
        super(BayesianMLP, self).__init__()

        # Handle both single activation function and list of functions
        if isinstance(activation_fn, list):
            assert len(activation_fn) == num_hidden_layers, (
                f"Number of activation functions ({len(activation_fn)}) does not match "
                f"the number of hidden layers ({num_hidden_layers})."
            )
            activations = activation_fn
        else:
            activations = [activation_fn] * num_hidden_layers

        layers = []
        layers.append(nn.Linear(input_dim, hidden_unit_size))
        layers.append(activations[0])

        for i in range(1, num_hidden_layers):
            layers.append(nn.Linear(hidden_unit_size, hidden_unit_size))
            layers.append(activations[i])

        self.hidden_layers = nn.Sequential(*layers)
        self.bayesian_output = BayesianLinearRegression(hidden_unit_size, 1)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> Normal:
        x = self.hidden_layers(x)

        # Get output from Bayesian linear regression
        y_dist = self.bayesian_output.predict_dist(x)

        if self.min_val or self.max_val:
            y_mean = torch.clamp(y_dist.mean, min=self.min_val, max=self.max_val)
        else:
            y_mean = y_dist.mean

        y_stddev = y_dist.stddev

        return Normal(y_mean, y_stddev)


# Model class using Bayesian MLP with adjustable hidden units and layers
class BayesianMLPModel(Model):
    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        hidden_unit_size: int = 64,
        num_hidden_layers: int = 3,
        activation_fn: nn.Module = nn.ReLU(),
        min_val: float = None,
        max_val: float = None,
        clipping: bool = False,
    ) -> None:
        super().__init__()
        self.bayesian_mlp = BayesianMLP(
            input_dim=train_X.shape[1],
            hidden_unit_size=hidden_unit_size,
            num_hidden_layers=num_hidden_layers,
            activation_fn=activation_fn,
            min_val=min_val,
            max_val=max_val,
            clipping=clipping,
        )
        self.likelihood = GaussianLikelihood()
        self._num_outputs = 1
        self._train_inputs = train_X.to(
            train_X.device
        )  # Ensure it's on the right device
        self._train_targets = train_Y.to(
            train_Y.device
        )  # Ensure it's on the right device

    def forward(self, x: torch.Tensor) -> Normal:
        return self.bayesian_mlp(x.to(x.device))

    def posterior(
        self, X: torch.Tensor, observation_noise=False, **kwargs
    ) -> MultivariateNormal:
        pred_dist = self.bayesian_mlp(X.to(X.device))
        mean = pred_dist.mean.squeeze(-1)  # Ensure mean is 2D
        stddev = pred_dist.stddev.squeeze(-1)  # Ensure stddev is 2D
        covar = torch.diag_embed(stddev**2)
        return MultivariateNormal(mean, covar)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def train_inputs(self) -> torch.Tensor:
        return self._train_inputs

    @property
    def train_targets(self) -> torch.Tensor:
        return self._train_targets

    def set_train_data(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> None:
        self._train_inputs = inputs
        self._train_targets = targets
