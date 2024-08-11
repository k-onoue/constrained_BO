import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from botorch.models.model import Model
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal


class BayesianLinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianLinearRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 重みとバイアスの事前分布のパラメータ
        self.w_mu = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.w_log_sigma = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.b_mu = nn.Parameter(torch.zeros(output_dim))
        self.b_log_sigma = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        w_sigma = torch.exp(self.w_log_sigma)
        b_sigma = torch.exp(self.b_log_sigma)
        
        # 重みとバイアスのサンプリング
        w = self.w_mu + w_sigma * torch.randn_like(self.w_mu)
        b = self.b_mu + b_sigma * torch.randn_like(self.b_mu)
        
        return torch.matmul(x, w) + b
    
    def predict_dist(self, x):
        y = self.forward(x)
        
        # 出力の不確実性の計算
        w_sigma = torch.exp(self.w_log_sigma)
        b_sigma = torch.exp(self.b_log_sigma)
        
        # 標準偏差の計算（重みとバイアスの不確実性を考慮）
        output_sigma = torch.sqrt(torch.matmul(x**2, w_sigma**2) + b_sigma**2)
        
        return Normal(y, output_sigma)


class BayesianMLP(nn.Module):
    def __init__(self, input_dim, min_val=None, max_val=None, clipping=False):
        super(BayesianMLP, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.bayesian_output = BayesianLinearRegression(64, 1)
        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        
        # ベイズ線形回帰の出力を取得
        y_dist = self.bayesian_output.predict_dist(x)

        if self.min_val or self.max_val:
            y_mean = torch.clamp(y_dist.mean, min=self.min_val, max=self.max_val)
        else:
            y_mean = y_dist.mean

        if self.clipping:
            y_mean = y_mean.clamp(min=-1e2, max=1e2)
            y_stddev = y_dist.stddev.clamp(min=1e-6, max=1e1)
        else:
            y_stddev = y_dist.stddev
        
        return Normal(y_mean, y_stddev)


class BayesianMLPModel(Model):
    def __init__(self, train_X, train_Y, min_val=None, max_val=None, clipping=False):
        super().__init__()
        self.bayesian_mlp = BayesianMLP(train_X.shape[1], min_val, max_val, clipping=clipping)
        self.likelihood = GaussianLikelihood()
        self._num_outputs = 1
        self._train_inputs = train_X
        self._train_targets = train_Y

    def forward(self, x):
        return self.bayesian_mlp(x)
    
    def posterior(self, X, observation_noise=False, **kwargs):
        pred_dist = self.bayesian_mlp(X)
        mean = pred_dist.mean.squeeze(-1)  # Ensure mean is 2D
        stddev = pred_dist.stddev.squeeze(-1)  # Ensure stddev is 2D
        covar = torch.diag_embed(stddev**2)
        return MultivariateNormal(mean, covar)
    
    @property
    def num_outputs(self):
        return self._num_outputs
    
    @property
    def train_inputs(self):
        return self._train_inputs

    @property
    def train_targets(self):
        return self._train_targets
    

def fit_pytorch_model(model, num_epochs=1000, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = -model(model.train_inputs).log_prob(model.train_targets).mean()
        loss.backward()
        optimizer.step()


