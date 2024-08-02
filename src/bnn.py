import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


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
    def __init__(self, input_dim, min_val=None, max_val=None):
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
        
        # 出力をクランプ
        y_mean = torch.clamp(y_dist.mean, min=self.min_val, max=self.max_val)
        y_stddev = y_dist.stddev  # 標準偏差はそのまま
        
        # 新しい分布を返す
        return Normal(y_mean, y_stddev)

