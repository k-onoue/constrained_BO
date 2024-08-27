import numpy as np
import torch


"""
バッチ処理に対応していないので注意
"""
def ackley_function(x, a=10, b=0.2, c=2*np.pi):
    """
    d次元Ackley関数を計算する。

    Parameters:
    x : ndarray
        入力ベクトル（d次元）。
    a : float, optional
        Ackley関数の定数。デフォルトは20。
    b : float, optional
        Ackley関数の定数。デフォルトは0.2。
    c : float, optional
        Ackley関数の定数。デフォルトは2π。

    Returns:
    float
        Ackley関数の値。
    """
    x = np.array(x)  # ここでリストをNumPy配列に変換
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term


"""
バッチ処理に対応していないので注意
"""
def rosenbrock_function(x, a=1, b=100):
    """
    d次元Rosenbrock関数を計算する

    Parameters:
    x : ndarray
        入力ベクトル（d次元）
    a : float, optional
        Rosenbrock関数の定数．デフォルトは1
    b : float, optional
        Rosenbrock関数の定数．デフォルトは100

    Returns:
    float
        Rosenbrock関数の値
    """
    x = np.array(x)  # ここでリストをNumPy配列に変換
    return np.sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)


"""
バッチ処理に対応していないので注意
"""
def discretize_function(x, c=0.1):
    """
    連続値の入力を離散化する

    Parameters:
    x : ndarray
        入力ベクトル（d次元）
    c : float, optional
        離散化の粒度．デフォルトは0.1．

    Returns:
    ndarray
        離散化された入力ベクトル（d次元）
    """
    return c * np.floor(x / c)


# Test function
def test_function(x):
    r"""
    $f(x) = e^{-(x-2)^2} + e^{-\left(\frac{x-6}{10}\right)^2} + \frac{1}{e^{x^2} + 1}$
    """
    return np.exp(-((x - 2) ** 2)) + np.exp(-((x - 6) ** 2) / 10) + 1 / (x**2 + 1)


# Schubert function with batch input
def schubert_function(X):
    r"""
    $f(x_1, x_2) = \prod_{i=1}^2 \left( \sum_{j=1}^5 j \cos((j + 1)x_i + j) \right)$
    """
    j = torch.arange(1, 6).float()
    sum1 = torch.sum(j * torch.cos((j + 1) * X[:, 0:1] + j), dim=1)
    sum2 = torch.sum(j * torch.cos((j + 1) * X[:, 1:2] + j), dim=1)
    return sum1 * sum2


# Eggholder function with batch input
def eggholder_function(X):
    r"""
    f(x_1, x_2) = - (x_2 + 47) \sin \left( \sqrt{\left| x_2 + \frac{x_1}{2} + 47 \right|} \right) - x_1 \sin \left( \sqrt{\left| x_1 - (x_2 + 47) \right|} \right)
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))
    return term1 + term2


# Griewank function with batch input
def griewank_function(X):
    r"""
    f(x) = \sum_{i=1}^d \frac{x_i^2}{4000} - \prod_{i=1}^d \cos \left( \frac{x_i}{\sqrt{i}} \right) + 1
    """
    sum_term = torch.sum(X**2 / 4000, dim=1)
    prod_term = torch.prod(
        torch.cos(X / torch.sqrt(torch.arange(1, X.shape[1] + 1).float())), dim=1
    )
    return sum_term - prod_term + 1