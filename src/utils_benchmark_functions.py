import numpy as np
import torch


"""
バッチ処理に対応していないので注意
"""


def ackley_function(x, a=10, b=0.2, c=2 * np.pi):
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
    return np.sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2)


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


def schubert_function(X):
    dtype = X.dtype
    device = X.device
    input_dim = X.dim()

    if X.dim() == 1:
        X = X.unsqueeze(0)  # 1次元のテンソルを2次元に変換

    # jの値を1から5までのテンソルで表現
    j = torch.arange(1, 6, dtype=X.dtype, device=X.device).view(1, -1)
    
    # 各次元のXに対して計算を行う
    X_expanded = X.unsqueeze(-1)  # (n, d, 1) の形に拡張
    terms = j * torch.cos((j + 1) * X_expanded + j)  # (n, d, 5)
    
    # 次元ごとにsumを取って、最終的なprodを取る
    result = torch.prod(terms.sum(dim=-1), dim=-1)  # sumは最後の次元、prodは次元d

    # 出力が1つの要素しかない場合はスカラー（0次元）にする
    if input_dim == 1:
        return torch.tensor(result.item()).to(dtype=dtype).to(device=device)  # スカラー値として返す
    return result


def eggholder_function(X):
    dtype = X.dtype
    device = X.device
    input_dim = X.dim()

    if input_dim == 1:
        X = X.unsqueeze(0)  # 1次元のテンソルを2次元に変換

    x1 = X[:, 0]
    x2 = X[:, 1]
    term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))
    result = term1 + term2

    if input_dim == 1:
        return torch.tensor(result.item()).to(dtype=dtype).to(device=device)
    return result

def griewank_function(X):
    dtype = X.dtype
    device = X.device
    input_dim = X.dim()

    if input_dim == 1:
        X = X.unsqueeze(0)  # 1次元のテンソルを2次元に変換

    sum_term = torch.sum(X**2 / 4000, dim=1)
    prod_term = torch.prod(
        torch.cos(
            X / torch.sqrt(torch.arange(1, X.size(1) + 1, dtype=dtype, device=device))
        ),
        dim=1,
    )
    result = sum_term - prod_term + 1

    if input_dim == 1:
        return torch.tensor(result.item()).to(dtype=dtype).to(device=device)
    return result

