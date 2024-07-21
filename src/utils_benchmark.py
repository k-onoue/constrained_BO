import numpy as np


"""
バッチ処理に対応していないので注意
"""
def ackley_function(x, a=20, b=0.2, c=2*np.pi):
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

