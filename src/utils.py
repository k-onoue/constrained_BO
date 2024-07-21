import numpy as np


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


def rosenbrock_function(x, a=1, b=100):
    """
    d次元Rosenbrock関数を計算する。

    Parameters:
    x : ndarray
        入力ベクトル（d次元）。
    a : float, optional
        Rosenbrock関数の定数。デフォルトは1。
    b : float, optional
        Rosenbrock関数の定数。デフォルトは100。

    Returns:
    float
        Rosenbrock関数の値。
    """
    x = np.array(x)  # ここでリストをNumPy配列に変換
    return np.sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)


def discretize_function(x, c=0.1):
    """
    連続値の入力を離散化する。

    Parameters:
    x : ndarray
        入力ベクトル（d次元）
    c : float, optional
        離散化の粒度．デフォルトは0.1．

    Returns:
    ndarray
        離散化された入力ベクトル（d次元）．
    """
    return c * np.floor(x / c)




if __name__ == '__main__':
    import numpy as np
    import plotly.graph_objects as go

    # 1次元のAckley関数を離散化してプロット
    x_1d = np.linspace(-5, 5, 400)
    x_1d_discrete = discretize_function(x_1d, c=0.2)
    y_1d_discrete = [ackley_function([x]) for x in x_1d_discrete]

    fig_1d_discrete = go.Figure(data=go.Scatter(x=x_1d, y=y_1d_discrete, mode='lines'))
    fig_1d_discrete.update_layout(title='1次元の離散化されたAckley関数', xaxis_title='x', yaxis_title='f(x)')
    fig_1d_discrete.show()

    # 2次元のAckley関数を離散化してプロット
    x_2d = np.linspace(-5, 5, 100)
    y_2d = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_2d, y_2d)
    X_discrete = discretize_function(X, c=0.5)
    Y_discrete = discretize_function(Y, c=0.5)
    Z_discrete = np.array([[ackley_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X_discrete, Y_discrete)])

    fig_2d_discrete = go.Figure(data=[go.Surface(z=Z_discrete, x=X, y=Y)])
    fig_2d_discrete.update_layout(title='2次元の離散化されたAckley関数', scene=dict(
                                xaxis_title='x',
                                yaxis_title='y',
                                zaxis_title='f(x, y)'))
    fig_2d_discrete.show()

    # 2次元のRosenbrock関数を離散化してプロット
    Z_discrete_rosenbrock = np.array([[rosenbrock_function([x, y]) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X_discrete, Y_discrete)])

    fig_2d_discrete_rosenbrock = go.Figure(data=[go.Surface(z=Z_discrete_rosenbrock, x=X, y=Y)])
    fig_2d_discrete_rosenbrock.update_layout(title='2次元の離散化されたRosenbrock関数', scene=dict(
                                xaxis_title='x',
                                yaxis_title='y',
                                zaxis_title='f(x, y)'))
    fig_2d_discrete_rosenbrock.show()