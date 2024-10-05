import sys
from path_info import PROJECT_DIR

sys.path.append(PROJECT_DIR)

import numpy as np
import plotly.graph_objects as go
from src.utils_benchmark_functions import ackley_function
from src.utils_benchmark_functions import rosenbrock_function
from src.utils_benchmark_functions import discretize_function


if __name__ == "__main__":
    # 1次元のAckley関数を離散化してプロット
    x_1d = np.linspace(-5, 5, 400)
    x_1d_discrete = discretize_function(x_1d, c=0.2)
    y_1d_discrete = [ackley_function([x]) for x in x_1d_discrete]

    fig_1d_discrete = go.Figure(data=go.Scatter(x=x_1d, y=y_1d_discrete, mode="lines"))
    fig_1d_discrete.update_layout(
        title="1次元の離散化されたAckley関数", xaxis_title="x", yaxis_title="f(x)"
    )
    fig_1d_discrete.show()

    # 2次元のAckley関数を離散化してプロット
    x_2d = np.linspace(-1, 1, 50)
    y_2d = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x_2d, y_2d)
    X_discrete = discretize_function(X, c=0.1)
    Y_discrete = discretize_function(Y, c=0.1)
    Z_discrete = np.array(
        [
            [ackley_function([x, y]) for x, y in zip(row_x, row_y)]
            for row_x, row_y in zip(X_discrete, Y_discrete)
        ]
    )

    fig_2d_discrete = go.Figure(data=[go.Surface(z=Z_discrete, x=X, y=Y)])
    fig_2d_discrete.update_layout(
        title="2次元の離散化されたAckley関数",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
    )
    fig_2d_discrete.show()

    # 2次元のRosenbrock関数を離散化してプロット
    Z_discrete_rosenbrock = np.array(
        [
            [rosenbrock_function([x, y]) for x, y in zip(row_x, row_y)]
            for row_x, row_y in zip(X_discrete, Y_discrete)
        ]
    )

    fig_2d_discrete_rosenbrock = go.Figure(
        data=[go.Surface(z=Z_discrete_rosenbrock, x=X, y=Y)]
    )
    fig_2d_discrete_rosenbrock.update_layout(
        title="2次元の離散化されたRosenbrock関数",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
    )
    fig_2d_discrete_rosenbrock.show()
