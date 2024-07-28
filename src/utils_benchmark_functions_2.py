import numpy as np
import torch


# Test function
def test_function(x):
    r"""
    $f(x) = e^{-(x-2)^2} + e^{-\left(\frac{x-6}{10}\right)^2} + \frac{1}{e^{x^2} + 1}$
    """
    return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)

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
    prod_term = torch.prod(torch.cos(X / torch.sqrt(torch.arange(1, X.shape[1] + 1).float())), dim=1)
    return sum_term - prod_term + 1




# if __name__ == "__main__":
#     # Plot Test function
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D

#     x = np.linspace(-2, 10, 400)
#     y = test_function(x)
#     plt.plot(x, y)
#     plt.title('Test function')
#     plt.xlabel('x')
#     plt.ylabel('f(x)')
#     plt.grid(True)
#     plt.show()

#     # Plot Schubert function
#     x = np.linspace(-10, 10, 400)
#     y = np.linspace(-10, 10, 400)
#     X, Y = np.meshgrid(x, y)
#     Z = schubert(X, Y)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='viridis')
#     ax.set_title('Schubert function')
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.set_zlabel('f(x1, x2)')
#     plt.show()

#     # Plot Eggholder function
#     x = np.linspace(-512, 512, 400)
#     y = np.linspace(-512, 512, 400)
#     X, Y = np.meshgrid(x, y)
#     Z = eggholder(X, Y)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z, cmap='viridis')
#     ax.set_title('Eggholder function')
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.set_zlabel('f(x1, x2)')
#     plt.show()

