import numpy as np
import torch

from .utils_warcraft import manhattan_distance, navigate_through_matrix


class WarcraftObjectiveBoTorch:
    def __init__(
        self,
        weight_matrix: torch.tensor,
    ) -> None:
        self.weight_matrix = weight_matrix / weight_matrix.sum()  # normalize
        self.shape = weight_matrix.shape
        self.search_space_1d_dict = {
            -3: "oo",
            -2: "ab",
            -1: "ac",
            0: "ad",
            1: "bc",
            2: "bd",
            3: "cd",
        }
        self.reverse_search_space_1d_dict = {
            v: k for k, v in self.search_space_1d_dict.items()
        }

    def string_to_tensor(self, direction_matrix: np.ndarray) -> torch.tensor:
        tensor_matrix = torch.zeros(self.shape, dtype=torch.int)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                tensor_matrix[i, j] = self.reverse_search_space_1d_dict.get(
                    direction_matrix[i, j], 0.0
                )
        return torch.tensor(tensor_matrix, dtype=torch.float32)

    def tensor_to_string(self, x: torch.tensor) -> np.ndarray:
        direction_matrix = np.zeros(self.shape, dtype=object)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                direction_matrix[i, j] = self.search_space_1d_dict.get(x[i, j].item())
        return direction_matrix

    def __call__(self, x: torch.tensor) -> torch.Tensor:
        if type(x) == torch.Tensor:
            direction_matrix = self.tensor_to_string(x)
        else:
            direction_matrix = x

        start = (0, 0)
        goal = (self.shape[0] - 1, self.shape[1] - 1)

        history = navigate_through_matrix(direction_matrix, start, goal)

        if history:
            path_weight = sum(self.weight_matrix[point] for point in history)
            norm_const = manhattan_distance(start, goal)
            loss1 = (
                1
                - (1 - manhattan_distance(history[-1], goal) / norm_const)
                + path_weight
            )
        else:
            loss1 = 1

        mask = direction_matrix != "oo"
        loss2 = self.weight_matrix[mask].sum()

        loss = loss1 + loss2
        score = -loss

        return score

    def visualize(self, x: torch.tensor) -> None:
        direction_matrix = self.tensor_to_string(x)
        print(direction_matrix)


# if __name__ == "__main__":
#     objective = WarcraftObjectiveBoTorch(
#         torch.tensor(
#             [
#                 [0.1, 0.4, 0.8, 0.8],
#                 [0.2, 0.4, 0.4, 0.8],
#                 [0.8, 0.1, 0.1, 0.2],
#             ]
#         )
#     )

#     val = objective(
#         torch.tensor(
#             [
#                 [2, -3, -3, -3],
#                 [1, 0, -3, -3],
#                 [-3, 1, -1, -1],
#             ]
#         )
#     )

#     print(val)
