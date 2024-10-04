import numpy as np
import torch


def to_numpy(x):
    """Convert torch.Tensor to numpy.ndarray if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def to_tensor(x):
    """Convert numpy.ndarray to torch.Tensor if necessary."""
    if isinstance(x, np.ndarray):
        return torch.tensor(x)
    return x


def get_opposite(direction: str) -> str:
    """
    Examples
    --------
    >>> get_opposite('a')
    'c'
    """
    pair_dict = {"a": "c", "c": "a", "b": "d", "d": "b"}
    return pair_dict.get(direction, "")


def judge_continuity(d_from: str, current_direction: str) -> bool:
    """
    Examples
    --------
    >>> judge_continuity('a', 'ad')
    False
    >>> judge_continuity('a', 'bc')
    True
    """
    d_opposite = get_opposite(d_from)
    return d_opposite in current_direction


def get_next_coordinate(
    d_to: str, current_coordinate: tuple[int, int]
) -> tuple[int, int]:
    """
    Examples
    --------
    >>> get_next_coordinate('a', (0, 0))
    (-1, 0)
    """
    update_dict = {"a": (-1, 0), "b": (0, -1), "c": (0, 1), "d": (1, 0)}
    delta = update_dict.get(d_to, (0, 0))
    return (current_coordinate[0] + delta[0], current_coordinate[1] + delta[1])


def judge_location_validity(current: tuple[int, int], shape: tuple[int, int]) -> bool:
    """
    Examples
    --------
    >>> judge_location_validity((-1, 0), (3, 3))
    False
    >>> judge_location_validity((1, 2), (3, 3))
    True
    """
    return 0 <= current[0] < shape[0] and 0 <= current[1] < shape[1]


def get_d_to(d_from: str, current_direction: str) -> str:
    """
    Examples
    --------
    >>> get_d_to('a', 'ad')
    'd'
    """
    return (
        current_direction[1] if current_direction[0] == d_from else current_direction[0]
    )


def navigate_through_matrix(direction_matrix, start, goal):
    history = []
    current = start
    shape = direction_matrix.shape

    if direction_matrix[current] in ["cd", "oo"]:
        return history

    history.append(current)
    d_to = "d"
    next_pos = get_next_coordinate(d_to, current)

    while judge_location_validity(next_pos, shape) and current != goal:
        if not judge_continuity(d_to, direction_matrix[next_pos]):
            break

        current = next_pos
        history.append(current)
        if current == goal:
            break

        direction = direction_matrix[current]
        d_from = get_opposite(d_to)
        d_to = get_d_to(d_from, direction)
        next_pos = get_next_coordinate(d_to, current)

    return history


def manhattan_distance(coord1: tuple[int, int], coord2: tuple[int, int]) -> int:
    """
    Examples
    --------
    >>> manhattan_distance((0, 0), (3, 3))
    6
    """
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


# def generate_initial_data(
#     objective_function: callable,
#     dataset_size: int,
#     shape: tuple[int, int],
#     use_torch: bool = True,  # Add option to use torch or numpy
# ):
#     values = np.array([-3, -2, -1, 0, 1, 2, 3])
#     n, m = shape

#     if use_torch:
#         # Use torch.tensor if specified
#         X_train = torch.tensor(values)[torch.randint(0, len(values), (dataset_size, n, m))]
#         y_train = torch.stack([objective_function(x) for x in X_train]).unsqueeze(-1)
#     else:
#         # Use numpy
#         X_train = values[np.random.randint(0, len(values), (dataset_size, n, m))]
#         y_train = np.array([objective_function(x) for x in X_train]).reshape(-1, 1)

#     return X_train, y_train


def generate_random_tuple(category_num, dim, num=1):
    """
    Function to generate random tuples
    To suggest the initial point to evalute the objective function
    To decide which elements to mask in the tensor
    """
    return [tuple(np.random.randint(0, category_num, dim)) for _ in range(num)]


def convert_tensor_index_to_map(path, map_shape):
    path_gen_reversed = iter(reversed(path))
    map = np.zeros(map_shape)
        
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            map[i, j] = next(path_gen_reversed)

    return map


# WarcraftObjective class supporting both torch and numpy
class WarcraftObjective:
    def __init__(
        self,
        weight_matrix,  # Supports both torch.tensor and np.ndarray
    ) -> None:
        weight_matrix = to_numpy(weight_matrix)
        self.weight_matrix = weight_matrix / np.sum(weight_matrix)  # normalize
        self.shape = weight_matrix.shape
        self.search_space_1d_dict = {
            0: "oo",
            1: "ab",
            2: "ac",
            3: "ad",
            4: "bc",
            5: "bd",
            6: "cd",
        }
        self.reverse_search_space_1d_dict = {
            v: k for k, v in self.search_space_1d_dict.items()
        }

    def string_to_tensor(self, direction_matrix):
        """Convert string-based matrix to integer tensor (supports both numpy and torch)."""
        tensor_matrix = np.zeros(self.shape, dtype=np.int32)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                tensor_matrix[i, j] = self.reverse_search_space_1d_dict.get(
                    direction_matrix[i, j], 0
                )
        return to_tensor(tensor_matrix.astype(np.float32))

    def tensor_to_string(self, x):
        """Convert tensor to string-based matrix (supports both numpy and torch)."""
        x = to_numpy(x)
        direction_matrix = np.zeros(self.shape, dtype=object)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                direction_matrix[i, j] = self.search_space_1d_dict.get(x[i, j])
        return direction_matrix
        
    def calculate_penalty_type2(self, idx, val, map_shape):
            # Define the mask dictionary within the function
            val_mask_dict = {
                "oo": np.zeros((3, 3)),
                "ab": np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]]),
                "ac": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
                "ad": np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]),
                "bc": np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]),
                "bd": np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
                "cd": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]),
            }

            # Initialize the expanded array with zeros, sized (2*map_shape[0] + 1, 2*map_shape[1] + 1)
            arr_expanded = np.zeros((map_shape[0] * 2 + 1, map_shape[1] * 2 + 1))

            # Calculate the starting positions for the mask application
            x_s, y_s = idx[0] * 2, idx[1] * 2

            # Apply the corresponding mask from the dictionary based on 'val'
            arr_expanded[x_s : x_s + 3, y_s : y_s + 3] = val_mask_dict.get(
                val, np.zeros((3, 3))
            )

            # Get the indices of the ones in the expanded array
            ones_indices = np.argwhere(arr_expanded == 1)

            # Inicialize a variable to store the minimum distance
            row = arr_expanded.shape[0] - 1
            col = arr_expanded.shape[1] - 1

            max_distance = manhattan_distance((0, 0), (row, col-1))
            min_distance = max_distance

            index_goal_list = [(row, col-1), (row-1, col)]

            # Iterate through all pairs of (1 indices, target indices)
            for one_idx in ones_indices:
                for target_idx in index_goal_list:
                    dist = manhattan_distance(one_idx, target_idx)
                    if dist < min_distance:
                        min_distance = dist

            penalty = min_distance / max_distance
            return penalty

    def __call__(self, x):
        """Calculate the objective function."""
        if isinstance(x, (torch.Tensor, np.ndarray)):
            direction_matrix = self.tensor_to_string(x)
        else:
            direction_matrix = x

        mask = np.where(direction_matrix == 'oo', 0, 1)
        penalty_1 = np.sum(self.weight_matrix * mask)

        start = (0, 0)
        goal = (self.shape[0] - 1, self.shape[1] - 1)

        history = navigate_through_matrix(direction_matrix, start, goal)

        # penalty_2 = np.sum(self.weight_matrix[point] for point in history)

        if history:
            penalty_3 = self.calculate_penalty_type2(history[-1], direction_matrix[history[-1]], self.shape)
        else:
            penalty_3 = 1

        print(f'penalty_1: {penalty_1}')
        # print(f'penalty_2: {penalty_2}')
        print(f'penalty_3: {penalty_3}')
        
        # score = penalty_1 + penalty_2 + penalty_3
        score = penalty_1 + penalty_3
        return score

    def visualize(self, x):
        """Visualize the direction matrix."""
        direction_matrix = self.tensor_to_string(x)
        print(direction_matrix)



# if __name__ == "__main__":
#     import optuna

#     def objective(trial):
#         # Define the shape of the array (same as map_targeted)
#         shape = (2, 2)
        
#         # Suggest values for each element in the array, which can be 0 to 6
#         x = np.array([
#             [trial.suggest_int('x00', 0, 6), trial.suggest_int('x01', 0, 6)],
#             [trial.suggest_int('x10', 0, 6), trial.suggest_int('x11', 0, 6)]
#         ])

#         # Calculate the score using WarcraftObjective
#         map_targeted_scaled = np.array([[1, 4], [2, 1]]) / np.sum(np.array([[1, 4], [2, 1]]))
#         objective_function = WarcraftObjective(map_targeted_scaled)
        
#         # Calculate the objective function score based on x
#         score = objective_function(x)
        
#         # Since we want to maximize the score, return -score for minimization
#         return score
    
#     # Run the optimization using Optuna
#     study = optuna.create_study(direction='minimize')  # We minimize because we return -score
#     study.optimize(objective, n_trials=100)
    
#     # Print the best result
#     print(f"Best value: {study.best_value}")
#     print(f"Best params: {study.best_params}")



# if __name__ == "__main__":

#     # Test with a numpy array weight matrix
#     weight_matrix_np = np.random.rand(3, 3)
#     objective_np = WarcraftObjective(weight_matrix_np)
#     direction_matrix_np = np.array([["bd", "oo", "bd"], ["oo", "oo", "bd"], ["bd", "oo", "oo"]])

#     # Test with a torch tensor weight matrix
#     weight_matrix_torch = torch.rand(3, 3)
#     objective_torch = WarcraftObjective(weight_matrix_torch)
#     direction_matrix_torch = np.array([["bd", "oo", "bd"], ["oo", "oo", "bd"], ["bd", "oo", "oo"]])

#     # Running the objective function
#     score_np = objective_np(objective_np.string_to_tensor(direction_matrix_np))
#     score_torch = objective_torch(objective_torch.string_to_tensor(direction_matrix_torch))

#     # Output the results
#     print(f"Score using numpy: {score_np}")
#     print(f"Score using torch: {score_torch}")