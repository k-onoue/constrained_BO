import logging
from typing import Literal

import numpy as np
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import non_negative_parafac
from .utils_warcraft import (
    convert_path_index_to_arr,
)


class InputManager:
    def __init__(self, category_num, dim, map_shape, objective_function, maximize=True):
        """
        Initialize the InputManager to handle tensor evaluations and indices.
        """
        self.category_num = category_num
        self.dim = dim
        self.map_shape = map_shape
        self.objective_function = objective_function
        self.maximize = maximize

        # Initialize tensors for evaluations with NaN
        self.tensor_eval = np.full(
            (category_num,) * dim, np.nan
        )  # Now initialized with NaN
        self.tensor_eval_bool = np.zeros(
            (category_num,) * dim
        )  # Bool mask, keeping as zeros
        self.index_list = []
        self.index_array_list = []

        self.tensor_eval_p1 = np.full(
            (category_num,) * dim, np.nan
        )  # Initialized with NaN
        self.tensor_eval_p3 = np.full(
            (category_num,) * dim, np.nan
        )  # Initialized with NaN

    def add_indices(self, index_list):
        """
        Add a list of indices and evaluate the corresponding arrays using the objective function.

        Parameters:
        - index_list: list of tuples, the indices to evaluate.
        """
        for index in index_list:
            if index not in self.index_list:
                # Convert index to array
                path_arr = convert_path_index_to_arr(index, self.map_shape)
                self.index_list.append(index)
                self.index_array_list.append(path_arr)

                # Evaluate the objective function and update tensors
                val, p1, p3 = self.objective_function(path_arr)
                self.tensor_eval[index] = val
                self.tensor_eval_bool[index] = 1

                self.tensor_eval_p1[index] = p1
                self.tensor_eval_p3[index] = p3

                logging.info(
                    f"Evaluated index: {index}, Value: {val}, P1: {p1}, P3: {p3}"
                )

    def get_evaluation_tensor(self) -> np.ndarray:
        """
        Get the tensor of evaluations.

        Returns:
        - tensor_eval: np.ndarray, the current tensor of evaluations.
        """
        return self.tensor_eval

    def get_mask_tensor(self) -> np.ndarray:
        """
        Get the tensor mask where evaluations have been performed.

        Returns:
        - tensor_eval_bool: np.ndarray, the current tensor mask.
        """
        return self.tensor_eval_bool

    def get_index_list(self) -> list[tuple[int]]:
        return self.index_list

    def get_optimal_value_and_index(self) -> tuple[float, tuple[int]]:
        """
        Get the optimal value and the index that achieves it.

        Returns:
        - optimal_value: float, the optimal evaluation value (max or min based on self.maximize).
        - optimal_index: tuple of ints, the index where the optimal value occurs.
        """
        arg_opt_func = np.nanargmax if self.maximize else np.nanargmin
        optimal_index = np.unravel_index(
            arg_opt_func(self.tensor_eval), self.tensor_eval.shape
        )
        optimal_value = self.tensor_eval[optimal_index]

        return optimal_index, optimal_value


class ParafacSampler:
    def __init__(
        self,
        cp_rank: int,
        als_iter_num: int,
        mask_ratio: float,
        trade_off_param: float = 1.0,
        batch_size: int = 10,
        maximize: bool = True,
        distribution_type: Literal["uniform", "normal"] = "uniform",
    ):
        """
        Initialize the ParafacSampler with the necessary settings.

        Parameters:
        - cp_rank: int, the rank of the CP decomposition.
        - als_iter_num: int, the number of ALS iterations to perform during decomposition.
        - mask_ratio: float, the ratio to control the number of masks for CP decomposition.
        - trade_off_param: float, the trade-off parameter between exploration and exploitation.
        - batch_size: int, the number of points to suggest. Default is 10.
        - maximize: bool, if True, maximize UCB values. If False, minimize UCB values.
        """
        self.cp_rank = cp_rank
        self.als_iter_num = als_iter_num
        self.mask_ratio = mask_ratio
        self.trade_off_param = trade_off_param
        self.batch_size = batch_size
        self.maximize = maximize
        self.distribution_type = distribution_type

    def _fit(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        all_evaluated_indices: list,
        distribution_type: Literal["uniform", "normal"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform CP decomposition and return the mean and variance tensors.
        """
        div = int(1 / self.mask_ratio)
        mask_split_list = split_list_equally(all_evaluated_indices, div)

        tensors_list = []

        init_tensor_eval = generate_random_array(
            low=np.nanmin(tensor_eval),  # Use nanmin to avoid issues with NaN
            high=np.nanmax(tensor_eval),  # Use nanmax to avoid issues with NaN
            shape=tensor_eval.shape,
            distribution_type=distribution_type,
            mean=np.nanmean(tensor_eval),  # Use nanmean to handle NaN values
            std_dev=np.nanstd(tensor_eval),  # Use nanstd to handle NaN values
        )

        init_tensor_eval[tensor_eval_bool == 1] = tensor_eval[tensor_eval_bool == 1]

        for mask_list in mask_split_list:
            mask_tensor = np.ones_like(tensor_eval_bool)
            for mask_index in mask_list:
                mask_tensor[mask_index] = 0

            # Perform CP decomposition
            cp_tensor = non_negative_parafac(
                init_tensor_eval,
                rank=self.cp_rank,
                mask=mask_tensor,
                n_iter_max=self.als_iter_num,
            )

            # Convert the CP decomposition back to a tensor
            reconstructed_tensor = cp_to_tensor(cp_tensor)

            # Append the reconstructed tensor to the list for later processing
            tensors_list.append(reconstructed_tensor)

        # Calculate mean and variance tensors
        tensors_stack = np.stack(tensors_list)
        mean_tensor = np.mean(tensors_stack, axis=0)
        std_tensor = np.std(tensors_stack, axis=0)

        # Replace the mean and variance of known points with the original values and zeros
        mean_tensor[tensor_eval_bool == 1] = tensor_eval[tensor_eval_bool == 1]
        std_tensor[tensor_eval_bool == 1] = 0

        return mean_tensor, std_tensor

    def _suggest_ucb_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        trade_off_param: float,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int]]:
        """
        Suggest candidate points based on UCB values, selecting the top batch_size points.

        Parameters:
        - mean_tensor: np.ndarray, the mean values at each point.
        - std_tensor: np.ndarray, the std values at each point.
        - trade_off_param: float, the trade-off parameter between exploration and exploitation.
        - batch_size: int, the number of points to suggest. Default is 10.
        - maximize: bool, if True, maximize UCB values. If False, minimize UCB values.

        Returns:
        - indices: list of tuples, the indices of the top batch_size points based on UCB.
        """

        def _ucb(mean_tensor, std_tensor, trade_off_param, maximize=True) -> np.ndarray:
            mean_tensor = mean_tensor if maximize else -mean_tensor
            ucb_values = mean_tensor + trade_off_param * std_tensor
            return ucb_values

        # Calculate the UCB values using the internal function
        ucb_values = _ucb(mean_tensor, std_tensor, trade_off_param, maximize)

        # Flatten the tensor and get the indices of the top UCB values
        flat_indices = np.argsort(ucb_values.flatten())[
            ::-1
        ]  # Sort in descending order

        # top_indices = [
        #     np.unravel_index(flat_index, ucb_values.shape)
        #     for flat_index in flat_indices[:batch_size]
        # ]
        top_indices = np.unravel_index(flat_indices[:batch_size], ucb_values.shape)
        top_indices = list(zip(*top_indices))

        for index in top_indices:
            logging.info(
                f"UCB value at {index}: {ucb_values[index]}, Mean: {mean_tensor[index]}, Std: {std_tensor[index]}"
            )

        return top_indices

    def sample(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        all_evaluated_indices: list,
    ) -> list[tuple[int]]:
        """
        Perform sampling by calculating the mean and variance tensors using CP decomposition,
        then suggesting candidate points based on UCB values.

        Parameters:
        - tensor_eval: np.ndarray, the current evaluation tensor.
        - tensor_eval_bool: np.ndarray, the mask tensor indicating evaluated points.
        - all_evaluated_indices: list, the indices of all evaluated points.

        Returns:
        - indices: list of tuples, the indices of the top batch_size points based on UCB.
        """
        # Perform CP decomposition to get the mean and variance tensors
        mean_tensor, std_tensor = self._fit(
            tensor_eval, tensor_eval_bool, all_evaluated_indices, self.distribution_type
        )

        # Display the mean and variance tensors
        logging.info(f"Mean tensor min: {np.min(mean_tensor)}")
        logging.info(f"Mean tensor mean: {np.mean(mean_tensor)}")
        logging.info(f"Mean tensor std: {np.std(mean_tensor)}")
        logging.info(f"Mean tensor max: {np.max(mean_tensor)}")
        logging.info(f"Std tensor min: {np.min(std_tensor)}")
        logging.info(f"Std tensor mean: {np.mean(std_tensor)}")
        logging.info(f"Std tensor std: {np.std(std_tensor)}")
        logging.info(f"Std tensor max: {np.max(std_tensor)}")

        # Use the mean and variance tensors to suggest the top candidates
        return self._suggest_ucb_candidates(
            mean_tensor=mean_tensor,
            std_tensor=std_tensor,
            trade_off_param=self.trade_off_param,
            batch_size=self.batch_size,
            maximize=self.maximize,
        )


def split_list_equally(
    input_list: list[tuple[int]], div: int
) -> list[list[tuple[int]]]:
    quotient, remainder = divmod(len(input_list), div)
    result = []
    start = 0
    for i in range(div):
        group_size = quotient + (1 if i < remainder else 0)
        result.append(input_list[start : start + group_size])
        start += group_size
    return result


def generate_random_array(
    low: float,
    high: float,
    shape: tuple[int],
    distribution_type: Literal["uniform", "normal"] = "uniform",
    mean: float = 0,
    std_dev: float = 1,
) -> np.ndarray:
    """
    Generate an array of random numbers with specified bounds and distribution type.

    Parameters:
    - low: float, the lower bound of the random values.
    - high: float, the upper bound of the random values.
    - shape: tuple, the shape of the output array.
    - distribution_type: str, "uniform" or "normal" to choose the distribution type.
    - mean: float, the mean value for the normal distribution (default is 0).
    - std_dev: float, the standard deviation for the normal distribution (default is 1).

    Returns:
    - np.ndarray: array of random numbers with the specified properties.
    """
    if distribution_type == "uniform":
        # Generate uniform random numbers
        return np.random.uniform(low, high, shape)
    elif distribution_type == "normal":
        # Generate normal random numbers and clip them to the specified bounds
        normal_random = np.random.normal(mean, std_dev, shape)
        return np.clip(normal_random, low, high)
    else:
        raise ValueError("distribution_type must be either 'uniform' or 'normal'.")
