import logging
import os
import time
import numpy as np
from scipy.stats import qmc

from _src import (
    InputManager,
    WarcraftObjective,
    convert_path_index_to_arr,
)
from _src import LOG_DIR
from _src import set_logger


class RandomSampler:
    def __init__(self, category_num: int, dim: int, iter_num: int):
        self.category_num = category_num
        self.dim = dim
        self.iter_num = iter_num

    def sample(self) -> list[tuple[int]]:
        indices = []
        while len(indices) < self.iter_num:
            random_index = tuple(np.random.randint(0, self.category_num, size=self.dim))
            if random_index not in indices:
                indices.append(random_index)
        return indices


class BruteforceSampler:
    def __init__(self, maximize: bool = True, time_budget: float = None):
        self.maximize = maximize
        self.time_budget = time_budget

    def sample(self, tensor_eval: np.ndarray) -> list[tuple[int]]:
        start_time = time.time()
        arg_opt_func = np.nanargmax if self.maximize else np.nanargmin
        all_possible_indices = np.argwhere(np.isnan(tensor_eval) == False)
        
        sorted_indices = []
        for idx in sorted(all_possible_indices, key=lambda idx: tensor_eval[tuple(idx)], reverse=self.maximize):
            if self.time_budget is not None and time.time() - start_time >= self.time_budget:
                break
            sorted_indices.append(tuple(idx))
        
        return sorted_indices


class SobolSampler:
    def __init__(self, category_num: int, dim: int, iter_num: int):
        self.category_num = category_num
        self.dim = dim
        self.iter_num = iter_num
        self.sobol_engine = qmc.Sobol(d=self.dim, scramble=False)

    def sample(self) -> list[tuple[int]]:
        sobol_points = self.sobol_engine.random(n=self.iter_num)
        scaled_points = (sobol_points * self.category_num).astype(int)
        return [tuple(index) for index in scaled_points]


def get_sampler(settings: dict, tensor_eval: np.ndarray = None):
    sampler_name = settings["sampler"]
    category_num = settings["category_num"]
    dim = settings["map"].ndim
    iter_num = settings["iter_bo"]

    if sampler_name == "random":
        return RandomSampler(category_num, dim, iter_num)
    elif sampler_name == "bruteforce":
        time_budget = settings.get("time_budget", None)  # Time budget for BruteforceSampler
        return BruteforceSampler(maximize=settings["maximize"], time_budget=time_budget)
    elif sampler_name == "sobol":
        return SobolSampler(category_num, dim, iter_num)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


def run_bo(settings: dict):
    try:
        # Initialize settings from the dictionary
        map_targeted = settings["map"]
        map_targeted_scaled = map_targeted / np.sum(map_targeted)

        # General settings
        category_num = settings["category_num"]
        iter_num = settings["iter_bo"]

        # Initialize the sampler and objective function
        objective_function = WarcraftObjective(map_targeted_scaled)
        input_manager = InputManager(category_num, map_targeted.ndim, map_targeted.shape, objective_function)

        # Handle BruteforceSampler separately as it might need the tensor for sorting
        if settings["sampler"] == "bruteforce":
            tensor_eval = input_manager.get_evaluation_tensor()
            sampler = get_sampler(settings, tensor_eval)
            all_indices = sampler.sample(tensor_eval)  # Bruteforce sampler uses tensor_eval
        else:
            sampler = get_sampler(settings)
            all_indices = sampler.sample()  # Generate all points upfront for non-Bruteforce samplers

        # Add initial indices (the first points to evaluate)
        input_manager.add_indices(all_indices[:settings["init_eval_num"]])

        # Bayesian Optimization loop
        for iteration, new_index in enumerate(all_indices[settings["init_eval_num"]:], start=1):
            logging.info(f"Iteration {iteration}/{iter_num}")

            # Add the new index to the input manager and evaluate
            input_manager.add_indices([new_index])

            # Get the optimal value and index
            opt_index, opt_val = input_manager.get_optimal_value_and_index()
            logging.info(f"Optimal index: {opt_index}")
            logging.info(f"Optimal value: {opt_val}")
            logging.info(objective_function(convert_path_index_to_arr(opt_index, (2, 2))))

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    name = os.path.splitext(__file__.split("/")[-1])[0]

    map_targeted = np.array([[1, 4], [2, 1]])

    settings = {
        "name": name,
        "map": map_targeted,
        "category_num": 7,
        "iter_bo": 2000,  # Number of iterations for Bayesian optimization
        "init_eval_num": 0,  # Number of initial evaluations
        "maximize": False,
        "sampler": "bruteforce",  # Choose between "random", "bruteforce", "sobol"
        "time_budget": 5.0,  # Time budget for BruteforceSampler in seconds (optional)
    }

    set_logger(settings["name"], LOG_DIR)
    logging.info(f"Settings: {settings}")

    run_bo(settings)