import logging
import os

import numpy as np

from _src import (
    InputManager,
    ParafacSampler,
    WarcraftObjective,
    generate_random_tuple,
)
from _src import LOG_DIR
from _src import set_logger


def run_bo(settings: dict):
    try:
        # Initialize settings from the dictionary
        map_targeted = settings["map"]
        map_targeted_scaled = map_targeted / np.sum(map_targeted)

        # General settings
        seed = settings["seed"]
        category_num = settings["category_num"]
        iter_num = settings["iter_bo"]  # Number of iterations
        init_eval_num = settings["init_eval_num"]

        # CP decomposition settings
        dim = settings["cp_settings"]["dim"]
        cp_rank = settings["cp_settings"]["rank"]
        als_iter_num = settings["cp_settings"]["als_iterations"]
        mask_ratio = settings["cp_settings"]["mask_ratio"]

        # Acquisition function settings
        trade_off_param = settings["acqf_settings"]["trade_off_param"]
        batch_size = settings["acqf_settings"]["batch_size"]
        maximize = settings["acqf_settings"]["maximize"]

        # Initialize objective function
        objective_function = WarcraftObjective(map_targeted_scaled)

        # Initialize InputManager
        input_manager = InputManager(
            category_num, dim, map_targeted.shape, objective_function, maximize=maximize
        )

        # Initialize ParafacSampler
        sampler = ParafacSampler(
            cp_rank,
            als_iter_num,
            mask_ratio,
            trade_off_param=trade_off_param,
            batch_size=batch_size,
            maximize=maximize,
        )

        # Generate initial random paths
        initial_path_index_list = generate_random_tuple(
            category_num=category_num, dim=dim, num=init_eval_num
        )

        # Add initial indices to the input manager
        input_manager.add_indices(initial_path_index_list)

        # Retrieve tensors and indices
        tensor_eval = input_manager.get_evaluation_tensor()
        tensor_eval_bool = input_manager.get_mask_tensor()
        all_evaluated_indices = input_manager.get_index_list()

        # Bayesian Optimization loop
        for iteration in range(iter_num):
            print(f"\nIteration {iteration + 1}/{iter_num}")

            # Perform CP decomposition and get mean and variance tensors
            # Suggest new indices based on UCB
            suggested_indices = sampler.sample(
                tensor_eval, tensor_eval_bool, all_evaluated_indices
            )
            logging.info(f"Suggested indices: {suggested_indices}")

            # Add the new indices to the input manager and evaluate
            input_manager.add_indices(suggested_indices)

            # Update tensors and indices
            tensor_eval = input_manager.get_evaluation_tensor()
            tensor_eval_bool = input_manager.get_mask_tensor()
            all_evaluated_indices = input_manager.get_index_list()

            # Get the optimal value and index
            opt_index, opt_val = input_manager.get_optimal_value_and_index()
            logging.info(f"Optimal index: {opt_index}")
            logging.info(f"Optimal value: {opt_val}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    name = os.path.splitext(__file__.split("/")[-1])[0]

    map_targeted = np.array([[1, 4], [2, 1]])

    settings = {
        "name": name,
        "seed": 0,
        "map": map_targeted,
        "category_num": 7,
        "iter_bo": 2000,  # Number of iterations for Bayesian optimization
        "init_eval_num": 7 * 7,  # Number of initial evaluations
        "cp_settings": {
            "dim": len(map_targeted.flatten()),
            "rank": 2,
            "als_iterations": 100,
            "mask_ratio": 0.2,
            "random_dist_type": "normal",
        },
        "acqf_settings": {
            "trade_off_param": 1.0,
            "batch_size": 10,
            "maximize": False,
        },
    }

    set_logger(settings["name"], LOG_DIR)

    run_bo(settings)
