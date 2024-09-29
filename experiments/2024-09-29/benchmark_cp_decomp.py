import logging
import os
import time
import argparse

import numpy as np
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import non_negative_parafac

from _src import LOG_DIR
from _src import set_logger


# Function to generate random tuples
def generate_random_tuple(size, dim, num=1):
    return [tuple(np.random.randint(0, size, dim)) for _ in range(num)]


# Function to apply missing data based on the provided index
def apply_mask(tensor, mask_subs_list):
    masked_tensor = tensor.copy()
    for index in mask_subs_list:
        masked_tensor[index] = np.nan
    return masked_tensor


# Function to fill missing values
def fill_missing_values(tensor, mask, rank, als_iterations):
    mask_bool = ~np.isnan(tensor)
    tensor[mask_bool == False] = 0  # Initialize missing data to 0
    return non_negative_parafac(tensor, rank=rank, mask=mask, n_iter_max=als_iterations)


# Argument parser function for easy extension
def get_arguments():
    parser = argparse.ArgumentParser(description="Run NN PARAFAC experiment")
    parser.add_argument(
        "--dim", type=int, required=True, help="Dimension of the tensor"
    )
    parser.add_argument(
        "--size", type=int, default=7, help="Size of each dimension of the tensor"
    )
    parser.add_argument(
        "--rank", type=int, default=2, help="Rank of the tensor decomposition"
    )
    parser.add_argument(
        "--mask_num",
        type=int,
        default=100,
        help="Number of masked entries in the tensor",
    )
    parser.add_argument(
        "--experiment_iterations",
        type=int,
        default=5,
        help="Number of iterations for the experiment",
    )
    parser.add_argument(
        "--als_iterations",
        type=int,
        default=100,
        help="Number of ALS iterations for CP decomposition",
    )
    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for reproducibility"
    )

    return parser.parse_args()


# Function to run the experiment and measure time iteration-by-iteration
def run_nn_parafac(params):
    try:
        logging.info(f"Experimental parameters: {params}")

        size = params["cp_settings"]["size"]
        dim = params["cp_settings"]["dim"]
        rank = params["cp_settings"]["rank"]
        mask_num = params["cp_settings"]["mask_num"]
        als_iterations = params["cp_settings"]["als_iterations"]
        experiment_iterations = params["experiment_iterations"]

        np.random.seed(params["random_seed"])

        # 1. Generate the original tensor (constructed once)
        start_time_tensor_construction = time.time()
        tensor = np.random.random((size,) * dim)
        time_to_construct_tensor = time.time() - start_time_tensor_construction
        logging.info(
            f"Time to construct the original tensor: {time_to_construct_tensor:.4f} seconds"
        )

        mask_subs_list = generate_random_tuple(size, dim, num=mask_num)

        # Iteration-wise timing
        for iteration in range(experiment_iterations):
            logging.info(
                f"--- Experiment Iteration {iteration + 1}/{experiment_iterations} ---"
            )

            # Apply missing data to the tensor
            masked_tensor = apply_mask(tensor, mask_subs_list)

            # Randomly select additional indices not in mask_subs_list and apply missing data
            non_mask_subs_list = generate_random_tuple(size, dim, num=mask_num)
            non_mask_subs_list = [
                idx for idx in non_mask_subs_list if idx not in mask_subs_list
            ]
            additional_masked_tensor = apply_mask(masked_tensor, non_mask_subs_list)

            # 2. Measure time for tensor decomposition (per experiment iteration)
            start_time_decomposition = time.time()
            mask_bool = ~np.isnan(additional_masked_tensor)
            factors = fill_missing_values(
                additional_masked_tensor, mask_bool, rank, als_iterations
            )
            decomposition_time = time.time() - start_time_decomposition
            logging.info(
                f"Time to perform tensor decomposition (ALS iterations): {decomposition_time:.4f} seconds"
            )

            # 3. Measure time for tensor reconstruction (per experiment iteration)
            start_time_reconstruction = time.time()
            reconstructed_tensor = cp_to_tensor(factors)
            reconstruction_time = time.time() - start_time_reconstruction
            logging.info(
                f"Time to reconstruct the tensor: {reconstruction_time:.4f} seconds"
            )

            # Save the filled values (same as before, no need to log the values per iteration)
            filled_mask_values = [
                reconstructed_tensor[index] for index in mask_subs_list
            ]
            filled_non_mask_values = [
                reconstructed_tensor[index] for index in non_mask_subs_list
            ]

        # Return the filled values for analysis after all iterations
        mean_filled_mask = np.mean(filled_mask_values)
        var_filled_mask = np.var(filled_mask_values)
        mean_filled_non_mask = np.mean(filled_non_mask_values)
        var_filled_non_mask = np.var(filled_non_mask_values)

        return {
            "mean_filled_mask": mean_filled_mask,
            "var_filled_mask": var_filled_mask,
            "mean_filled_non_mask": mean_filled_non_mask,
            "var_filled_non_mask": var_filled_non_mask,
        }

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        return None


# Main function
if __name__ == "__main__":
    # Parse arguments using the argument parser function
    args = get_arguments()

    # Experiment parameters dictionary
    name = os.path.splitext(__file__.split("/")[-1])[0] + f"_dim{args.dim}"

    settings = {
        "name": name,
        "cp_settings": {
            "size": args.size,
            "dim": args.dim,
            "rank": args.rank,
            "mask_num": args.mask_num,
            "als_iterations": args.als_iterations,
        },
        "experiment_iterations": args.experiment_iterations,
        "random_seed": args.random_seed,
    }

    set_logger(settings["name"], LOG_DIR)

    # Run the experiment
    results = run_nn_parafac(settings)

    if results:
        # Log experiment results
        logging.info(
            f"Mean of the filled values in mask_subs_list: {results['mean_filled_mask']}"
        )
        logging.info(
            f"Variance of the filled values in mask_subs_list: {results['var_filled_mask']}"
        )
        logging.info(
            f"Mean of the filled values in other locations: {results['mean_filled_non_mask']}"
        )
        logging.info(
            f"Variance of the filled values in other locations: {results['var_filled_non_mask']}"
        )
    else:
        logging.error("Experiment failed due to an error.")
