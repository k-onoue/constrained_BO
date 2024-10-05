import numpy as np
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac
from utils_warcraft import generate_random_tuple, convert_path_index_to_arr, WarcraftObjective


def split_list_equally(input_list, div):
    quotient, remainder = divmod(len(input_list), div)
    result = []
    start = 0
    for i in range(div):
        group_size = quotient + (1 if i < remainder else 0)
        result.append(input_list[start:start + group_size])
        start += group_size
    return result


def ucb(mean_tensor, variance_tensor, trade_off_param=1.0, batch_size=10, maximize=True):
    """
    Calculate the Upper Confidence Bound (UCB) for each point in the tensor and suggest
    the best points based on UCB values.

    Parameters:
    - mean_tensor: np.ndarray, the mean values at each point.
    - variance_tensor: np.ndarray, the variance values at each point.
    - trade_off_param: float, the trade-off parameter between exploration and exploitation.
                        Default is 1.0.
    - batch_size: int, the number of points to suggest. Default is 10.
    - maximize: bool, if True, maximize UCB values. If False, minimize UCB values. Default is True.

    Returns:
    - indices: list of tuples, the indices of the top batch_size points.
    """
    # For minimization, negate the mean_tensor
    mean_tensor = mean_tensor if maximize else -mean_tensor
    
    # Compute the UCB values
    ucb_values = mean_tensor + trade_off_param * np.sqrt(variance_tensor)
    
    # Flatten the tensor and get the indices of the top UCB values
    flat_indices = np.argsort(ucb_values.flatten())[::-1]  # Sort in descending order
    
    # Get the multi-dimensional indices for the top batch_size UCB values
    top_indices = np.unravel_index(flat_indices[:batch_size], mean_tensor.shape)
    
    # Combine the multi-dimensional indices into a list of tuples
    top_indices = list(zip(*top_indices))
    
    return top_indices


if __name__ == "__main__":

    map_targeted = np.array([[1, 4], [2, 1]])
    map_targeted_scaled = map_targeted / np.sum(map_targeted)

    settings = {
        "name": "test" * 10,
        "seed": 0,
        "category_num": 7,
        "iter": 10,  # Number of iterations for Bayesian optimization
        "init_eval_num": 7 ** 3,
        "cp_settings": {
            "dim": len(map_targeted.flatten()),
            "rank": 2,
            "als_iterations": 100,
            "mask_ratio": 0.2,
        },
        "acqf_settings": {
            "trade_off_param": 1.0,
            "batch_size": 10,
            "maximize": False,
        },
    }

    # General settings
    seed = settings["seed"]
    category_num = settings["category_num"]
    iter_num = settings["iter"]  # Number of iterations
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

    # Initialize tensors for evaluations and masking
    tensor_eval = np.zeros((category_num,) * dim)
    tensor_eval_bool = np.zeros((category_num,) * dim)

    # Generate initial random paths
    initial_path_index_list = generate_random_tuple(
        category_num=category_num, 
        dim=dim, 
        num=init_eval_num
    )
    initial_path_arr_list = [
        convert_path_index_to_arr(path, map_targeted.shape) 
        for path in initial_path_index_list
    ]
    
    # Evaluate the initial paths
    for path_index, path_arr in zip(initial_path_index_list, initial_path_arr_list):
        tensor_eval[path_index] = objective_function(path_arr)
        tensor_eval_bool[path_index] = 1

    all_evaluated_indices = initial_path_index_list.copy()

    # Bayesian Optimization loop
    for iteration in range(iter_num):
        print(f"\nIteration {iteration + 1}/{iter_num}")

        # Add the suggested indices to the mask creation
        div = int(1 / mask_ratio)
        mask_split_list = split_list_equally(all_evaluated_indices, div)

        tensors_list = []

        # Perform CP decomposition with masking
        for mask_list in mask_split_list:
            mask_tensor = tensor_eval_bool.copy()
            for mask_index in mask_list:
                mask_tensor[mask_index] = 0

            # Perform CP decomposition
            cp_tensor = parafac(tensor_eval, rank=cp_rank, mask=mask_tensor, n_iter_max=als_iter_num)
            
            # Convert the CP decomposition back to a tensor
            reconstructed_tensor = cp_to_tensor(cp_tensor)
            
            # Append the reconstructed tensor to the list for later processing
            tensors_list.append(reconstructed_tensor)

        # Calculate mean and variance tensors
        tensors_stack = np.stack(tensors_list)
        mean_tensor = np.mean(tensors_stack, axis=0)
        variance_tensor = np.var(tensors_stack, axis=0)

        # Display the mean and variance tensors
        print(f"Mean tensor max: {np.max(mean_tensor)}, min: {np.min(mean_tensor)}")
        print(f"Variance tensor max: {np.max(variance_tensor)}, min: {np.min(variance_tensor)}")

        # Suggest new indices based on UCB
        suggested_indices = ucb(mean_tensor, variance_tensor, trade_off_param, batch_size, maximize)
        print(f"Suggested indices: {suggested_indices}")

        # Evaluate the new points and update the tensors
        for index in suggested_indices:
            path_arr = convert_path_index_to_arr(index, map_targeted.shape)
            tensor_eval[index] = objective_function(path_arr)
            tensor_eval_bool[index] = 1

        # Add the suggested indices to the list of all evaluated indices
        all_evaluated_indices.extend(suggested_indices)