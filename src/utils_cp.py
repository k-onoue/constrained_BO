import numpy as np
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import non_negative_parafac


def generate_random_tuple(category_size, dim, num=1):
    """
    Function to generate random tuples
    To suggest the initial point to evalute the objective function
    To decide which elements to mask in the tensor
    """
    return [tuple(np.random.randint(0, category_size, dim)) for _ in range(num)]


# Function to apply missing data based on the provided index




if __name__ == "__main__":

    map_targeted = np.array([
        [1, 4],
        [2, 1]
    ])

    map_targeted = map_targeted.flatten() / np.sum(map_targeted)

    settings = {
        "name": "test" * 10,
        "seed": 0,
        "category_size": "7",
        "iter": 5,
        "cp_settings": {
            "dim": len(map_targeted),
            "rank": 2,
            "als_iterations": 100,
        }
    }

    dim = settings["cp_settings"]["dim"]
    category_size = settings["category_size"]
    
    tensor_targeted_bool = np.zeros((category_size,) * dim)


    temp = generate_random_tuple(10, 3, 5)
    print(temp)


    