import numpy as np
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import non_negative_parafac

# from .utils_warcraft import generate_random_tuple, WarcraftObjective
from utils_warcraft import generate_random_tuple, convert_tensor_index_to_map, WarcraftObjective


if __name__ == "__main__":

    map_targeted = np.array([[1, 4], [2, 1]])

    map_targeted_scaled = map_targeted / np.sum(map_targeted)

    settings = {
        "name": "test" * 10,
        "seed": 0,
        "category_num": 7,
        "iter": 5,
        "cp_settings": {
            "dim": len(map_targeted.flatten()),
            "rank": 2,
            "als_iterations": 100,
        },
    }

    dim = settings["cp_settings"]["dim"]
    category_num = settings["category_num"]

    objective_function = WarcraftObjective(map_targeted_scaled)

    tensor_targeted_bool = np.zeros((category_num,) * dim)

    path_initial_list = generate_random_tuple(category_num=category_num, dim=dim, num=3)

    print(path_initial_list)

    for path in path_initial_list:
        print("hello")
        temp  = convert_tensor_index_to_map(path, map_targeted.shape)
        print(temp)
        score = objective_function(temp)
        print(f'score: {score}')
        print()






# import numpy as np
# from tensorly.cp_tensor import cp_to_tensor
# from tensorly.decomposition import non_negative_parafac

# # from .utils_warcraft import generate_random_tuple, WarcraftObjective
# from utils_warcraft import generate_random_tuple, convert_tensor_index_to_map, WarcraftObjective


# if __name__ == "__main__":

#     map_targeted = np.array([[1, 4], [2, 1]])

#     map_targeted_scaled = map_targeted / np.sum(map_targeted)

#     settings = {
#         "name": "test" * 10,
#         "seed": 0,
#         "category_num": 7,
#         "iter": 5,
#         "cp_settings": {
#             "dim": len(map_targeted.flatten()),
#             "rank": 2,
#             "als_iterations": 100,
#         },
#     }

#     dim = settings["cp_settings"]["dim"]
#     category_num = settings["category_num"]

#     objective_function = WarcraftObjective(map_targeted_scaled)

#     tensor_targeted_bool = np.zeros((category_num,) * dim)

#     path_initial_list = generate_random_tuple(category_num=category_num, dim=dim, num=3)

#     print(path_initial_list)

#     for path in path_initial_list:
#         print("hello")
#         temp  = convert_tensor_index_to_map(path, map_targeted.shape)
#         print(temp)
#         score = objective_function(temp)
#         print(f'score: {score}')
#         print()