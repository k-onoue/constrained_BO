import sys
project_dir_path = "/Users/keisukeonoue/ws/constrained_BO"
sys.path.append(project_dir_path)

import numpy as np
from src.utils import navigate_through_matrix



if '__main__' == __name__:

    # # Example usage:
    # direction_matrix = np.array([
    #     ['ad', 'oo', 'oo'],
    #     ['bc', 'ac', 'ad'],
    #     ['oo', 'oo', 'bd']
    # ])

    # # Example usage:
    # direction_matrix = np.array([
    #     ['ad', 'oo', 'oo'],
    #     ['bc', 'oo', 'ad'],
    #     ['oo', 'oo', 'bd']
    # ])

    # Example usage:
    direction_matrix = np.array([
        ['bd', 'oo', 'oo'],
        ['bc', 'ac', 'ad'],
        ['oo', 'oo', 'bd']
    ])

    start = (0,0)
    goal = (2,2)
    history = navigate_through_matrix(direction_matrix, start, goal)
    print(f'history: {history}')

