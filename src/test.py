from utils_warcraft import get_opposite
from utils_warcraft import judge_continuity
from utils_warcraft import get_next_coordinate
from utils_warcraft import judge_location_validity
from utils_warcraft import get_d_to
from utils_warcraft import get_d_from
from utils_warcraft import navigate_through_matrix



if __name__ == '__main__':

    import numpy as np

    # Example usage: ----------------------------------------------------
    direction_matrix = np.array([
        ['ad', 'oo', 'oo'],
        ['bc', 'ac', 'ad'],
        ['oo', 'oo', 'bd']
    ])

    start = (0,0)
    goal = (2,2)
    history = navigate_through_matrix(direction_matrix, start, goal)
    print(f'history: {history}')


    # Example usage: ----------------------------------------------------
    direction_matrix = np.array([
        ['ad', 'oo', 'oo'],
        ['bc', 'oo', 'ad'],
        ['oo', 'oo', 'bd']
    ])

    start = (0,0)
    goal = (2,2)
    history = navigate_through_matrix(direction_matrix, start, goal)
    print(f'history: {history}')


    # Example usage: ----------------------------------------------------
    direction_matrix = np.array([
        ['bd', 'oo', 'oo'],
        ['bc', 'ac', 'ad'],
        ['oo', 'oo', 'bd']
    ])

    start = (0,0)
    goal = (2,2)
    history = navigate_through_matrix(direction_matrix, start, goal)
    print(f'history: {history}')