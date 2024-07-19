import sys
from constants import *
# sys.path.append(PROJECT_DIR_PATH)
sys.path.append(PATH_INFO.get('project_dir'))

import numpy as np
from src.utils import navigate_through_matrix
from src.utils import get_counterpart
from src.utils import get_next_coordinate
from src.utils import judge_continuity
from src.utils import judge_location_validity
from src.utils import get_d_to
from src.utils import get_d_from


def test_functions():
    # Test get_counterpart function
    assert get_counterpart('a') == 'c'
    assert get_counterpart('c') == 'a'
    assert get_counterpart('b') == 'd'
    assert get_counterpart('d') == 'b'
    print("get_counterpart passed")

    # ex1 が間違ってそう
    # Test judge_continuity function
    assert judge_continuity('a', 'ad') == False
    assert judge_continuity('a', 'bc') == True
    print("judge_continuity passed")

    # Test get_next_coordinate function
    assert get_next_coordinate('a', (0, 0)) == (-1, 0)
    assert get_next_coordinate('b', (0, 0)) == (0, -1)
    assert get_next_coordinate('c', (0, 0)) == (0, 1)
    assert get_next_coordinate('d', (0, 0)) == (1, 0)
    print("get_next_coordinate passed")

    # Test judge_location_validity function
    assert judge_location_validity((-1, 0), (3, 3)) == False
    assert judge_location_validity((1, 2), (3, 3)) == True
    assert judge_location_validity((3, 3), (3, 3)) == False
    assert judge_location_validity((0, 0), (3, 3)) == True
    print("judge_location_validity passed")




if '__main__' == __name__:

    test_functions()




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


