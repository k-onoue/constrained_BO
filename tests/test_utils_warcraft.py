import sys
from path_info import PROJECT_DIR

sys.path.append(PROJECT_DIR)

import numpy as np
from src.utils_warcraft import get_opposite
from src.utils_warcraft import judge_continuity
from src.utils_warcraft import get_next_coordinate
from src.utils_warcraft import judge_location_validity
from src.utils_warcraft import get_d_to
from src.utils_warcraft import manhattan_distance
from src.utils_warcraft import navigate_through_matrix


# Tests: ----------------------------------------------------
if __name__ == "__main__":
    # get_opposite のテスト
    assert get_opposite("a") == "c"
    assert get_opposite("c") == "a"
    assert get_opposite("b") == "d"
    assert get_opposite("d") == "b"
    assert get_opposite("e") == ""  # 無効な入力

    # judge_continuity のテスト
    assert judge_continuity("a", "ad") == False
    assert judge_continuity("a", "bc") == True
    assert judge_continuity("b", "bd") == True
    assert judge_continuity("c", "da") == True

    # get_next_coordinate のテスト
    assert get_next_coordinate("a", (0, 0)) == (-1, 0)
    assert get_next_coordinate("b", (0, 0)) == (0, -1)
    assert get_next_coordinate("c", (0, 0)) == (0, 1)
    assert get_next_coordinate("d", (0, 0)) == (1, 0)
    assert get_next_coordinate("a", (1, 1)) == (0, 1)

    # judge_location_validity のテスト
    assert judge_location_validity((-1, 0), (3, 3)) == False
    assert judge_location_validity((1, 2), (3, 3)) == True
    assert judge_location_validity((3, 3), (3, 3)) == False
    assert judge_location_validity((2, 2), (3, 3)) == True

    # get_d_to のテスト
    assert get_d_to("a", "ad") == "d"
    assert get_d_to("b", "bc") == "c"
    assert get_d_to("c", "ca") == "a"
    assert get_d_to("d", "db") == "b"

    # manhattan_distance のテスト
    assert manhattan_distance((0, 0), (3, 3)) == 6

    print("All tests passed.")

    import numpy as np

    # Example usage: ----------------------------------------------------
    direction_matrix = np.array(
        [["ad", "oo", "oo"], ["bc", "ac", "ad"], ["oo", "oo", "bd"]]
    )

    start = (0, 0)
    goal = (2, 2)
    history = navigate_through_matrix(direction_matrix, start, goal)
    print(f"history: {history}")

    # Example usage: ----------------------------------------------------
    direction_matrix = np.array(
        [["ad", "oo", "oo"], ["bc", "oo", "ad"], ["oo", "oo", "bd"]]
    )

    start = (0, 0)
    goal = (2, 2)
    history = navigate_through_matrix(direction_matrix, start, goal)
    print(f"history: {history}")

    # Example usage: ----------------------------------------------------
    direction_matrix = np.array(
        [["bd", "oo", "oo"], ["bc", "ac", "ad"], ["oo", "oo", "bd"]]
    )

    start = (0, 0)
    goal = (2, 2)
    history = navigate_through_matrix(direction_matrix, start, goal)
    print(f"history: {history}")
