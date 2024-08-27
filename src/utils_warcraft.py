import torch


def get_opposite(direction: str) -> str:
    """
    Examples
    --------
    >>> get_opposite('a')
    'c'
    """
    pair_dict = {"a": "c", "c": "a", "b": "d", "d": "b"}
    return pair_dict.get(direction, "")


def judge_continuity(d_from: str, current_direction: str) -> bool:
    """
    Examples
    --------
    >>> judge_continuity('a', 'ad')
    False
    >>> judge_continuity('a', 'bc')
    True
    """
    d_opposite = get_opposite(d_from)
    return d_opposite in current_direction


def get_next_coordinate(
    d_to: str, current_coordinate: tuple[int, int]
) -> tuple[int, int]:
    """
    Examples
    --------
    >>> get_next_coordinate('a', (0, 0))
    (-1, 0)
    """
    update_dict = {"a": (-1, 0), "b": (0, -1), "c": (0, 1), "d": (1, 0)}
    delta = update_dict.get(d_to, (0, 0))
    return (current_coordinate[0] + delta[0], current_coordinate[1] + delta[1])


def judge_location_validity(current: tuple[int, int], shape: tuple[int, int]) -> bool:
    """
    Examples
    --------
    >>> judge_location_validity((-1, 0), (3, 3))
    False
    >>> judge_location_validity((1, 2), (3, 3))
    True
    """
    return 0 <= current[0] < shape[0] and 0 <= current[1] < shape[1]


def get_d_to(d_from: str, current_direction: str) -> str:
    """
    Examples
    --------
    >>> get_d_to('a', 'ad')
    'd'
    """
    return (
        current_direction[1] if current_direction[0] == d_from else current_direction[0]
    )


def navigate_through_matrix(direction_matrix, start, goal):
    history = []
    current = start
    shape = direction_matrix.shape

    if direction_matrix[current] != "bd":
        return history

    history.append(current)
    d_to = "d"
    next_pos = get_next_coordinate(d_to, current)

    while judge_location_validity(next_pos, shape) and current != goal:
        if not judge_continuity(d_to, direction_matrix[next_pos]):
            break

        current = next_pos
        history.append(current)
        if current == goal:
            break

        direction = direction_matrix[current]
        d_from = get_opposite(d_to)
        d_to = get_d_to(d_from, direction)
        next_pos = get_next_coordinate(d_to, current)

    return history


def manhattan_distance(coord1: tuple[int, int], coord2: tuple[int, int]) -> int:
    """
    Examples
    --------
    >>> manhattan_distance((0, 0), (3, 3))
    6
    """
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def generate_initial_data(
    objective_function: callable,
    dataset_size: int,
    shape: tuple[int, int],
) -> torch.tensor:
    values = torch.tensor([-3, -2, -1, 0, 1, 2, 3])
    n, m = shape
    X_train = values[torch.randint(0, len(values), (dataset_size, n, m))]
    y_train = torch.stack([objective_function(x) for x in X_train]).unsqueeze(-1)
    return X_train, y_train