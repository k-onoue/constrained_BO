def get_counterpart(direction: str) -> str:
    pair_dict = {
        'a': 'c',
        'c': 'a',
        'b': 'd',
        'd': 'b'
    }
    return pair_dict[direction]


def judge_continuity(
    d_from: str, 
    current_direction: str
) -> bool:
    d_counterpart = get_counterpart(d_from)
    return d_counterpart in current_direction

    
def get_next_coordinate(
    d_to: str, 
    current: tuple[int, int]
) -> tuple[int, int]:
    
    next_dict = {
        'a': (-1, 0),
        'b': (0, -1),
        'c': (0, 1),
        'd': (1, 0)
    }
    next_coordinate = (
        current[0] + next_dict[d_to][0], 
        current[1] + next_dict[d_to][1]
    )
    return next_coordinate
    

def judge_location_validity(
    current: tuple[int, int], 
    shape: tuple[int, int]
) -> bool:
    return (
        0 <= current[0] < shape[0] and
        0 <= current[1] < shape[1]
    )


def get_d_to(d_from: str, current: str) -> str:
    return current[0] if current[0] != d_from else current[1]


def get_d_from(d_to: str) -> str:
    d_from = get_counterpart(d_to)
    return d_from



def navigate_through_matrix(direction_matrix, start, goal):

    history = []
    current = start
    shape = direction_matrix.shape
    direction = direction_matrix[current]
    if 'bd' not in direction:
        return history 
    
    history.append(current)
    d_to = direction[1]
    next_pos = get_next_coordinate(d_to, current)
    next_pos_flag = judge_location_validity(next_pos, shape)
    continuity_flag = True

    while next_pos_flag and continuity_flag and current != goal:
        d_from = get_d_from(d_to)
        current = next_pos
        direction = direction_matrix[current]
        d_to = get_d_to(d_from, direction)
        history.append(current)
        if current == goal:
            break
        next_pos = get_next_coordinate(d_to, current)
        next_pos_flag = judge_location_validity(next_pos, shape)
        continuity_flag = judge_continuity(d_to, direction_matrix[next_pos])

    return history