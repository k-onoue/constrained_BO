def get_counterpart(
    direction: str # ex: 'a'
) -> str: # ex: 'c'
    pair_dict = {
        'a': 'c',
        'c': 'a',
        'b': 'd',
        'd': 'b'
    }
    return pair_dict[direction]


def judge_continuity(
    d_from: str, # ex1: 'a', ex2: 'a'
    current_direction: str # ex1: 'ad', ex2: 'bc'
) -> bool: # ex1: True, ex2: False
    d_counterpart = get_counterpart(d_from)
    return d_counterpart in current_direction

    
def get_next_coordinate(
    d_to: str, # ex: 'a'
    current_coordinate: tuple[int, int] # ex: (0, 0)
) -> tuple[int, int]: # ex: (-1, 0)
    
    update_dict = {
        'a': (-1, 0),
        'b': (0, -1),
        'c': (0, 1),
        'd': (1, 0)
    }

    return (
        current_coordinate[0] + update_dict[d_to][0], 
        current_coordinate[1] + update_dict[d_to][1]
    )

    

def judge_location_validity(
    current: tuple[int, int], # ex1: (-1, 0), ex2: (1, 2)
    shape: tuple[int, int] # ex1: (3, 3), ex2: (3, 3)
) -> bool: # ex1: False, ex2: True
    return (
        0 <= current[0] < shape[0] and
        0 <= current[1] < shape[1]
    )


# バグってる（多分）
def get_d_to(
    d_from: str, # ex1: 'a', ex2: 'd'
    current_direction: str # ex1: 'ad', ex2: ''
) -> str:
    if current_direction[0] != d_from:
        return current_direction[0] # ex1: 'd'
    else:
        return current_direction[1]


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