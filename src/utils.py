


def judge_continuity(d_from: tuple[int, int], current_direction: str) -> bool:
    if d_from == 'a' and 'c' in current_direction:
        return True
    elif d_from == 'c' and 'a' in current_direction:
        return True
    elif d_from == 'b' and 'd' in current_direction:
        return True 
    elif d_from == 'd' and 'b' in current_direction:
        return True
    else:
        return False
    
    
def get_next_coordinate(d_to: str, current: tuple[int, int]) -> tuple[int, int]:
    next_dict = {
        'a': (-1, 0),
        'b': (0, -1),
        'c': (0, 1),
        'd': (1, 0)
    }
    next_coordinate = (current[0] + next_dict[d_to][0], current[1] + next_dict[d_to][1])
    return next_coordinate


def judge_location_validity(current: tuple[int, int], shape: tuple[int, int]) -> bool:
    if current[0] < 0 or current[0] >= shape[0] or current[1] < 0 or current[1] >= shape[1]:
        return False
    else:
        return True
    

def get_d_to(d_from: str, current: str) -> str:
    d_to = current[0] if current[0] != d_from else current[1]
    return d_to


def get_d_from(d_to: str) -> str:
    if d_to == 'a':
        d_from = 'c'
    elif d_to == 'b':
        d_from = 'd'
    elif d_to == 'c':
        d_from = 'a'
    elif d_to == 'd':
        d_from = 'b'
    else:
        raise ValueError
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