import numpy as np


def judge_continuity(d_from: tuple[int, int], current_direction: str) -> bool:
    print('judge_continuity')
    print(f'd_from: {d_from}')
    print(f'current_direction: {current_direction}')
    print()
    print()
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






if '__name__' == '__main__':

    direction_matrix = np.array([
        ['ad', 'oo', 'oo'],
        ['bc', 'ac', 'ad'],
        ['oo', 'oo', 'bd']
    ])


    history = []

    current = (0,0)
    goal = (2,2)

    shape = direction_matrix.shape

    history.append(current)
    print(f'history: {history}')

    # initial_valid_directions = ['ac', 'ad', 'bc', 'bc']

    direction = direction_matrix[current]
    print(f'direction: {direction}')
    d_to = direction[1]
    print(f'd_to: {d_to}')

    next_pos = get_next_coordinate(d_to, current)

    next_pos_flag = judge_location_validity(next_pos, shape)
    # continuity_flag = judge_continuity(d_to, direction_matrix[current])
    continuity_flag = True

    print(f'next_pos: {next_pos}')
    print(f'next_pos_flag: {next_pos_flag}')
    print(f'continuity_flag: {continuity_flag}')



    # while judge_location_validity(current, shape) and judge_continuity(d_to, direction_matrix[current]) and current != goal:
    while next_pos_flag and continuity_flag and current != goal:    
        d_from = d_to
        current = get_next_coordinate(d_to, current)
        direction = direction_matrix[current]
        d_to = get_d_to(d_from, direction)

        print()
        print(f'in while loop')
        print(f'current: {current}')
        print(f'direction: {direction}')
        print(f'd_to: {d_to}')

        # next_pos = get_next_coordinate(d_to, current)

        # history.append(current)

        next_pos_flag = judge_location_validity(next_pos, shape)
        continuity_flag = judge_continuity(d_to, direction_matrix[current])

        print(f'next_pos: {next_pos}')
        print(f'next_pos_flag: {next_pos_flag}')
        print(f'continuity_flag: {continuity_flag}')