import sys
from constants import PATH_INFO
sys.path.append(PATH_INFO.get('project_dir'))

import re
from src.utils_experiment import extract_info_from_filename



if __name__ == '__main__':
    test_filenames = [
        ("/path/to/2024-07-21_Ackley_v1.py", "2024-07-21", "Ackley"),
        ("/another/path/2024-08-01_Rosenbrock_v2.py", "2024-08-01", "Rosenbrock"),
        ("no_date_function.py", None, None),
        ("2024-07-21_no_version.py", None, None),
        ("/Users/keisukeonoue/ws/constrained_BO/experiments/2024-07-21_Ackley_v1.py", "2024-07-21", "Ackley")
    ]
    
    for filename, expected_date, expected_function in test_filenames:
        date, objective_function = extract_info_from_filename(filename)
        assert date == expected_date, f"Test failed for {filename}: expected date {expected_date}, got {date}"
        assert objective_function == expected_function, f"Test failed for {filename}: expected function {expected_function}, got {objective_function}"
        print(filename)
        print(f"date: {date}, objective_function: {objective_function}")
        print(f"Test passed for {filename}")
        print()

    print("All tests passed.")