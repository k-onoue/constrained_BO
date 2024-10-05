import configparser
import sys

# Load configuration
config = configparser.ConfigParser()
config_path = "./config.ini"
config.read(config_path)
PROJECT_DIR = config["paths"]["project_dir"]
LOG_DIR = config["paths"]["logs_dir"]
sys.path.append(PROJECT_DIR)

from src.utils_experiment import set_logger
from src.utils_warcraft import (
    generate_random_tuple,
    convert_path_index_to_arr,
    WarcraftObjective,
)
from src.utils_cp import InputManager, ParafacSampler, suggest_ucb_candidates

__all__ = [
    "set_logger",
    "generate_random_tuple",
    "convert_path_index_to_arr",
    "WarcraftObjective",
    "InputManager",
    "ParafacSampler",
    "suggest_ucb_candidates",
]
