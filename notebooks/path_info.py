import os

temp = __file__.rfind('/', 0, __file__.rfind('/'))
prefix = __file__[:temp]

PROJECT_DIR = prefix
DATA_DIR = os.path.join(prefix, "data")
EXPT_RESULT_DIR = os.path.join(prefix, "results")


# if __name__ == "__main__":
#     print(PROJECT_DIR)
#     print(DATA_DIR)
#     print(EXPT_RESULT_DIR)

#     """
#     >>>
#     /Users/keisukeonoue/ws/constrained_BO
#     /Users/keisukeonoue/ws/constrained_BO/data
#     /Users/keisukeonoue/ws/constrained_BO/results
#     """