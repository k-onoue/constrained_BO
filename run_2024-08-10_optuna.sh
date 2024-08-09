#!/bin/bash

# Resource configuration
CPUS_PER_TASK=3  # Adjust the number of CPUs per task
RES_DIR_NAME="temp"  # Directory name for storing results
PARTITION="gpu_short"  # Partition name
TIME="09:00:00"  # Maximum execution time

# Create results directory if it doesn't exist
mkdir -p results/$RES_DIR_NAME

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/constrained_BO
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results"

# Overwrite config.ini file
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Run multiple Python files in parallel using sbatch
sbatch --job-name=benchmark_${PARTITION} \
       --output=results/$RES_DIR_NAME/output_gp_%j.txt \
       --gres=gpu:1 \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-08-10_optuna/Warcraft_gp.py"

sbatch --job-name=benchmark_${PARTITION} \
       --output=results/$RES_DIR_NAME/output_nsga_%j.txt \
       --gres=gpu:1 \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-08-10_optuna/Warcraft_nsga.py"

sbatch --job-name=benchmark_${PARTITION} \
       --output=results/$RES_DIR_NAME/output_random_%j.txt \
       --gres=gpu:1 \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-08-10_optuna/Warcraft_random.py"

sbatch --job-name=benchmark_${PARTITION} \
       --output=results/$RES_DIR_NAME/output_tpe_%j.txt \
       --gres=gpu:1 \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-08-10_optuna/Warcraft_tpe.py"