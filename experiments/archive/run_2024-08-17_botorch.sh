#!/bin/bash

# Resource configuration
CPUS_PER_TASK=3  # Adjust the number of CPUs per task
PARTITION="gpu_long"  # Partition name
TIME="09:30:00"  # Maximum execution time

# Create results directory if it doesn't exist
mkdir -p results/

# Create logs directory if it doesn't exist
mkdir -p logs

# Overwrite config.ini file
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/constrained_BO
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results
logs_dir = %(project_dir)s/logs"

# Overwrite config.ini file
echo "$config_content" > $config_file

# Confirm the overwrite
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Run multiple Python files in parallel using sbatch
sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_constraint_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-08-17_botorch/Griewank_3d_constrained1.py"

# Run multiple Python files in parallel using sbatch
sbatch --job-name=benchmark_${PARTITION} \
       --output=results/output_constraint_%j.txt \
       --cpus-per-task=$CPUS_PER_TASK \
       --partition=$PARTITION \
       --time=$TIME \
       --wrap="python3 experiments/2024-08-17_botorch/Griewank_3d_constrained1.py"