#!/bin/bash
#SBATCH --job-name=python_parallel       # ジョブ名
#SBATCH --gres=gpu:1                     # 各タスクにGPUを1つ使用
#SBATCH --cpus-per-task=10               # 各タスクに10コアを割り当てる
#SBATCH --time=03:00:00                  # ジョブの最大実行時間
#SBATCH --partition=gpu_short            # 使用するパーティション
#SBATCH --ntasks=4                       # 並列に実行するタスクの数

# config.ini ファイルを上書き
config_file="config.ini"

config_content="[paths]
project_dir = /work/keisuke-o/ws/constrained_BO
data_dir = %(project_dir)s/data
results_dir = %(project_dir)s/results"

# config.ini ファイルの上書き
echo "$config_content" > $config_file

# 上書き完了のメッセージと内容の確認
echo "config.ini has been overwritten with the following content:"
cat $config_file

# Pythonファイルを並列に実行（出力ファイルを各ジョブごとに分ける）
srun --exclusive -N1 --ntasks=1 --output=results/output_gp_%j.txt python3 experiments/2024-08-09_optuna/Warcraft_gp.py &
srun --exclusive -N1 --ntasks=1 --output=results/output_nsga_%j.txt python3 experiments/2024-08-09_optuna/Warcraft_nsga.py &
srun --exclusive -N1 --ntasks=1 --output=results/output_random_%j.txt python3 experiments/2024-08-09_optuna/Warcraft_random.py &
srun --exclusive -N1 --ntasks=1 --output=results/output_tpe_%j.txt python3 experiments/2024-08-09_optuna/Warcraft_tpe.py &

# 全てのジョブが完了するまで待機
wait
