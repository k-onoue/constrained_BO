#!/bin/bash
#SBATCH --job-name=python_parallel    # ジョブ名
#SBATCH --output=results/output_%j.txt  # 出力ファイルの保存先
#SBATCH --gres=gpu:1                   # GPUを1つ使用
#SBATCH --cpus-per-task=4              # 各タスクに割り当てるCPUコア数
#SBATCH --time=03:00:00                # ジョブの最大実行時間
#SBATCH --partition=gpu_short          # 使用するパーティション

# Pythonファイルを並列に実行
srun --exclusive --ntasks=1 python3 experiments/2024-08-09_optuna/Warcraft_gp.py &
srun --exclusive --ntasks=1 python3 experiments/2024-08-09_optuna/Warcraft_nsga.py &
srun --exclusive --ntasks=1 python3 experiments/2024-08-09_optuna/Warcraft_random.py &
srun --exclusive --ntasks=1 python3 experiments/2024-08-09_optuna/Warcraft_tpe.py &

# 全てのジョブが完了するまで待機
wait