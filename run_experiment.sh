#!/bin/bash

python3 experiments/2024-08-30_botorch/Simple_5d_unconstrained_ReLU.py &
python3 experiments/2024-08-30_botorch/Simple_5d_unconstrained_tanh.py &

wait

echo "Experiments done!"
