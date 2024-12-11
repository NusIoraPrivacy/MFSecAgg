#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
epsilons=(0.0001 0.001 0.01 0.1 1)

for eps in ${epsilons[@]}
do
    echo "Train for epsilon $eps"
    python -u yelp_eps_torch.py --pb_eps=$eps
done