#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
p_ratios=(10 20 30 40 50 60 70 80 90 100)

for p_ratio in ${p_ratios[@]}
do
    echo "Train for p_ratio $p_ratio"
    python -u yelp_eps_torch.py --p_ratio=$p_ratio
done