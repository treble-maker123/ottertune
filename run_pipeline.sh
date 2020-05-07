#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cs645

# remove the previous run files
echo "Removing previous run files"
rm outputs/b_s_matrix_*
rm outputs/pruned_metrics.txt
rm outputs/y_and_y_hat.csv

echo "Removing previous workload models"
rm models/wl_*

export prune="python3 prune.py \
--output-path=outputs/pruned_metrics.txt \
--num-factors=5" # \
#--use-k \
#--k=10"

export workload="python3 train_workload_gprs.py"

export latency="python3 latency_prediction.py \
--pruned-metrics=outputs/pruned_metrics.txt"

echo "Starting pruning script"
$prune

echo "Starting training workload GPRs"
$workload

echo "Starting latency prediction"
$latency
