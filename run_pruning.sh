#!/bin/bash -l

#############################################################
#     One shot with specific pruning rate results
#############################################################

python prune_models.py --name "recording_50" --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --folder $6 --pruning_rate $7 --experiment $8 --data_folder $9
