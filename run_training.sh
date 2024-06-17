#!/bin/bash -l
#############################################################
#     Training a model with specific RF
#############################################################
#
python train_CIFAR10.py --record_time --batch_size 128 --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9  --save_folder "${10}" --data_folder "${11}" --use_wandb "${12}"
