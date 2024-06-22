#!/bin/bash -l
#save_results_folder=$1
#data_folder=$2
# The order of the parameters are the following
#--model --dataset --num_workers --RF_level--type --epochs  --name  --width  --record  --save_folder --data_folder

# "training_Level_2_resnet50_small_imagenet"
#./run_training.sh "resnet50" "small_imagenet" 4 "2" "normal" 50 "recording_50" 1 1 $1 $2
# "training_Level_3_resnet50_small_imagenet"
#./run_training.sh "resnet50" "small_imagenet" 4 "3" "normal" 200 "recording_200" 1 1 $1 $2 1

# "training_Level_4_resnet50_small_imagenet"
#./run_training.sh "resnet50" "small_imagenet" 4 "4" "normal" 200 "recording_200" 1 1 $1 $2 1
# "training_Level_5_resnet50_small_imagenet"

./run_training.sh "resnet50" "imagenet" 4 "5" "normal" 50 "recording_50" 1 1 $1 $2 1
# "training_Level_6_resnet50_small_imagenet"
#./run_training.sh "resnet50" "imagenet" 4 "6" "normal" 200 "recording_200" 1 1 $1 $2 1
# "training_Level_7_resnet50_small_imagenet"
#./run_training.sh "resnet50" "imagenet" 4 "7" "normal" 200 "recording_200" 1 1 $1 $2 1
# "training_Level_8_resnet50_small_imagenet"
#./run_training.sh "resnet50" "imagenet" 4 "8" "normal" 50 "recording_50" 1 1 $1 $2 1
## "training_Level_9_resnet50_small_imagenet"
#./run_training.sh "resnet50" "small_imagenet" 4 "9" "normal" 200 "recording_200" 1 1 $1 $2
## "training_Level_10_resnet50_small_imagenet"
#./run_training.sh "resnet50" "small_imagenet" 4 "10" "normal" 200 "recording_200" 1 1 $1 $2
#python train_CIFAR10.py --batch_size 128 --model "resnet50" --dataset "small_imagenet"  --num_workers 4 --RF_level "2" --type "normal" --epochs 200  --name "recording_200"  --width 1 --record 1  --save_folder "${10}" --data_folder "${11}"
