#!/bin/bash -l
./run_training.sh "resnet50" "imagenet" 4 "10" "normal" 50 "recording_50" 1 1 $1 $2 1 $3 &&
./run_training.sh "resnet50" "imagenet" 4 "10" "normal" 50 "recording_50" 1 1 $1 $2 1 $3
