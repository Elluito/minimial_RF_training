#!/bin/bash -l


run_pruning() {
model=$1
dataset=$2
directory=$3
data_path=$4

echo "model ${model} and dataset ${dataset}"
pruning_rates=("0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
  if [ "${5}" -gt 0 ]
  then
#      rf_levels=("3" "4" "5" "6")
       rf_levels=("2" "5" "8")

  else
        rf_levels=("7" "8" "9" "10")
  fi

levels_max=${#rf_levels[@]}                                  # Take the length of that array
number_pruning_rates=${#pruning_rates[@]}                            # Take the length of that array
for ((idxA=0; idxA<number_pruning_rates; idxA++)); do                # iterate idxA from 0 to length
for ((idxB=0; idxB<levels_max; idxB++));do              # iterate idxB from 0 to length
./run_pruning.sh "${model}" "${dataset}" 4  "${rf_levels[$idxB]}" "normal" "${directory}" "${pruning_rates[$idxA]}" 1 "${data_path}" &
done
done
}

# The arguments are : model,  dataset, path to the output of training, data_path (were imagenet is),



run_pruning "resnet50" "imagenet" $1 $2 1
#run_pruning "resne50" "imagenet" "${HOME}/checkpoints" 0

