#!/bin/bash

experiment_path=$1
n_avg=${2-10}
remove_ckpts=${3-"true"}

checkpoint_name=avg_best_${n_avg}_checkpoint
path_to_ckpt=${experiment_path}/ckpts/${checkpoint_name}.pt

if [ ! -f $path_to_ckpt ]; then
    python ${ZS_ROOT}/zs_st/utils/find_best_ckpts.py \
        ${experiment_path}/ckpts $n_avg
    inputs=$(head -n 1 ${experiment_path}/ckpts/best_${n_avg}.txt)

    python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
        --inputs $inputs \
        --output $path_to_ckpt
fi

if [[ -f $path_to_ckpt ]] && [[ $remove_ckpts == "true" ]]; then
    find "${experiment_path}"/ckpts/*.pt -type f ! -name '*avg*' -delete
fi