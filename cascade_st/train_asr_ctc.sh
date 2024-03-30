#!/bin/bash

exp_config=$1

export WANDB_NAME=$(basename $exp_config .yaml)

exp_path=${SAVE_DIR}/asr_models/${WANDB_NAME}
ckpt_dir=${exp_path}/ckpts
mkdir -p $ckpt_dir

if [ -f "${ckpt_dir}/avg_best_10_checkpoints.pt" ]; then
    echo "Training has finished and evaluation has been performed."
    exit 0
fi

if [ -f "${ckpt_dir}/crash.pt" ]; then
    rm ${ckpt_dir}/crash.pt
fi

if [ -L "${ckpt_dir}/checkpoint_last.pt" ]; then
    rm ${ckpt_dir}/checkpoint_last.pt
fi

find ${ckpt_dir} -maxdepth 1 -name '*.pt.tmp' -delete

# check if there are any .pt files
if [ -n "$(find ${ckpt_dir} -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)" ]; then
    # get the last .pt file according to the timestamp of creation
    last_ckpt=$(ls --time-style=full-iso -t ${ckpt_dir}/*.pt | head -n1)
    ln -s $last_ckpt ${ckpt_dir}/checkpoint_last.pt
    
    echo "Continuing training from file: $last_ckpt"

else
    echo "No checkpoint files found. Initializing training from scratch."

fi

base_update_freq=$(grep 'update_freq:' ${ZS_ROOT}/cascade_st/${WANDB_NAME}.yaml | awk '{print $2}' | sed 's/[^0-9]//g')
n_gpus=$(python -c "import torch; print(torch.cuda.device_count())")

if [ $((base_update_freq % n_gpus)) -gt 0 ]; then
    echo "WARNING: update_freq ($base_update_freq) is not divisible by n_gpus ($n_gpus)."
    exit 1
fi

if [ $((base_update_freq % n_gpus)) -ne 0 ]; then
    echo "Error: \$base_update_freq is not perfectly divisible by \$n_gpus"
    exit 1
fi

update_freq=$(( $base_update_freq / $n_gpus ))
echo "Using update_freq: $update_freq (base: $base_update_freq) and n_gpus: $n_gpus"

fairseq-hydra-train \
    --config-dir ${ZS_ROOT}/cascade_st \
    --config-name ${WANDB_NAME}.yaml \
    optimization.update_freq=[${update_freq}] \
    dataset.num_workers=$([ $n_gpus -gt 1 ] && echo 1 || echo 2)