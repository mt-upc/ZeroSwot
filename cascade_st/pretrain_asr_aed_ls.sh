#!/bin/bash

DATASET=LibriSpeech

lr=2e-4
embed_dim=768
ffn_embed_dim_mult=4
num_layers=6

ffn_embed_dim=$((embed_dim * ffn_embed_dim_mult))
mask_strategy=orig

if [ $mask_strategy == less ]; then
    mask_length=10
    mask_prob=0.1
    mask_channel_length=32
    mask_channel_prob=0.1
elif [ $mask_strategy == stnd ]; then
    mask_length=10
    mask_prob=0.25
    mask_channel_length=64
    mask_channel_prob=0.1
elif [ $mask_strategy == more ]; then
    mask_length=10
    mask_prob=0.3
    mask_channel_length=64
    mask_channel_prob=0.2
elif [ $mask_strategy == orig ]; then
    mask_length=10
    mask_prob=0.5
    mask_channel_length=64
    mask_channel_prob=0.25
fi

num_heads=8
if [ $embed_dim == 1024 ]; then
    num_heads=16
fi

data_root=$DATA_ROOT/asr/${DATASET}

max_tokens=2000000
base_update_freq=16

experiment_name=asr-aed_ls_70M

experiment_path=${SAVE_DIR}/asr_models/${experiment_name}
ckpt_dir=${experiment_path}/ckpts
mkdir -p ${ckpt_dir}

export WANDB_NAME=$experiment_name
export WANDB_DIR=${experiment_path}

n_cpus=$(eval nproc)
n_gpus=$(python -c "import torch; print(torch.cuda.device_count())")

if [ $n_gpus -eq 1 ]; then
    num_workers=2
    bs_multiple=8
else
    num_workers=1
    bs_multiple=1
fi

# check if avg_best_10_checkpoints.pt exists
if [ -f "${ckpt_dir}/avg_best_10_checkpoints.pt" ]; then
    echo "Training has finished and evaluation has been performed."
    exit 0
fi

if [ -f "${ckpt_dir}/crash.pt" ]; then
    rm ${ckpt_dir}/crash.pt
fi

if [ -f "${ckpt_dir}/checkpoint_last.pt" ]; then
    rm ${ckpt_dir}/checkpoint_last.pt
fi

if [ -L "${ckpt_dir}/checkpoint_last.pt" ]; then
    rm ${ckpt_dir}/checkpoint_last.pt
fi

# check if there are any .pt files
if [ -n "$(find ${ckpt_dir} -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)" ]; then
    # get the last .pt file according to the timestamp of creation
    last_ckpt=$(ls --time-style=full-iso -t ${ckpt_dir}/*.pt | head -n1)
    ln -s $last_ckpt ${ckpt_dir}/checkpoint_last.pt
    
    echo "Continuing training from file: $last_ckpt"

else
    echo "No checkpoint files found. Initializing training from scratch."
fi

fairseq-train $data_root \
--save-dir ${experiment_path}/ckpts/ \
--wandb-project $WANDB_PROJECT \
--log-interval 100 \
--empty-cache-freq 100 \
--config-yaml config_asr_comb_16000.yaml \
--train-subset train-other-500_asr,train-clean-100_asr,train-clean-360_asr \
--valid-subset dev-clean_asr,dev-other_asr \
--num-workers $num_workers \
--max-tokens $max_tokens \
--max-tokens-valid $((max_tokens * 2)) \
--max-update 50_000 \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--keep-best-checkpoints 10 \
--patience 10 \
--early-stop-metric nll_loss \
--best-checkpoint-metric nll_loss \
--no-last-checkpoints \
--no-epoch-checkpoints \
--no-absolute-best-checkpoints \
--no-interval-updates-checkpoints \
--validate-interval 250 \
--save-interval 999_999 \
--save-interval-updates 250 \
--arch wav2vec_seq2seq \
--optimizer adam \
--lr $lr \
--lr-scheduler inverse_sqrt \
--warmup-updates 1000 \
--seed 42 \
--fp16 \
--max-source-positions 400_000 \
--decoder-embed-dim $embed_dim \
--decoder-ffn-embed-dim $ffn_embed_dim \
--decoder-layers $num_layers \
--decoder-attention-heads $num_heads \
--decoder-normalize-before \
--decoder-dropout 0.1 \
--decoder-attention-dropout 0.1 \
--decoder-activation-dropout 0.1 \
--share-decoder-input-output-embed \
--w2v-path $MODELS_ROOT/wav2vec2/wav2vec_vox_new.pt \
--activation-dropout 0.1 \
--apply-mask \
--mask-length $mask_length \
--mask-prob $mask_prob \
--mask-channel-length $mask_channel_length \
--mask-channel-prob $mask_channel_prob \
--skip-invalid-size-inputs-valid-test \
--skip-remainder-batch \
--required-batch-size-multiple $bs_multiple \
--update-freq $((base_update_freq / n_gpus))