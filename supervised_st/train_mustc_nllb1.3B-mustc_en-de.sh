#!/bin/bash

lang_pair=en-de
experiment_name=st_mustc_nllb1.3B-mustc_${lang_pair}
experiment_path=${SAVE_DIR}/st/${experiment_name}
ckpt_dir=${experiment_path}/ckpts
zs_experiment_path=${SAVE_DIR}/speech_encoders/mustc_w2v-Lrg_nllb1.3B-mustc
decoder_path=${SAVE_DIR}/mt_models/mt_nllb1.3B_MUSTC_v1.0_mult/ckpts/avg_best_10_checkpoint.pt

export WANDB_NAME=${experiment_name}

n_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
base_update_freq=24

if [ $((base_update_freq % n_gpus)) -ne 0 ]; then
    echo "Error: \$base_update_freq is not perfectly divisible by \$n_gpus"
    exit 1
fi

update_freq=$(( $base_update_freq / $n_gpus ))

mkdir -p $ckpt_dir

fairseq-train ${DATA_ROOT}/st/MUSTC_v1.0/${lang_pair} \
--wandb-project $WANDB_PROJECT \
--save-dir ${experiment_path}/ckpts/ \
--user-dir ${FAIRSEQ_ROOT}/examples/extended_siamese \
--log-format json \
--lna \
--log-interval 50 \
--max-update 20_000 \
--config-yaml ${ZS_ROOT}/zs_st/config_st.yaml \
--train-subset train_fltr_st \
--valid-subset dev_st \
--num-workers 1 \
--required-batch-size-multiple 1 \
--batch-size 32 \
--max-tokens 1_333_333 \
--max-source-positions 400_000 \
--max-tokens-valid 2_400_000 \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--ignore-prefix-size 1 \
--label-smoothing 0.1 \
--keep-best-checkpoints 10 \
--patience 40 \
--no-epoch-checkpoints \
--no-absolute-best-checkpoints \
--no-interval-updates-checkpoints \
--no-last-checkpoints \
--best-checkpoint-metric sacrebleu \
--maximize-best-checkpoint-metric \
--scoring sacrebleu \
--early-stop-metric nll_loss \
--arch siamese_zs_s2t_model \
--optimizer adam \
--lr 4e-5 \
--update-freq $update_freq \
--lr-scheduler tri_stage \
--warmup-steps 1_000 \
--hold-steps 5_000 \
--decay-steps 14_000 \
--seed 42 \
--eval-bleu \
--prefix-size 1 \
--fp16 \
--memory-efficient-fp16 \
--validate-interval 250 \
--empty-cache-freq 250 \
--save-interval-updates 250 \
--save-interval 999999 \
--skip-remainder-batch \
--skip-invalid-size-inputs-valid-test \
--encoder-path $zs_experiment_path/ckpts/avg_best_10_checkpoint.pt \
--not-load-submodules \
--find-unused-parameters \
--decoder-path $decoder_path \
--restore-file $zs_experiment_path/ckpts_zs/avg_best_10_checkpoint/checkpoint_zs.pt \