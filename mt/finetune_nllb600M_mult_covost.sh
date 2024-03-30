#!/bin/bash

lr=2e-4
drop=0.2
attn_drop=0.1
act_drop=0.0
label_smoothing=0.1
max_tokens=2048
base_update_freq=16

DATASET=$COVOST_NAME
MODEL_ID=nllb-200-distilled-600M
CODE_PAIRS=eng_Latn-arb_Arab,eng_Latn-cat_Latn,eng_Latn-cym_Latn,eng_Latn-deu_Latn,eng_Latn-est_Latn,eng_Latn-pes_Arab,eng_Latn-ind_Latn,eng_Latn-jpn_Jpan,eng_Latn-lvs_Latn,eng_Latn-khk_Cyrl,eng_Latn-slv_Latn,eng_Latn-swe_Latn,eng_Latn-tam_Taml,eng_Latn-tur_Latn,eng_Latn-zho_Hans

experiment_name=mt_nllb600M_${DATASET}_mult
experiment_path=${SAVE_DIR}/mt_models/${experiment_name}
ckpt_dir=${experiment_path}/ckpts
mkdir -p $ckpt_dir

n_cpus=$(eval nproc)
n_gpus=$(python -c "import torch; print(torch.cuda.device_count())")

export WANDB_NAME=$experiment_name
export WANDB_DIR=${experiment_path}

model_root=${MODELS_ROOT}/nllb
data_root=${DATA_ROOT}/mt/${DATASET}/multilingual

restore_file=${model_root}/${MODEL_ID}.pt

fairseq-train $data_root/data-bin \
--wandb-project $WANDB_PROJECT \
--save-dir ${experiment_path}/ckpts/ \
--log-format json \
--lang-dict ${model_root}/lang_dict.txt \
--log-interval 50 \
--fp16 \
--reset-optimizer \
--reset-dataloader \
--share-decoder-input-output-embed \
--share-all-embeddings \
--encoder-normalize-before \
--decoder-normalize-before \
--encoder-embed-dim 1024 \
--encoder-ffn-embed-dim 4096 \
--encoder-attention-heads 16 \
--encoder-layers 12 \
--decoder-layers 12 \
--arch transformer \
--task translation_multi_simple_epoch  \
--sampling-method "temperature" \
--sampling-temperature 1.0 \
--encoder-langtok "src" \
--decoder-langtok \
--lang-pairs $CODE_PAIRS \
--criterion label_smoothed_cross_entropy \
--label-smoothing $label_smoothing \
--optimizer adam \
--adam-eps 1e-06 \
--adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt \
--lr $lr \
--warmup-updates 2000 \
--dropout $drop \
--attention-dropout $attn_drop \
--activation-dropout $act_drop \
--weight-decay 0.0 \
--max-tokens $max_tokens \
--max-tokens-valid $((max_tokens * 2)) \
--update-freq $(($base_update_freq / $n_gpus)) \
--num-workers 0 \
--keep-best-checkpoints 10 \
--no-last-checkpoints \
--no-epoch-checkpoints \
--no-last-checkpoints \
--no-absolute-best-checkpoints \
--no-interval-updates-checkpoints \
--skip-invalid-size-inputs-valid-test \
--patience 20 \
--validate-interval 500 \
--save-interval-updates 500 \
--save-interval 999999 \
--seed 42 \
--best-checkpoint-metric nll_loss \
--early-stop-metric nll_loss \
--restore-file $restore_file