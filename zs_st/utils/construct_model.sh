#!/bin/bash

experiment_path=$1
decoder_path=$2

ckpt_name=${3-avg_best_10_checkpoint}

encoder_path=${experiment_path}/ckpts/${ckpt_name}.pt

zs_model_path=${experiment_path}/ckpts_zs/${ckpt_name}
mkdir -p $zs_model_path

echo "Constructing model for zero-shot"
fairseq-hydra-train \
    --config-dir ${ZS_ROOT}/zs_st \
    --config-name construct_model_config.yaml \
    checkpoint.save_dir=${zs_model_path} \
    model.encoder_path=${encoder_path} \
    model.decoder_path=${decoder_path} \
    hydra.run.dir=${experiment_path}/tmp_zs_hydra_dir