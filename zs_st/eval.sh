#!/bin/bash

experiment_path=$1
decoder_path=$2
dataset_name=$3
split=$4
tgt_lang=$5

ckpt_name=${6-avg_best_10_checkpoint}

device_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
if [[ $device_name == *"3090"* ]]; then
    max_tokens=2400000
    max_source_positions=2400000
else
    max_tokens=1500000
    max_source_positions=1500000
fi

lenpen=1.0

split=${split}_st

encoder_path=${experiment_path}/ckpts/${ckpt_name}.pt
zs_model_path=${experiment_path}/ckpts_zs/${ckpt_name}

data_root=${DATA_ROOT}/st/${dataset_name}/en-${tgt_lang}
file_name=${ckpt_name}-${dataset_name}-${tgt_lang}-${split}.txt
tmp_res_path=${experiment_path}/results_zs/${ckpt_name}-${dataset_name}-${tgt_lang}-${split}
generation_path_A=${tmp_res_path}/generate-${split}.txt
generation_path_B=${experiment_path}/results_zs/${file_name}

tokenizer=13a
if [[ $tgt_lang == zh-CN ]] || [[ $tgt_lang == ja ]]; then
    tokenizer=char
fi

if [ -s $generation_path_B ]; then
    exit 0
fi

mkdir -p $tmp_res_path

fairseq-generate $data_root \
    --path ${zs_model_path}/checkpoint_zs.pt \
    --config-yaml ${ZS_ROOT}/zs_st/config_st.yaml \
    --user-dir examples/extended_siamese \
    --task speech_to_text \
    --gen-subset ${split} \
    --seed 42 \
    --prefix-size 1 \
    --scoring sacrebleu \
    --max-source-positions $max_source_positions \
    --max-target-positions 1024 \
    --max-tokens $max_tokens \
    --beam 5 \
    --lenpen $lenpen \
    --results-path $tmp_res_path \
    --sacrebleu-tokenizer $tokenizer \
    --model-overrides "{'not_load_submodules':True,'encoder_path':'${encoder_path}','decoder_path':'${decoder_path}', 'max_source_positions':'${max_source_positions}'}"

mv $generation_path_A $generation_path_B
rm -rf ${tmp_res_path}

echo "Results saved at: $generation_path_B"