#!/bin/bash

experiment_path=$1

dataset=${2-MUSTC_v1.0}
subset=${3-valid}
tgt_lang=${4-de}
ckpt_name=${5-avg_best_10_checkpoint}

src_lang=en
src_code=eng_Latn

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

device_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
if [[ $experiment_path == *"1.3"* ]]; then
    if [[ $device_name == *"3090"* ]]; then
        max_tokens=2048
    else
        max_tokens=1024
    fi
else
    if [[ $device_name == *"3090"* ]]; then
        max_tokens=4096
    else
        max_tokens=2048
    fi
fi

nllb_path=${MODELS_ROOT}/nllb
path_to_ckpt=${experiment_path}/ckpts/${ckpt_name}.pt
mkdir -p ${experiment_path}/results

tgt_code=${NLLB_LANG_CODES["$tgt_lang"]}

lang_pair=${src_lang}-${tgt_lang}
lang_pair_code=${src_code}-${tgt_code}
data_path=${DATA_ROOT}/mt/${dataset}/${lang_pair}

results_path=${experiment_path}/results/${dataset}/${lang_pair}/${subset}
tmp_result_path=${experiment_path}/results_tmp_${dataset}_${subset}_${tgt_lang}

mkdir -p $results_path
mkdir -p $tmp_result_path

tmp_file=${tmp_result_path}/generate-${subset}.txt

ref_file=${results_path}/ref.txt
hyp_file=${results_path}/hyp.txt
bleu_file=${results_path}/bleu.txt

if [ -s $bleu_file ]; then
    echo "Translation already generated for ${dataset} - ${subset} - ${lang_pair}"
    continue
fi

echo "________________________________________________"
echo "Generating ${dataset} - ${subset} - ${lang_pair}"

if [ ! -s $tmp_file ]; then
    fairseq-generate $data_path/data-bin \
    --path $path_to_ckpt \
    --results-path $tmp_result_path \
    --max-tokens $max_tokens \
    --gen-subset $subset \
    --source-lang $src_code \
    --target-lang $tgt_code \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-pairs $lang_pair_code \
    --task translation_multi_simple_epoch \
    --sacrebleu \
    --remove-bpe 'sentencepiece' \
    --lang-dict ${nllb_path}/lang_dict.txt
fi

python $ZS_ROOT/mt/get_hyp_from_fairseq_generate.py \
    $tmp_file \
    $hyp_file \
    $ref_file

tokenizer=13a
if [[ $tgt_lang == ja ]] || [[ $tgt_lang == zh-CN ]]; then
    tokenizer=${TOKENIZERS["$tgt_lang"]}
fi

sacrebleu $ref_file -i $hyp_file -m bleu -b -w 2 -tok $tokenizer > $bleu_file
sacrebleu $ref_file -i $hyp_file -m bleu -w 2 -tok $tokenizer >> $bleu_file

rm -r $tmp_result_path