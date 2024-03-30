#!/bin/bash

ckpt_path=$1
path_to_tsv=$2
output_dir=$3

CONFIG=config_asr_comb_16000.yaml

exp_path=$(dirname $(dirname $ckpt_path))
res_path=${exp_path}/results
mkdir -p $res_path

split_name=$(basename $path_to_tsv .tsv)

fairseq-generate $(dirname $path_to_tsv) \
--config-yaml $CONFIG \
--gen-subset $split_name \
--results-path $res_path \
--task speech_to_text \
--path ${ckpt_path} \
--max-tokens 2_400_000 \
--beam 5 \
--scoring wer \
--wer-tokenizer 13a \
--wer-lowercase \
--wer-remove-punct \
--max-source-positions 2_400_000 \
--seed 42 \
--model-overrides "{'w2v_path': '${MODELS_ROOT}/wav2vec2/wav2vec_vox_new.pt'}"

generation_file=${res_path}/generate-${split_name}.txt

hypothesis_file=${output_dir}/hyp.txt
reference_file=${output_dir}/ref.txt
wer_file=${output_dir}/wer.txt
bleu_file=${output_dir}/bleu.txt

python $ZS_ROOT/mt/get_hyp_from_fairseq_generate.py \
    $generation_file \
    $hypothesis_file \
    $reference_file

result_line=$(tail -n 1 $generation_file)
echo $result_line | awk '{print $NF}' > $wer_file
echo $result_line >> $wer_file

sacrebleu $reference_file -i $hypothesis_file -m bleu -b -w 2 > $bleu_file
sacrebleu $reference_file -i $hypothesis_file -m bleu -w 2 >> $bleu_file