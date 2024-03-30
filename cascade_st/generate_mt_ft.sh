#!/bin/bash

ckpt_path=$1
tgt_code=$2
path_to_input=$3
path_to_output=$4

SRC_CODE=eng_Latn
SPM=${SPM_ROOT}/build/src/spm_encode
NLLB_ROOT=${MODELS_ROOT}/nllb
SPM_MODEL=${NLLB_ROOT}/flores200_sacrebleu_tokenizer_spm.model
DICT=${NLLB_ROOT}/dictionary.txt
LANG_DICT=${NLLB_ROOT}/lang_dict.txt

results_path=$(dirname $path_to_output)

${SPM} --model=${SPM_MODEL} < $path_to_input > ${results_path}/asr.spm.${SRC_CODE}

fairseq-preprocess \
--source-lang $SRC_CODE \
--target-lang $tgt_code \
--testpref ${results_path}/asr.spm \
--destdir ${results_path}/data-bin \
--thresholdsrc 0 \
--srcdict $DICT \
--workers $(eval nproc) \
--only-source

cp ${results_path}/data-bin/dict.${SRC_CODE}.txt ${results_path}/data-bin/dict.${tgt_code}.txt

rm ${results_path}/asr.spm.${SRC_CODE}

device_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
if [[ $ckpt_path == *"1.3"* ]]; then
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

fairseq-generate ${results_path}/data-bin \
--path $ckpt_path \
--results-path $results_path \
--max-tokens $max_tokens \
--gen-subset test \
--source-lang $SRC_CODE \
--target-lang $tgt_code \
--encoder-langtok "src" \
--decoder-langtok \
--lang-pairs ${SRC_CODE}-${tgt_code} \
--task translation_multi_simple_epoch \
--sacrebleu \
--remove-bpe 'sentencepiece' \
--lang-dict ${LANG_DICT}

generation_file=${results_path}/generate-test.txt

python $ZS_ROOT/mt/get_hyp_from_fairseq_generate.py \
    $generation_file \
    $path_to_output

rm $generation_file