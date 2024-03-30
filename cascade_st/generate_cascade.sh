#!/bin/bash

dataset_name=$1
lang_pair=$2
split=$3
asr_model=$4
mt_model=$5

src_lang=${lang_pair%-*}
tgt_lang=${lang_pair#*-}

src_code=${NLLB_LANG_CODES["$src_lang"]}
tgt_code=${NLLB_LANG_CODES["$tgt_lang"]}

asr_model_name=$(basename $(dirname $(dirname $asr_model)))

mt_model_name=$(basename $mt_model .pt)
if [[ $mt_model_name == nllb-200-distilled-600M ]] || [[ $mt_model_name == nllb-200-distilled-1.3B ]]; then
    is_mt_ft=false
else
    mt_model_name=$(basename $(dirname $(dirname $mt_model)))
    is_mt_ft=true
fi

aed=false
if [[ $asr_model_name == *aed* ]]; then
    aed=true
fi

asr_out_dir=${SAVE_DIR}/cascade_results/${dataset_name}/${lang_pair}/${split}/${asr_model_name}
mt_out_dir=${asr_out_dir}/${mt_model_name}
mkdir -p $mt_out_dir

tsv_path=${DATA_ROOT}/asr/${dataset_name}/${split}_${tgt_lang}_asr.tsv

asr_out=${asr_out_dir}/hyp.txt
if [ ! -s $asr_out ]; then
    if $aed; then
        bash $ZS_ROOT/cascade_st/generate_asr_aed.sh \
            $asr_model \
            $tsv_path \
            $asr_out_dir
    else
        python $ZS_ROOT/cascade_st/generate_asr_ctc.py \
            $asr_model \
            $tsv_path \
            $asr_out_dir
    fi
fi

mt_out=${mt_out_dir}/hyp.txt
mt_ref=${mt_out_dir}/ref.txt
bleu_file=${mt_out_dir}/bleu.txt

if [[ $split == dev ]]; then
    split_=valid
else
    split_=test
fi

cp $DATA_ROOT/mt/${dataset_name}/${lang_pair}/${split_}.${tgt_code} $mt_ref

if [ ! -s $mt_out ]; then
    if $is_mt_ft; then
        bash $ZS_ROOT/cascade_st/generate_mt_ft.sh \
            $mt_model \
            $tgt_code \
            $asr_out \
            $mt_out
    else
        python $ZS_ROOT/cascade_st/generate_mt.py \
            $mt_model_name \
            $tgt_lang \
            $asr_out \
            $mt_out
    fi

fi

tokenizer=13a
if [[ $tgt_lang == ja ]] || [[ $tgt_lang == zh-CN ]]; then
    tokenizer=${TOKENIZERS["$tgt_lang"]}
fi

sacrebleu $mt_ref -i $mt_out -m bleu -b -w 2 -tok $tokenizer > $bleu_file
sacrebleu $mt_ref -i $mt_out -m bleu -w 2 -tok $tokenizer >> $bleu_file