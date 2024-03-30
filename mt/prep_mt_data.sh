#!/bin/bash

source ${ZS_ROOT}/constants.sh

dataset=$1 # MUSTC_v1.0 or CoVoST2

SRC_LANG=en
SRC_CODE=eng_Latn
MODEL_NAME=nllb

SPM=${SPM_ROOT}/build/src/spm_encode

SPM_MODEL=${MODELS_ROOT}/${MODEL_NAME}/flores200_sacrebleu_tokenizer_spm.model
DICT=${MODELS_ROOT}/${MODEL_NAME}/dictionary.txt

if [ $dataset == $MUSTC_NAME ]; then
    tgt_langs=${TGT_LANGS["$MUSTC_NAME"]}
    test_set_name=tst-COMMON
elif [ $dataset == $COVOST_NAME ]; then
    tgt_langs=${TGT_LANGS["$COVOST_NAME"]}
    test_set_name=test
else
    echo "Dataset ${dataset} not supported"
    exit 1
fi

mult_mt_data_bin=${DATA_ROOT}/mt/${dataset}/multilingual/data-bin
mkdir -p ${mult_mt_data_bin}

ln -s ${DICT} ${mult_mt_data_bin}/dict.txt

for tgt_lang in $tgt_langs; do
    tgt_code=${NLLB_LANG_CODES[$tgt_lang]}

    lang_pair=${SRC_LANG}-${tgt_lang}
    mt_data=${DATA_ROOT}/mt/${dataset}/${lang_pair}
    st_data=${DATA_ROOT}/st/${dataset}/${lang_pair}

    mkdir -p ${mt_data}

    echo "________________________________________________________________________________"
    echo "Preparing ${lang_pair} for ${dataset} ..."
    echo " "

    awk -F'\t' 'NR>1 {print $7}' ${st_data}/train_fltr_st.tsv > "${mt_data}/train.${SRC_CODE}"
    awk -F'\t' 'NR>1 {print $4}' ${st_data}/train_fltr_st.tsv > "${mt_data}/train.${tgt_code}"
    awk -F'\t' 'NR>1 {print $7}' ${st_data}/dev_st.tsv > "${mt_data}/valid.${SRC_CODE}"
    awk -F'\t' 'NR>1 {print $4}' ${st_data}/dev_st.tsv > "${mt_data}/valid.${tgt_code}"
    awk -F'\t' 'NR>1 {print $7}' ${st_data}/${test_set_name}_st.tsv > "${mt_data}/test.${SRC_CODE}"
    awk -F'\t' 'NR>1 {print $4}' ${st_data}/${test_set_name}_st.tsv > "${mt_data}/test.${tgt_code}"

    ${SPM} --model=${SPM_MODEL} < ${mt_data}/train.${SRC_CODE} > "${mt_data}/train.spm.${SRC_CODE}"
    ${SPM} --model=${SPM_MODEL} < ${mt_data}/train.${tgt_code} > "${mt_data}/train.spm.${tgt_code}"
    ${SPM} --model=${SPM_MODEL} < ${mt_data}/valid.${SRC_CODE} > "${mt_data}/valid.spm.${SRC_CODE}"
    ${SPM} --model=${SPM_MODEL} < ${mt_data}/valid.${tgt_code} > "${mt_data}/valid.spm.${tgt_code}"
    ${SPM} --model=${SPM_MODEL} < ${mt_data}/test.${SRC_CODE} > "${mt_data}/test.spm.${SRC_CODE}"
    ${SPM} --model=${SPM_MODEL} < ${mt_data}/test.${tgt_code} > "${mt_data}/test.spm.${tgt_code}"

    fairseq-preprocess \
    --source-lang $SRC_CODE \
    --target-lang $tgt_code \
    --trainpref ${mt_data}/train.spm \
    --validpref ${mt_data}/valid.spm \
    --testpref ${mt_data}/test.spm \
    --destdir ${mt_data}/data-bin \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers $(eval nproc)

    for file in ${mt_data}/data-bin/*; do
        base_name=$(basename $file)
        if [[ $base_name != "preprocess.log" ]] && [[ $base_name != "dict.${SRC_CODE}.txt" ]]; then
            ln -s $file ${mult_mt_data_bin}/${base_name}
        fi
    done

done

ln -s ${mt_data}/data-bin/dict.${SRC_CODE}.txt ${mult_mt_data_bin}/dict.${SRC_CODE}.txt