## Machine Translation

We finetune NLLB models on EN-X multingual MT on MUSTC and CoVoST2. The finetuned models are used to train a speech encoder on their embedding space. They are also used in the cascade_st exaperiments.

### Data

To prepare the MT data, first clone the sentencepiece repository and export its path to `SPM_ROOT`:

```bash
export SPM_ROOT=/path/to/sentencepiece
git clone https://github.com/google/sentencepiece.git $SPM_ROOT
```

Then run the preparation script for MUSTC_v1.0 or CoVoST2:

```bash
bash ${ZS_ROOT}/mt/prep_mt_data.sh $dataset_name
```

### Finetuning

To finetune a NLLB model on MUSTC or CoVoST2, run the corresponding scripts. For example for nllb600M and MUSTC:

```bash
bash ${ZS_ROOT}/mt/finetune_nllb600M_mult_mustc.sh
```

### Evaluation

After finetuning, do checkpoint averaging, and evaluate as:

```bash
bash ${ZS_ROOT}/mt/generate_finetuned_nllb.sh $mt_exp_path $dataset_name $split_name $tgt_lang $ckpt_name
```

To evaluate the original version of NLLB, run:

```bash
python ${ZS_ROOT}/mt/generate_original_nllb.py -m $model_id -d $dataset_name -split $split_name -tgt $tgt_lang
```
