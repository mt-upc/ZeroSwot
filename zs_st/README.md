# Pushing the Limits of Zero-shot End-to-End Speech Translation

Here we provide more details on the ZeroSwot model and the training process. We also provide the scripts used to train and evaluate the model.

## Setup

### Environment variables

```bash
export FAIRSEQ_ROOT=...     # path to fairseq repo
export ZS_ROOT=...          # path to zeroswot repo
export SPM_ROOT=...         # path to sentencepiece repo
export MODELS_ROOT=...      # where pretrained models (wa2vec2.0, nllb) are stored
export SAVE_DIR=...         # where the models will be saved
export DATA_ROOT=...        # where the data is stored
```

### Install environment

```bash
conda env create -f ${ZS_ROOT}/environment.yml
conda activate zeroswot
```

```bash
git clone -b zeroswot https://github.com/mt-upc/fairseq-internal.git ${FAIRSEQ_ROOT}
pip install --editable ${FAIRSEQ_ROOT}
export PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples:${ZS_ROOT}:${PYTHONPATH}
```

```bash
git clone https://github.com/google/sentencepiece.git $SPM_ROOT
```

```bash
source ${ZS_ROOT}/constants.sh
```

## Pretrained CTC and MT models

Our models are based on pretrained CTC Encoders and MT models. We are using wav2vec 2.0 and NLLB models, but you can use any other CTC and MT models (with some modifications in the code). The models are stored at `${MODELS_ROOT}`.

### wav2vec 2.0

Download the CTC-finetuned wav2vec 2.0 models and the letter dictionary. Save them at `${MODELS_ROOT}/wav2vec2`.

```bash
mkdir -p ${MODELS_ROOT}/wav2vec2
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt -O ${MODELS_ROOT}/wav2vec2/wav2vec_small_960h.pt
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt -O ${MODELS_ROOT}/wav2vec2/wav2vec_vox_960h_pl.pt
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -O ${MODELS_ROOT}/wav2vec2/dict.ltr.txt
```

### NLLB

Download the two distilled NLLB models, 600M (Med) and 1.3B (Lrg), and the spm tokenizer. Save them at `${MODELS_ROOT}/nllb`. Also copy there the full dictionary (with appended lang_codes and masks).

```bash
mkdir -p ${MODELS_ROOT}/nllb
wget https://tinyurl.com/nllb200densedst600mcheckpoint -O ${MODELS_ROOT}/nllb/nllb-200-distilled-600M.pt
wget https://tinyurl.com/nllb200densedst1bcheckpoint -O ${MODELS_ROOT}/nllb/nllb-200-distilled-1.3B.pt
wget https://tinyurl.com/flores200sacrebleuspm -O ${MODELS_ROOT}/nllb/flores200_sacrebleu_tokenizer_spm.model
wget https://tinyurl.com/nllb200dictionary -O ${MODELS_ROOT}/nllb/dictionary.txt
cp ${ZS_ROOT}/mt/dictionary.full.txt ${MODELS_ROOT}/nllb/dictionary.full.txt
cp ${ZS_ROOT}/mt/lang_dict.txt ${MODELS_ROOT}/nllb/lang_dict.txt
```

## Data

We used MUSTC v1.0, LibriSpeech, CommonVoice and CoVoST2 and FLEURS. MUSTC-ASR, LibriSpeech and CommonVoice are used for the speech encoder training. We optionally use the MT data of MUSTC and CoVoST2 for NLLB finetuning. Speech translation evaluation is done on the test splits of MUSTC (En-X, 8 directions), CoVoST2 (En-X, 15 directions) and FLEURS (En-X, 88 directions).

Follow the instructions below to prepare the data.

```bash
mkdir -p $DATA_ROOT/{orig,siamese,st,mt,asr}
```

### MUSTC v1.0

Download the data from [here](https://ict.fbk.eu/must-c/) to `${DATA_ROOT}/orig/MUSTC_v1.0`.

Preprocess the data into the standard tsvs:

```bash
for tgt_lang in {de,es,fr,it,pt,nl,ro,ru}; do
    for task in {st,asr}; do
        python ${FAIRSEQ_ROOT}/examples/speech_to_text/prep_mustc_data.py -d ${DATA_ROOT}/orig/MUSTC_v1.0 --task $task --use-audio-input -l $tgt_lang
    done
done
```

Apply text-based filtering for the MUSTC data and save them at `${DATA_ROOT}/st/MUSTC_v1.0`:

```bash
python ${ZS_ROOT}/zs_st/data_prep/filter_mustc.py
```

Prepare the MUSTC data at `${DATA_ROOT}/siamese/MUSTC_v1.0`. The script will process and combine the ASR data from the 8 language splits into a single tsv (one train and one dev).

```bash
python ${ZS_ROOT}/zs_st/data_prep/prep_mustc_siamese.py
```

### LibriSpeech

Download and prepare the data at `${DATA_ROOT}/orig/LibriSpeech`:
    
```bash
 python $FAIRSEQ_ROOT/examples/speech_to_text/prep_librispeech_data.py --output-root $DATA_ROOT/orig --no-vocab --use-audio-input
```

Prepare the LibriSpeech data at `${DATA_ROOT}/siamese/LibriSpeech`. This might take a while (around 2-3 hours since we are restoring the casing and punctuation with a BERT model).

```bash
python ${ZS_ROOT}/zs_st/data_prep/prep_ls_siamese.py
```

### CommonVoice

Download and the v11 at `${DATA_ROOT}/orig/CommonVoice`. You can access it [here](https://commonvoice.mozilla.org/en/datasets).

Preprocess the data into the standard tsvs:

```bash
for split in {train,dev,test}; do
    python ${FAIRSEQ_ROOT}/examples/speech_to_text/prep_commonvoice_data.py -d ${DATA_ROOT}/orig/CommonVoice -s $split --use-audio-input
done
```

Prepare the data at `${DATA_ROOT}/siamese/CommonVoice`:

```bash
python ${ZS_ROOT}/zs_st/data_prep/prep_cv_siamese.py
```

### CoVoST2 En-X

We optionally use the MT data of CoVoST2 EN-X for NLLB finetuning. The ST test split is used for evaluation. Prepare the st data at `${DATA_ROOT}/st/CoVoST2`, by following the instructions of [the official repo](https://github.com/facebookresearch/covost) and the [fairseq s2t](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/covost_example.md). When you prepapre the data, make sure to change the language codes to the ones expected from NLLB (e.g. `en` -> `eng_Latn`).

### FLEURS

Download the FLORES-200 data at `${DATA_ROOT}/mt/flores200_dataset`:

```bash
wget https://tinyurl.com/flores200dataset -O ${DATA_ROOT}/mt/flores200_dataset.tar.gz
tar -xvf ${DATA_ROOT}/mt/flores200_dataset.tar.gz -C ${DATA_ROOT}/mt/flores200_dataset
rm ${DATA_ROOT}/mt/flores200_dataset.tar.gz
```

Prepare the test splits at `${DATA_ROOT}/st/FLEURS`:

```bash
bash ${ZS_ROOT}/zs_st/data_prep/prep_fleurs.sh
```

## Training

The configs of the main experiments presented in the paper are at `${ZS_ROOT}/training/exp_configs`. To train a speech encoder to be adapted to the space of a multilingual MT model (NLLB) simply run the following command.

```bash
bash ${ZS_ROOT}/zs_st/train.sh $path_to_exp_config
```

* The naming conventions are `{dataset-nane}_{w2v-version}_{nllb-version}.yaml`.
* We are training with a batch size of 32M tokens. The configs are optimized for training with a NVIDIA 3090, change the `dataset/max_tokens` and `optimization/update_freq` parameter in the config according to your machine setting.
* Multi-gpu training is by default on if more than 1 devices are detected.
* We used wandb for logging. Make sure to have it installed and logged in. Also set the `WANDB_PROJECT` environment variable. If you don't want to use it remove the argument `common/wandb_project` from the config.
* In practice, since the MT text encoder is frozen we can pre-extract the text representations offile. You can checkout the `${ZS_ROOT}/zs_st/utils/extract_text_representations.py` script for this purpose. In that case, when training with cached text representations, set argument `model/text_encoder/remove` to `True`, and add the path of the extracted representations at `task/cached_text_representations_path`.
* Due to the long training times in CommonVoice, we found it useful to train only one (seed) model with the Medium NLLB, and then when trainign with different NLLB version used the seed model for initalization. Models converged in 1/10 of the training time, and performance is the same. Check the `${ZS_ROOT}/zs_st/exp_configs/cv_w2v-Lrg_nllb1.3B_init.yaml` for a reference.
* To adapt a speech encoder on a finetuned version of NLLB, check the `${ZS_ROOT}/mt` folder for some scripts regarding the NLLB finetuning on MUSTC/CoVoST2. When you complete the finetuning, you can use the finetuned model as the `model/text_encoder/path` in the config.

## Zero-shot ST Evaluation

After training a speech encoder, we average the 10 best checkpoints, and construct the ZeroSwot model by replacing the MT embedding layer with the speech encoder. The model will be created at `${SAVE_DIR}/speech_encoders/${exp_name}/ckpts_zs`.

```bash
bash ${ZS_ROOT}/zs_st/utils/avg_best_ckpts.sh $path_to_exp
bash ${ZS_ROOT}/zs_st/utils/construct_model.sh $path_to_exp $path_to_mt_model
```

Where `path_to_mt_model` is the path to the MT checkpoint file (either original or finetuned).

To evaluate the ZeroSwot model, run the following command:

```bash
bash ${ZS_ROOT}/zs_st/eval.sh $path_to_exp $path_to_mt_model $dataset_name $dataset_split $tgt_lang
```

Where `$dataset_name` is either `MUSTC_v1.0`, `CoVoST2` or `FLEURS`, `$dataset_split` is the split (e.g. `test`), and `$tgt_lang` is the target language (e.g. `de`).

