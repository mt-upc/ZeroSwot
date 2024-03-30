## Cascade Speech Translation

We compared ZeroSwot to CTC-based and attention-based cascaded ST models. Here you can find the corresponding scripts.

### Data

Prepare the ASR tsv files for LibriSpeech, MUSTC and CommonVoice (they depend on the data used for the speech encoder training):

```bash
python ${ZS_ROOT}/cascade_st/data_prep/prep_ls_asr.py
python ${ZS_ROOT}/cascade_st/data_prep/prep_mustc_asr.py
python ${ZS_ROOT}/cascade_st/data_prep/prep_cv_asr.py

```

### CTC-based ASR training

To train a CTC-based ASR model on MUSTC or CommonVoice:

```bash
bash ${ZS_ROOT}/cascade_st/train_asr_ctc.sh $config_path
```

Where `$config_path` is either `{ZS_ROOT}/cascade_st/asr_ctc_mustc.yaml` or `{ZS_ROOT}/cascade_st/asr_ctc_cv.yaml`.

### Attention-based ASR training

We pair wav2vec2.0 with a Transformer decoder to train an attention-based ASR model. We first train the model on LibriSpeech and then finetune it in either MUSTC or CommonVoice.

Download the pretrained (not CTC) wav2vec 2.0 model:

```bash
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt -O $MODELS_ROOT/wav2vec2/wav2vec_vox_new.pt
```

Learn a combined SPM model on LibriSpeech, MUSTC and CommonVoice.

```bash
python ${ZS_ROOT}/cascade_st/data_prep/learn_comb_asr_vocab.py
```

Pretrain the attention-based ASR model on LibriSpeech:

```bash
bash ${ZS_ROOT}/cascade_st/pretrain_asr_ls.sh
```

Finetune the model on MUSTC or CommonVoice:

```bash
bash ${ZS_ROOT}/cascade_st/finetune_asr.sh $dataset_name
```

### MT training

Refer to the dedicated scripts at [mt/README.md](../mt/README.md) to finetune NLLB models on En-X multingual MT on MUSTC or CoVoST2.

### Evaluation

To generate with a cascade (either CTC or AED ASR), run:

```bash
bash ${ZS_ROOT}/cascade_st/generate_cascade.sh $dataset_name $lang_pair $split $path_to_asr_checkpoint $path_to_mt_checkpoint
```

Where `$dataset_name` is either `MUSTC_v1.0` or `CoVoST2`, `$lang_pair` is the language pair (e.g. `en-de`), `$split` is the split (e.g. `test`), `$path_to_asr_checkpoint` is the path to the ASR checkpoint and `$path_to_mt_checkpoint` is the path to the MT checkpoint.
