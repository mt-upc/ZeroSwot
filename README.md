# ZeroSwot

This repository contains the code for the paper "Pushing the Limits of Zero-shot End-to-End Speech Translation".

The preprint is available on [arXiv](https://arxiv.org/abs/2402.10422).

## Installation

Set the following environment variables:

```bash
export FAIRSEQ_ROOT=...     # path to fairseq repo
export ZS_ROOT=...          # path to zeroswot repo
export MODELS_ROOT=...      # where pretrained models (wa2vec2.0, nllb) are stored
export SAVE_DIR=...         # where the models will be saved
export DATA_ROOT=...        # where the data is stored
```

Clone the repository and install the dependencies:

```bash
git clone https://github.com/mt-upc/ZeroSwot.git ${ZS_ROOT}
conda env create -f ${ZS_ROOT}/environment.yml
conda activate zeroswot
source ${ZS_ROOT}/constants.sh
```

Install the fairseq fork with the zeroswot branch:

```bash
git clone -b zeroswot https://github.com/mt-upc/fairseq-internal.git ${FAIRSEQ_ROOT}
pip install --editable ${FAIRSEQ_ROOT}
export PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples:${ZS_ROOT}:${PYTHONPATH}
```

## Download pretrained CTC and MT models

Our models are based on pretrained CTC Encoders and MT models. We are using wav2vec 2.0 and NLLB models, but you can use any other CTC and MT models (with some modifications in the code). The models are stored at `${MODELS_ROOT}`.

Download the CTC-finetuned wav2vec 2.0 model and the letter dictionary:

```bash
mkdir -p ${MODELS_ROOT}/wav2vec2
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt -O ${MODELS_ROOT}/wav2vec2/wav2vec_vox_960h_pl.pt
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -O ${MODELS_ROOT}/wav2vec2/dict.ltr.txt
```

Download the two distilled NLLB models, 600M (Med) and 1.3B (Lrg), and the spm tokenizer:

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

Download and prepare the tsv files for MUSTC/CoVoST2 with raw waveforms as input at `${DATA_ROOT}/st`. You can find more details at [zs_st/README.md](zs_st/README.md). If you already have prepared the data for a different project, make sure to replace the `tgt_lang` with the corresponding code used by NLLB (e.g. `de` -> `deu_Latn`).

## Model Weights and Usage

We trained Speech Encoders using ASR data with Subword Compression and Optimal Transport in order to adapt them to the representation space of multilingual MT models. The MT models are either the original NLLB models (600M/1.3B) or finetuned versions on MUSTC/CoVoST2. The weights for the Speech Encoders and MT models are available for download below.

Download and extract the weights of a Speech Encoder at `${SAVE_DIR}/speech_encoders`. The training scripts for the Speech Encoders can be found at [zs_st/README.md](zs_st/README.md).

| ASR Data     | Adapted to               | Link                                              | MUSTC ZS-ST BLEU | CoVoST2 ZS-ST BLEU |
|--------------|--------------------------|---------------------------------------------------| ---------------| ----------------|
| MUSTC       | NLLB-600M (original)     | [Download](https://drive.google.com/file/d/1Hy_yXdYsDFTzBC5OvEitiOeOTCeFz-Tb/view?usp=drive_link)        | 29.6           | /               |
| MUSTC       | NLLB-600M (MUSTC)         | [Download](https://drive.google.com/file/d/1Hxg7w7om_WxdygDe7T_kpvENwm6nuuEl/view?usp=drive_link) | 31.9           | /               |
| MUSTC       | NLLB-1.3B (original)     | [Download](https://drive.google.com/file/d/1I0psY3urTRWgfWFZpNPvCQkEstzj_2Ax/view?usp=drive_link)        | 31.4           | /               |
| MUSTC       | NLLB-1.3B (MUSTC)         | [Download](https://drive.google.com/file/d/1IHINAt3AW5Bq0m4AkVncT8FYMx7I-4Pp/view?usp=drive_link) | 32.9           | /               |
| CommonVoice  | NLLB-600M (original)     | [Download](https://drive.google.com/file/d/1IN152EAfHN5SpeP5B4AsIlvxIjj_N33y/view?usp=drive_link)        | 26.0           | 23.1            |
| CommonVoice  | NLLB-600M (CoVoST2)        | [Download](https://drive.google.com/file/d/1IOrBo9VfyE5IwIdim727XU9Wns1kL9N2/view?usp=drive_link)| /              | 30.2            |
| CommonVoice  | NLLB-1.3B (original)     | [Download](https://drive.google.com/file/d/1Hnz4c3fy7Ky8uQsSfxDlIGrUeMz8SHJp/view?usp=drive_link)        | 27.4           | 25.5            |
| CommonVoice  | NLLB-1.3B (CoVoST2)        | [Download](https://drive.google.com/file/d/1IAKznmk066AT6DQBnOSmgBxWuZI_kpDn/view?usp=drive_link)| /              | 31.2            |

In case you want to use one of the Speeech Encoders that was adapted to a finetuned NLLB, download and extract the weights of the corresponding MT model at `${SAVE_DIR}/mt_models`. The training scripts for the MT models can be found at [mt/README.md](mt/README.md).

| Model     | MT Data        | Link                                             | MUSTC MT BLEU | CoVoST2 MT BLEU |
|-----------|----------------|--------------------------------------------------| --------------| ----------------|
| NLLB-600M | MUSTC En-X    | [Download](https://drive.google.com/file/d/1HQZYa0030DHL67-E0_FIPzHJtcgUlHGs/view?usp=drive_link)      | 35.9          | /               |
| NLLB-600M | CoVoST2 En-X   | [Download](https://drive.google.com/file/d/1HV_vz5f82tTKHfGozfAdI1Dg1q9KPpWs/view?usp=drive_link)     | /             | 35.0            |
| NLLB-1.3B | MUSTC En-X    | [Download](https://drive.google.com/file/d/1HVXB_TlxBzDqraU-Nk6zUoLRBJtvbZ_J/view?usp=drive_link)      | 37.2          | /               |
| NLLB-1.3B | CoVoST2 En-X   | [Download](https://drive.google.com/file/d/1HaoeF9yUWYT8vVGk8YRQxczcXNf3mBDV/view?usp=drive_link)     | /             | 36.1            |

Due to the size of the models, we cannot host all the experiments done in the paper. If you need the weights of another model from the paper, please open an issue and we will provide you with the download link.

Based on a Speech Encoder and an MT model, you can build a ZeroSwot model for Speech Translation as follows. The script basically replaces the MT Embedding layer with the newly trained Speech Encoder.

```bash
bash ${ZS_ROOT}/zs_st/utils/construct_model.sh $path_to_speech_encoder $path_to_mt_model
```

`$path_to_speech_encoder` should be pointing to the directory of the experiment (i.e `${exp_path}/ckpts/avg_best_10_checkpoint.pt`), while `$path_to_mt_model` should be pointing directly to the `.pt` checkpoint file of the MT model. This will create the ZeroSwot checkpoint in `${exp_path}/ckpts_zs`.

To use the model for zero-shot ST inference, refer to `zs_st/README.md` in order to prepare the test sets of MUSTC or CoVoST2, and use the following command, where `$dataset_name` is either `MUSTC_v1.0` or `CoVoST2`, `$dataset_split` is the split of the dataset (e.g. `tst-COMMON`), and `$tgt_lang` is the target language of the translation (e.g. `de`):

```bash
bash ${ZS_ROOT}/zs_st/eval.sh $path_to_speech_encoder $path_to_mt_model $dataset_name $dataset_split $tgt_lang
```

## Training and More

To train a new Speech Encoder using our method, refer to [zs_st/README.md](zs_st/README.md) for more details. 

For finetuning the NLLB models on MUSTC or CoVoST2, refer to [mt/README.md](mt/README.md) for more details.

For our experiments regarding Cascade ST, refer to [cascade_st/README.md](cascade_st/README.md) for more details.

We also provide some scripts for supervised ST finetuning of our ZeroSwot models, refer to [supervised_st/README.md](supervised_st/README.md) for more details.

## Citation

If you use this code or the models in your research, please cite this work as:

```
@misc{tsiamas2024pushing,
      title={{Pushing the Limits of Zero-shot End-to-End Speech Translation}}, 
      author={Ioannis Tsiamas and Gerard I. Gállego and José A. R. Fonollosa and Marta R. Costa-jussà},
      year={2024},
      eprint={2402.10422},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
