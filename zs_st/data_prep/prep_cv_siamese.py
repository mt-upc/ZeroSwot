import os
from pathlib import Path

import sentencepiece as spm
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from tqdm import tqdm

from text_cleaning import (
    handle_html_non_utf,
    normalize_punctuation,
    remove_spaces,
    replace_with_spelled_out_form,
    tokenize_asr_text,
    tokenize_asr_text_punct,
)
from constants import CV_NAME, SPLITS

tqdm.pandas()


def clean_text_cv(txt):
    txt = remove_spaces(txt)
    txt = handle_html_non_utf(txt)
    txt = normalize_punctuation(txt)
    txt = txt.strip()
    return txt


SRC_LANG_CODE = "eng_Latn"
dataset = CV_NAME
splits = SPLITS[dataset]
REMOVE_COLS = ["age", "gender", "accent"]

orig_data_root = Path(os.environ["DATA_ROOT"]) / "orig" / dataset
siamese_data_root = Path(os.environ["DATA_ROOT"]) / "siamese" / dataset
models_root = Path(os.environ["MODELS_ROOT"])

siamese_data_root.mkdir(parents=True, exist_ok=True)

# load wav2vec char dict
ltr_dict = set()
with open(models_root / "wav2vec2" / "dict.ltr.txt", "r") as f:
    for line in f.read().splitlines():
        ltr_dict.add(line.split(" ")[0])

sp = spm.SentencePieceProcessor()
sp.Load(str(models_root / "nllb" / "flores200_sacrebleu_tokenizer_spm.model"))

# prepare individual language pairs
for split in splits:
    
    df = load_df_from_tsv(orig_data_root / f"{split}_asr.tsv")
    df.drop(columns=REMOVE_COLS, inplace=True)

    if split == "train":
        df["tgt_text"] = df.progress_apply(lambda x: clean_text_cv(x["tgt_text"]), axis=1)

    df["src_text"] = df.tgt_text.tolist()
    df["tgt_lang"] = SRC_LANG_CODE
    df["src_lang"] = SRC_LANG_CODE
    
    # everything below concerns only the CTC
    if split == "train":
        df["tgt_text"] = df.progress_apply(
            lambda x: replace_with_spelled_out_form(x["tgt_text"]), axis=1
        )

    df["tgt_text_tok"] = df.progress_apply(
        lambda x: " ".join(sp.encode_as_pieces(x["tgt_text"])), axis=1
    )
    df["tgt_text_tok_punct"] = df.progress_apply(
        lambda x: tokenize_asr_text_punct(x["tgt_text_tok"]), axis=1
    )

    df["tgt_text_tok"] = df.progress_apply(
        lambda x: tokenize_asr_text(x["tgt_text_tok"], vocab=ltr_dict), axis=1
    )
    df["tgt_text"] = df.progress_apply(
        lambda x: tokenize_asr_text(x["tgt_text"], vocab=ltr_dict), axis=1
    )
    save_df_to_tsv(df, siamese_data_root / f"{split}_siamese.tsv")