import os
from pathlib import Path
import logging

import sentencepiece as spm
from tqdm import tqdm

tqdm.pandas()
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv

from text_cleaning import clean_text
from constants import NLLB_LANG_CODES, TGT_LANGS, SPLITS

DATASET = "MUSTC_v1.0"
SRC_LANG = "en"
TGT_LANGS = TGT_LANGS[DATASET]
SPLITS = SPLITS[DATASET]

MIN_CHARS = 4
R_MIN = 0.5
R_MAX = 2.0

orig_data_root = Path(os.environ["DATA_ROOT"]) / "orig" / DATASET
tgt_data_root = Path(os.environ["DATA_ROOT"]) / "st" / DATASET
models_root = Path(os.environ["MODELS_ROOT"])

# load nllb spm
spm_file = models_root / "nllb" / "flores200_sacrebleu_tokenizer_spm.model"
spm_model = spm.SentencePieceProcessor(str(spm_file))


def get_logger(file):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(file, "w", "utf-8")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def clean_example(x, logger):
    original_x = x.copy()

    x["src_text"] = clean_text(x["src_text"], is_mustc=True)
    if x["src_text"] != original_x["src_text"]:
        logger.info(f"{x.id} [MODIFIED SRC]: {original_x.src_text} -> {x.src_text}")

    x["tgt_text"] = clean_text(x["tgt_text"], is_mustc=True)
    if x["tgt_text"] != original_x["tgt_text"]:
        logger.info(f"{x.id} [MODIFIED TGT]: {original_x.tgt_text} -> {x.tgt_text}")

    if len(x["src_text"]) < MIN_CHARS:
        logger.info(f"{x.id} [SHORT SRC]: {x.src_text}")
        x.loc[:] = None
        return x

    if len(x["tgt_text"]) < MIN_CHARS:
        logger.info(f"{x.id} [SHORT TGT]: {x.tgt_text}")
        x.loc[:] = None
        return x

    len_src = len(spm_model.encode(x["src_text"], out_type=str))
    len_tgt = len(spm_model.encode(x["tgt_text"], out_type=str))
    r = round(len_src / len_tgt, 3)

    if r < R_MIN or r > R_MAX:
        logger.info(
            f"{x.id} [BAD RATIO]: {len_src}/{len_tgt} = {r} -> {x.src_text} /// {x.tgt_text}"
        )
        x.loc[:] = None
        return x

    return x


for tgt_lang in TGT_LANGS:
    lang_pair = f"{SRC_LANG}-{tgt_lang}"

    orig_lang_pair_root = orig_data_root / lang_pair
    tgt_lang_pair_root = tgt_data_root / lang_pair
    tgt_lang_pair_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {lang_pair}...")

    for split in SPLITS:
        # load and combine data
        df_asr = load_df_from_tsv(orig_lang_pair_root / f"{split}_asr.tsv")
        df_st = load_df_from_tsv(orig_lang_pair_root / f"{split}_st.tsv")
        df_st["src_text"] = df_asr.tgt_text.tolist()
        df_st["src_lang"] = NLLB_LANG_CODES[SRC_LANG]
        df_st["tgt_lang"] = NLLB_LANG_CODES[tgt_lang]
        
        save_df_to_tsv(df_st, tgt_lang_pair_root / f"{split}_st.tsv")

        # create a filtered version of the training set
        if split == "train":
            logger = get_logger(tgt_lang_pair_root / f"filter_{split}.log")
            df_st = df_st.progress_apply(clean_example, args=(logger,), axis=1).dropna()
            df_st["n_frames"] = df_st.n_frames.astype(int)
            save_df_to_tsv(df_st, tgt_lang_pair_root / f"{split}_fltr_st.tsv")
