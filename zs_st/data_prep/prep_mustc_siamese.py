import os
from pathlib import Path
from tqdm import tqdm
import string
import logging

import pandas as pd
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from tqdm import tqdm
import numpy as np
from text_cleaning import replace_with_spelled_out_form, tokenize_asr_text, tokenize_asr_text_punct, clean_text
import sentencepiece as spm

tqdm.pandas()


def get_logger(file):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(file, "w", "utf-8")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def remove_unk(x):
    # replace ^ * ♫ ♪ / _ # with space
    for char in "^*♫♪/_#-":
        x = x.replace(char, " ")
    while "  " in x:
        x = x.replace("  ", " ")
    return x


DATASET = "MUSTC_v1.0"
SRC_LANG = "en"
SRC_LANG_CODE = "eng_Latn"
TGT_LANGS = ["de", "es", "fr", "pt", "it", "nl", "ru", "ro"]
SPLITS = ["train_fltr", "dev", "tst-COMMON"]

st_data_root = Path(os.environ["DATA_ROOT"]) / "st" / DATASET
siamese_data_root = Path(os.environ["DATA_ROOT"]) / "siamese" / DATASET
models_root = Path(os.environ["MODELS_ROOT"])

siamese_data_root.mkdir(parents=True, exist_ok=True)
logger = get_logger(siamese_data_root / "prep_mustc_siamese.log")

# load wav2vec char dict
ltr_dict = set()
with open(models_root / "wav2vec2" / "dict.ltr.txt", "r") as f:
    for line in f.read().splitlines():
        ltr_dict.add(line.split(" ")[0])
        
sp = spm.SentencePieceProcessor()
sp.Load(str(models_root / "nllb" / "flores200_sacrebleu_tokenizer_spm.model"))

# prepare individual language pairs
all_train_df, all_dev_df = {}, {}
for tgt_lang in TGT_LANGS:
    lang_pair = f"{SRC_LANG}-{tgt_lang}"
    siamese_lang_pair_root = siamese_data_root / lang_pair
    siamese_lang_pair_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {lang_pair} ...")
    
    for split in SPLITS:
        
        df = load_df_from_tsv(st_data_root / lang_pair / f"{split}_st.tsv")
        
        df["src_text"] = df.progress_apply(lambda x: clean_text(x["src_text"], is_mustc=True), axis=1)
        df["tgt_text"] = df.src_text.tolist()
        df["tgt_lang"] = df.src_lang.tolist()
        df["tgt_text"] = df.progress_apply(
            lambda x: replace_with_spelled_out_form(x["tgt_text"]), axis=1
        )
        
        # standard targets for CTC labels
        df["tgt_text"] = df.progress_apply(
            lambda x: remove_unk(x["tgt_text"]), axis=1
        )
        # subword tokenization for CTC labels
        df["tgt_text_tok"] = df.progress_apply(
            lambda x: " ".join(sp.encode_as_pieces(x["tgt_text"])), axis=1
        )
        # subword tokenization for CTC labels (+ unk token inplace of unknown characters)
        df["tgt_text_tok_punct"] = df.progress_apply(
            lambda x: tokenize_asr_text_punct(x["tgt_text_tok"]), axis=1
        )
        df["tgt_text_tok"] = df.progress_apply(
            lambda x: tokenize_asr_text(x["tgt_text_tok"], vocab=ltr_dict), axis=1
        )
        df["tgt_text"] = df.progress_apply(
            lambda x: tokenize_asr_text(x["tgt_text"], vocab=ltr_dict), axis=1
        )
        
        save_df_to_tsv(df, siamese_lang_pair_root / f"{split}_siamese.tsv")
        logger.info(f"Saved {split} for {lang_pair}")
        
        if split.startswith("train"):
            logger.info(f"{tgt_lang} - {split} has {len(df)} examples and {round(df.n_frames.sum()/16000/3600)} hours")
        if split == "train_fltr":
            all_train_df[tgt_lang] = df.copy()
        elif split == "dev":
            # append the tgt_lang to each id
            df["id"] = df["id"] + f"_{tgt_lang}"
            if split == "dev":
                all_dev_df[tgt_lang] = df.copy()
            save_df_to_tsv(df, siamese_data_root / f"{split}_{tgt_lang}_siamese.tsv")


### Prepare combined train set

DF = pd.concat(all_train_df.values())
del all_train_df
logger.info(f"Concatenated train DF has {len(DF)} examples and {round(DF.n_frames.sum()/16000/3600)} hours")

# create a seconds column for lower precision
DF["seconds"] = DF["n_frames"] / 16000
DF["seconds"] = DF["seconds"].round(1)
DF.loc[DF.seconds == 0, "seconds"] = 0.1

# create a source column to keep track of which audio file the example came from
DF["source"] = DF["audio"].apply(lambda x: x.split("en-")[-1].split("/")[0])
DF["source"] = DF.groupby(["seconds", "speaker", "src_text"])["source"].transform(
    lambda x: ",".join(x)
)

# drop exact duplicates
DF.sort_values(["speaker", "src_text", "seconds"], inplace=True)
DF.drop_duplicates(subset=["seconds", "speaker", "src_text"], inplace=True)
logger.info(f"DF has {len(DF)} examples and {round(DF.n_frames.sum()/16000/3600)} hours after dropping duplicates")

# create a speaking ratio column and get the average of each speaker
DF["spk_ratio"] = (
    DF["tgt_text"].apply(lambda x: len(x.split("|"))) / DF["seconds"]
)
DF["spk_ratio"] = DF["spk_ratio"].round(2)
DF["avg_spk_ratio"] = DF.groupby(["speaker"])["spk_ratio"].transform("mean")

def filter_by_speaking_ratio(group):
    group["keep"] = False
    group["diff"] = np.abs(group["spk_ratio"] - group["avg_spk_ratio"])
    group.loc[group["diff"] == group["diff"].min(), "keep"] = True
    return group


# filter by speaking ratio
# keep only the example that has the closest speaking spk_ratio
# to the average spk_ratio for that speaker
new_groups = []
spk_groups = DF.groupby(["speaker"])
for spk, group in tqdm(spk_groups):
    avg_spk_ratio = group["spk_ratio"].mean()
    group = group.groupby(["src_text"], group_keys=True).apply(
        filter_by_speaking_ratio
    )
    new_groups.append(group[group["keep"] == True])

# combine
DF = pd.concat(new_groups)
DF = DF.drop(columns=["diff", "keep"])
DF = DF.reset_index(drop=True)
logger.info(f"DF has {len(DF)} examples and {round(DF.n_frames.sum()/16000/3600)} hours after filtering duplicates by speaking ratio")

DF = DF[DF.spk_ratio < 10]
logger.info(f"DF has {len(DF)} examples and {round(DF.n_frames.sum()/16000/3600)} hours after filtering by speaking ratio < 10")

# Modify the id column to be unique by appending ascending letters
counts = DF.groupby('id').cumcount()
mapping = dict(enumerate(string.ascii_lowercase))
counts = counts.map(mapping)
DF.loc[DF.duplicated('id', keep=False), 'id'] += '_' + counts.astype(str)

save_df_to_tsv(DF, siamese_data_root / "train_comb_siamese.tsv")

### Prepare combined dev and test sets

DF = pd.concat(all_dev_df.values(), ignore_index=True)
del all_dev_df
logger.info(f"Concatenated dev DF has {len(DF)} examples and {round(DF.n_frames.sum()/16000/3600)}")

DF = DF.sort_values(by=["id"])
DF = DF.drop_duplicates(subset=["n_frames", "speaker", "src_text"], keep="first")
logger.info(f"DF has {len(DF)} examples and {round(DF.n_frames.sum()/16000/3600)} hours after dropping duplicates")

save_df_to_tsv(DF, siamese_data_root / "dev_comb_siamese.tsv")