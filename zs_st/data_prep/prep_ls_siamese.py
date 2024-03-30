import logging
import os
from pathlib import Path

from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv

from text_cleaning import tokenize_asr_text, restore_puncts, tokenize_asr_text_punct
import sentencepiece as spm
from constants import LS_NAME, SPLITS
from tqdm import tqdm

tqdm.pandas()

logging.disable(logging.INFO)

SRC_LANG = "eng_Latn"
dataset = LS_NAME
splits = SPLITS[dataset]
BS = 64

models_root = Path(os.environ["MODELS_ROOT"])
root = Path(os.environ["DATA_ROOT"]) / "orig" / dataset
siamese_root = Path(os.environ["DATA_ROOT"]) / "siamese" / dataset
siamese_root.mkdir(parents=True, exist_ok=True)

# load wav2vec char dict
ltr_dict = set()
with open(models_root / "wav2vec2" / "dict.ltr.txt", "r") as f:
    for line in f.read().splitlines():
        ltr_dict.add(line.split(" ")[0])
        
sp = spm.SentencePieceProcessor()
sp.Load(str(models_root / "nllb" / "flores200_sacrebleu_tokenizer_spm.model"))
        
for split in splits:
    print(f"Processing {split} ...")
    
    df = load_df_from_tsv(root / f"{split}.tsv")
    
    df["src_text"] = df.tgt_text.tolist()
    df["src_lang"] = SRC_LANG
    df["tgt_lang"] = SRC_LANG
    df["tgt_text"] = df.apply(
        lambda x: tokenize_asr_text(x["tgt_text"], vocab=ltr_dict), axis=1
    )
    df["src_text"] = restore_puncts(df["src_text"].tolist(), bs=BS)
    
    df["tgt_text_tok"] = df.progress_apply(
        lambda x: " ".join(sp.encode_as_pieces(x["src_text"])), axis=1
    )
    df["tgt_text_tok_punct"] = df.progress_apply(
        lambda x: tokenize_asr_text_punct(x["tgt_text_tok"]), axis=1
    )
    df["tgt_text_tok"] = df.progress_apply(
        lambda x: tokenize_asr_text(x["tgt_text_tok"], vocab=ltr_dict), axis=1
    )

    save_df_to_tsv(df, siamese_root / f"{split}_siamese.tsv")
    