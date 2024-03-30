import argparse
import os
import sys
from pathlib import Path

import subprocess
import torch
from examples.speech_to_text.data_utils import load_df_from_tsv
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)

from constants import (
    DATASETS,
    MBART_LANG_CODES,
    MODELS,
    NLLB_LANG_CODES,
    TOKENIZERS,
)

parser = argparse.ArgumentParser(description="Process command line arguments.")
parser.add_argument(
    "--dataset", "-d", type=str, default="MUSTC_v1.0", choices=DATASETS,
    help="Name of the dataset",
)
parser.add_argument(
    "--src-lang", "-src", type=str, default="en",
    help="Source language"
)
parser.add_argument(
    "--tgt-lang", "-tgt", type=str, default="de",
    help="Target language"
)
parser.add_argument(
    "--split", "-split", type=str, default="dev",
    help="Data split to use",
)
parser.add_argument(
    "--model-id", "-m", type=str, default="nllb-200-distilled-600M", choices=MODELS,
    help="Model ID to use",
)
args = parser.parse_args()

lang_pair = f"{args.src_lang}-{args.tgt_lang}"
st_data_root = Path(os.environ["DATA_ROOT"]) / "st" / args.dataset / lang_pair
siamese_data_root = Path(os.environ["DATA_ROOT"]) / "siamese" / args.dataset / lang_pair
out_dir = (
    Path(os.environ["SAVE_DIR"])
    / "mt_foundation_results"
    / args.model_id
    / args.dataset
    / lang_pair
    / args.split
)
out_dir.mkdir(parents=True, exist_ok=True)
ref_file = out_dir / "ref.txt"
hyp_file = out_dir / "hyp.txt"
bleu_file = out_dir / "bleu.txt"

# if bleu exists and is not empty, skip
if bleu_file.exists() and bleu_file.stat().st_size > 0:
    print(f"Bleu file {bleu_file} exists and is not empty. Skipping...")
    sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

if "nllb" in args.model_id:
    src_code = NLLB_LANG_CODES[args.src_lang]
    tgt_code = NLLB_LANG_CODES[args.tgt_lang]
    model = AutoModelForSeq2SeqLM.from_pretrained(f"facebook/{args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        f"facebook/{args.model_id}", src_lang=src_code
    )
    if "3.3B" in args.model_id:
        bs = 4
    elif "1.3B" in args.model_id:
        bs = 16 if "3090" in device_name else 4
    else:
        bs = 32 if "3090" in device_name else 8
elif "mbart" in args.model_id:
    src_code = MBART_LANG_CODES[args.src_lang]
    tgt_code = MBART_LANG_CODES[args.tgt_lang]
    if src_code is None or tgt_code is None:
        print(f"Language pair {lang_pair} not supported for MBART")
        sys.exit(0)
    model = MBartForConditionalGeneration.from_pretrained(f"facebook/{args.model_id}")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        f"facebook/{args.model_id}", src_lang=src_code
    )
    bs = 32
else:
    raise ValueError(f"Model {args.model_id} not supported")

model.eval()
model = model.to(device)

df = load_df_from_tsv(st_data_root / f"{args.split}_st.tsv")
tgt_texts = df["tgt_text"].tolist()
src_texts = df["src_text"].tolist()

src_lens = [len(s.split()) for s in src_texts]
order = sorted(range(len(src_lens)), key=lambda k: src_lens[k])
src_texts = [src_texts[i] for i in order]
tgt_texts = [tgt_texts[i] for i in order]

hypotheses, references = [], []
pbar = tqdm(total=len(src_texts) // bs + 1)
while src_texts:
    src_batch = []
    for _ in range(bs):
        if src_texts:
            src_batch.append(src_texts.pop())
            references.append(tgt_texts.pop())

    with torch.no_grad():
        encoded_ar = tokenizer(src_batch, padding="longest", return_tensors="pt")
        encoded_ar["input_ids"] = encoded_ar["input_ids"].to(device)
        encoded_ar["attention_mask"] = encoded_ar["attention_mask"].to(device)

        generated_tokens = model.generate(
            **encoded_ar,
            num_beams=5,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
        )
        hyp = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for h in hyp:
        hypotheses.append(h)

    pbar.update(1)

tokenizer=TOKENIZERS.get(args.tgt_lang, "13a")

with open(ref_file, "w") as f:
    f.write("\n".join(references))
    
with open(hyp_file, "w") as f:
    f.write("\n".join(hypotheses))

subprocess.run(f"sacrebleu {str(ref_file)} -i {str(hyp_file)} -m bleu -b -w 2 -tok {tokenizer} > {str(bleu_file)}", shell=True)
subprocess.run(f"sacrebleu {str(ref_file)} -i {str(hyp_file)} -m bleu -w 2 -tok {tokenizer} >> {str(bleu_file)}", shell=True)