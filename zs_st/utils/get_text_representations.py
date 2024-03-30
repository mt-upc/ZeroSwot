import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from examples.speech_to_text.data_utils import load_df_from_tsv
from fairseq.data import Dictionary
from fairseq.models.transformer import Embedding
from fairseq.models.transformer.transformer_encoder import TransformerEncoder
from tqdm import tqdm
from constants import DATASETS, SPLITS, LS_NAME, CV_NAME, MUSTC_NAME

# dataset name: LS_NAME/CV_NAME/MUSTC_NAME
dataset = sys.argv[1]
# path to the MT model checkpoint
model_path = Path(sys.argv[2])
# comma separated layer ids for which to extract the representations
# 6,7,8,9,10,11 for medium models, and 12,14,16,18,20,22 for large ones
layer_ids = [int(x) for x in sys.argv[3].split(",")]

SRC_LANG = "eng_Latn"
assert dataset in DATASETS

data_root = Path(os.environ["DATA_ROOT"]) / "siamese" / dataset
nllb_root = Path(os.environ["MODELS_ROOT"]) / "nllb"

if "avg_best" in model_path.name:
    # finetuned version
    model_name = model_path.parent.parent.name
else:
    model_name = model_path.stem

out_root = (
    Path(os.environ["SAVE_DIR"]) / "nllb_representations" / model_name / dataset
)
out_root.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(model_path, map_location="cpu")
model_args = ckpt["cfg"]["model"]

spm_file = nllb_root / "flores200_sacrebleu_tokenizer_spm.model"
dict_file = nllb_root / "dictionary.full.txt"
spm_model = spm.SentencePieceProcessor(str(spm_file))
dictionary = Dictionary.load(dict_file.as_posix())
bos_id = dictionary.index(f"<lang:{SRC_LANG}>")
eos_id = dictionary.index("</s>")

embedding = Embedding(len(dictionary), model_args.encoder_embed_dim, dictionary.pad())
model = TransformerEncoder(model_args, dictionary, embedding, return_ln=True)

encoder_ckpt = {}
for k, v in ckpt["model"].items():
    if k.startswith("encoder"):
        encoder_ckpt[k.replace("encoder.", "")] = v
ckpt = None

model.load_state_dict(encoder_ckpt, strict=False)
model = model.to(device)
model.eval()

print("Loading data...")
splits = {}

if dataset == LS_NAME or dataset == CV_NAME:
    for split_name in SPLITS[dataset]:
        split = data_root / f"{split_name}_siamese.tsv"
        splits[split] = load_df_from_tsv(split)
        
elif dataset == MUSTC_NAME:
    train_split = str(data_root / "train_comb_siamese.tsv")
    splits[train_split] = load_df_from_tsv(train_split)
    comb_dev_split = str(data_root / "dev_comb_siamese.tsv")
    splits[comb_dev_split] = load_df_from_tsv(comb_dev_split)
    for lang in ["de", "es", "fr", "it", "pt", "nl", "ru", "ro"]:
        for split in ["dev", "tst-COMMON"]:
            dev_test_split = str(data_root / f"{split}_{lang}_siamese.tsv")
            splits[dev_test_split] = load_df_from_tsv(dev_test_split)

bs = 64
if "1.3B" in model_name:
    bs = 32
elif "3.3B" in model_name:
    bs = 16
    
for split_name, df in splits.items():
    
    print("Extracting representations for {}...".format(split_name))
    
    texts = df["src_text"].tolist()
    ids = df["id"].tolist()
    indices = df.index.tolist()
    
    examples = []
    for i in range(len(texts)):
        str_tokens = spm_model.encode(texts[i].strip(), out_type=str)
        encoded = [bos_id] + [dictionary.index(t) for t in str_tokens] + [eos_id]
        examples.append({
            "id": ids[i],
            "index": indices[i],
            "encoded": encoded,
            "text": texts[i],
            "length": len(encoded),
        })

    # sort by length from longest to shortest
    lengths = [x["length"] for x in examples]
    sort_idx = np.argsort(lengths)
    examples = [examples[i] for i in sort_idx]

    start = time.time()
    pbar = tqdm(total=len(examples) // bs + 1)
    while examples:
        batch, batch_info = [], []
        for _ in range(bs):
            if examples:
                example = examples.pop()
                batch.append(example["encoded"])
                batch_info.append(example)

        batch = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in batch],
            batch_first=True,
            padding_value=dictionary.pad(),
        )
        batch = batch.to(device)

        with torch.no_grad():
            out = model(batch, return_all_hiddens=True)
            
            out_ = {
                "encoder_out": out["encoder_out"][0].transpose(0, 1).detach().cpu().half()
            }
            for layer_id in layer_ids:
                out_[f"ln_results{layer_id}"] = out["ln_results"][layer_id].transpose(0, 1).detach().cpu().half()

            out = None
            
            for i in range(len(batch_info)):
                
                id_ = batch_info[i]["id"]
                index = batch_info[i]["index"]
                length = batch_info[i]["length"]
                file = out_root / f"{id_}.pt"
                out_i = {}
                
                for k, v in out_.items():
                    out_i[k] = v[i, :length, :].clone()
                    
                torch.save(out_i, file)

        pbar.update(1)

    pbar.close()
        
print(f"Finished in {round((time.time() - start) / 60)} minutes")