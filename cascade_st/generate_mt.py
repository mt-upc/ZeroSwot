import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from constants import NLLB_LANG_CODES

SRC_LANG = "en"

model_id = sys.argv[1]
tgt_lang = sys.argv[2]
path_to_input = Path(sys.argv[3])
path_to_output = Path(sys.argv[4])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

try:
    src_code = NLLB_LANG_CODES[SRC_LANG]
    tgt_code = NLLB_LANG_CODES[tgt_lang]
except KeyError:
    src_code = SRC_LANG
    tgt_code = tgt_lang

model = AutoModelForSeq2SeqLM.from_pretrained(f"facebook/{model_id}")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_id}", src_lang=src_code)

if "3.3B" in model_id:
    bs = 4
elif "1.3B" in model_id:
    bs = 16 if "3090" in device_name else 4
else:
    bs = 32 if "3090" in device_name else 8
        
model.eval()
model = model.to(device)

with open(path_to_input, "r") as f:
    src_texts = f.readlines()

src_lens = [len(s.split()) for s in src_texts]

order = sorted(range(len(src_lens)), key=lambda k: src_lens[k])
src_texts = [src_texts[i] for i in order]

hypotheses = []
pbar = tqdm(total=len(src_texts) // bs + 1)
while src_texts:
    src_batch = []
    for _ in range(bs):
        if src_texts:
            src_batch.append(src_texts.pop())

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
    
# put back in original order
hypotheses = hypotheses[::-1]

inverse_order = [0] * len(order)
for i, pos in enumerate(order):
    inverse_order[pos] = i

hypotheses = [hypotheses[i] for i in inverse_order]

with open(path_to_output, "w") as f:
    f.write("\n".join(hypotheses))