from convert_fairseq_w2v_to_hf import convert
from transformers import Wav2Vec2Processor
import torch
import sys
from pathlib import Path
from examples.speech_to_text.data_utils import load_df_from_tsv
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform
from tqdm import tqdm
from jiwer import wer
from sacrebleu import corpus_bleu
import re

HF_MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
BS = 2_000_000

ckpt_path = Path(sys.argv[1])
tsv_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = convert(ckpt_path, is_siamese="avg" in ckpt_path.name)
model = model.to(device)
model.eval()

processor = Wav2Vec2Processor.from_pretrained(HF_MODEL_NAME)

df = load_df_from_tsv(tsv_path)

audio_paths = df["audio"].tolist()
n_frames = df["n_frames"].tolist()
tgt_text = df["tgt_text"].tolist()

# sort according to n_frames and keep track of the original order
order = sorted(range(len(n_frames)), key=lambda k: n_frames[k])
audio_paths = [audio_paths[i] for i in order]
n_frames = [n_frames[i] for i in order]
tgt_text = [tgt_text[i] for i in order]

hypotheses, references = [], []
iterations = 0
cur_size = 0
for length in n_frames:
    if cur_size > BS:
        iterations += 1
        cur_size = 0
    cur_size += length
    
pbar = tqdm(total=iterations)
while audio_paths:
    batch = []
    cur_size = 0
    while cur_size < BS and audio_paths:
        wav = get_features_or_waveform(audio_paths.pop(), need_waveform=True)
        batch.append(wav)
        references.append(tgt_text.pop().lower())
        cur_size += n_frames.pop()

    with torch.no_grad():
        batch = processor(batch, sampling_rate=16_000, return_tensors="pt", padding=True).to(device)
        logits = model(batch.input_values, attention_mask=batch.attention_mask).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = processor.batch_decode(predicted_ids)
            
    hypotheses.extend(transcriptions)
    
    pbar.update(1)
    
hypotheses = [re.sub(r'\s+', ' ', hyp.replace("<unk>", "").strip()).lower() for hyp in hypotheses]

# restore original order
hypotheses = hypotheses[::-1]
references = references[::-1]

hypotheses = sorted(zip(order, hypotheses), key=lambda x: x[0])
hypotheses = [hyp for _, hyp in hypotheses]

references = sorted(zip(order, references), key=lambda x: x[0])
references = [ref for _, ref in references]

hypothesis_file = Path(out_dir) / "hyp.txt"
reference_file = Path(out_dir) / "ref.txt"
wer_file = Path(out_dir) / "wer.txt"
bleu_file = Path(out_dir) / "bleu.txt"

with open(hypothesis_file, "w") as f:
    for hyp in hypotheses:
        f.write(f"{hyp}\n")
        
with open(reference_file, "w") as f:
    for ref in references:
        f.write(f"{ref}\n")
    
wer_score = wer(references, hypotheses)
wer_score = round(100 * wer_score, 1)

bleu_score = corpus_bleu(hypotheses, [references]).score
bleu_score = round(bleu_score, 2)

with open(wer_file, "w") as f:
    f.write(f"{wer_score}\n")
    
with open(bleu_file, "w") as f:
    f.write(f"{bleu_score}\n")