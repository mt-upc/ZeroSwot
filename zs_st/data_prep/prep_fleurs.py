from datasets import load_dataset
from pathlib import Path
import os
import pandas as pd
from examples.speech_to_text.data_utils import save_df_to_tsv
from tqdm import tqdm

DATA_ROOT = Path(os.environ["DATA_ROOT"])
FLORES_ROOT = DATA_ROOT / "mt" / "flores200_dataset"
FLEURS_ROOT = DATA_ROOT / "st" / "fleurs"
SRC_CODE = "eng_Latn"

FLEURS_ROOT.mkdir(parents=True, exist_ok=True)

fleurs_en = load_dataset("google/xtreme_s", "fleurs.en_us")

fleurs_name = "test"
flores_name = "devtest"
    
fleurs_split = fleurs_en[fleurs_name]
fleurs_transcription = [example["raw_transcription"] for example in fleurs_split]

genders_features = fleurs_split.features["gender"].names

with open(FLORES_ROOT / flores_name / f"{SRC_CODE}.{flores_name}", "r") as f:
    flores_transcription = f.read().splitlines()
    
data = []
id_freq = {}
for i, text in enumerate(fleurs_transcription):
    if text in flores_transcription:
        id_ = fleurs_split[i]["id"]
        if id_ in id_freq:
            id_freq[id_] += 1
        else:
            id_freq[id_] = 1
            
        gender_id = fleurs_split[i]["gender"]
        
        data.append({
            "id": f"{id_}_{id_freq[id_] - 1}",
            "audio": fleurs_split[i]["audio"]["path"],
            "n_frames": fleurs_split[i]["num_samples"],
            "tgt_text": fleurs_split[i]["raw_transcription"],
            "tgt_lang": SRC_CODE,
            "gender": genders_features[gender_id],
            "flores_idx": flores_transcription.index(text),
        })
        
df = pd.DataFrame(data)
save_df_to_tsv(df, FLEURS_ROOT / f"{fleurs_name}_asr.tsv")

df = df.rename(columns={"tgt_text": "src_text", "tgt_lang": "src_lang"})
df["tgt_text"] = ""
df["tgt_lang"] = ""
        
for file_path in tqdm((FLORES_ROOT / flores_name).iterdir()):
    lang_code = file_path.name.split(".")[0]
    if lang_code == SRC_CODE:
        continue
    
    with open(file_path, "r") as f:
        flores_translation = f.read().splitlines()
        
    # fill-in the target text and language
    for idx, row in df.iterrows():
        flores_idx = row["flores_idx"]
        df.loc[idx, "tgt_text"] = flores_translation[flores_idx]
        df.loc[idx, "tgt_lang"] = lang_code
        
    save_df_to_tsv(df, FLEURS_ROOT / f"{fleurs_name}_{SRC_CODE}-{lang_code}_st.tsv")