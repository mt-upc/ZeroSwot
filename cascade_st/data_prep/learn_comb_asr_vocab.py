from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    gen_vocab,
    gen_config_yaml
)
import shutil
import yaml
import os
from tempfile import NamedTemporaryFile
from constants import CV_NAME, LS_NAME, MUSTC_NAME

DATA_ROOT = Path(os.environ["DATA_ROOT"])
VOCAB_TYPE = "unigram"
VOCAB_SIZE = 16000

asr_root = DATA_ROOT / "asr"
train_files = [
    asr_root / MUSTC_NAME / "train_comb_asr.tsv",
    asr_root / CV_NAME / "train_asr.tsv",
    asr_root / LS_NAME / "train-other-500_asr.tsv",
    asr_root / LS_NAME / "train-clean-100_asr.tsv",
    asr_root / LS_NAME / "train-clean-360_asr.tsv",
]

text = []
for file in train_files:
    df = load_df_from_tsv(file)
    text.extend(df["tgt_text"].tolist())

spm_filename_prefix = f"spm_comb_{VOCAB_TYPE}_{VOCAB_SIZE}"
with NamedTemporaryFile(mode="w") as f:
    for t in text:
        f.write(t + "\n")
    gen_vocab(
        Path(f.name),
        asr_root / spm_filename_prefix,
        VOCAB_TYPE,
        VOCAB_SIZE,
    )

yaml_filename = f"config_asr_comb_{VOCAB_SIZE}.yaml"
gen_config_yaml(
    asr_root,
    spm_filename=spm_filename_prefix + ".model",
    yaml_filename=yaml_filename,
    specaugment_policy=None,
    extra={"use_audio_input": True}
)

with open(asr_root / yaml_filename, 'r') as file:
    data = yaml.safe_load(file)
    
del data["input_feat_per_channel"]
data["vocab_filename"] = str(asr_root / data["vocab_filename"])

with open(asr_root / yaml_filename, 'w') as file:
    yaml.dump(data, file)

for file in asr_root.iterdir():
    if file.is_dir():
        continue
    for dataset in [LS_NAME, CV_NAME, MUSTC_NAME]:
        shutil.copy(file, asr_root / dataset / file.name)