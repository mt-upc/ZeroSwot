from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
    gen_vocab,
    gen_config_yaml
)
import yaml
import os
import sys
from tempfile import NamedTemporaryFile
from constants import LS_NAME


COLUMNS = ["id", "audio", "n_frames", "src_text", "src_lang"]
DATA_ROOT = Path(os.environ["DATA_ROOT"])
VOCAB_TYPE = "unigram"
VOCAB_SIZE = 10000

siamese_root = DATA_ROOT / "siamese" / LS_NAME
asr_root = DATA_ROOT / "asr" / LS_NAME
asr_root.mkdir(parents=True, exist_ok=True)

train_files = [
    siamese_root / "train-other-500_siamese.tsv",
    siamese_root / "train-clean-100_siamese.tsv",
    siamese_root / "train-clean-360_siamese.tsv",
]
dev_test_files = [
    siamese_root / "dev-clean_siamese.tsv",
    siamese_root / "dev-other_siamese.tsv",
    siamese_root / "test-clean_siamese.tsv",
    siamese_root / "test-other_siamese.tsv",
]

df_train = [load_df_from_tsv(file) for file in train_files]
df_dev = [load_df_from_tsv(file) for file in dev_test_files]

for i in range(len(df_train)):
    df_train[i] = df_train[i][COLUMNS]
    df_train[i] = df_train[i].rename(columns={"src_text": "tgt_text", "src_lang": "tgt_lang"})
    save_df_to_tsv(df_train[i], asr_root / train_files[i].name.replace("siamese", "asr"))

for i in range(len(df_dev)):
    df_dev[i] = df_dev[i][COLUMNS]
    df_dev[i] = df_dev[i].rename(columns={"src_text": "tgt_text", "src_lang": "tgt_lang"})
    save_df_to_tsv(df_dev[i], asr_root / dev_test_files[i].name.replace("siamese", "asr"))

text = []
for df in df_train:
    text.extend(df["tgt_text"].tolist())

spm_filename_prefix = f"spm_ls_{VOCAB_TYPE}_{VOCAB_SIZE}"
with NamedTemporaryFile(mode="w") as f:
    for t in text:
        f.write(t + "\n")
    gen_vocab(
        Path(f.name),
        asr_root / spm_filename_prefix,
        VOCAB_TYPE,
        VOCAB_SIZE,
    )
    
gen_config_yaml(
    asr_root,
    spm_filename=spm_filename_prefix + ".model",
    yaml_filename="config_asr.yaml",
    specaugment_policy=None,
    extra={"use_audio_input": True}
)

with open(asr_root / "config_asr.yaml", 'r') as file:
    data = yaml.safe_load(file)
    
del data["input_feat_per_channel"]
data["vocab_filename"] = str(asr_root / data["vocab_filename"])

with open(asr_root / "config_asr.yaml", 'w') as file:
    yaml.dump(data, file)
(asr_root / "config_asr.yaml").rename(asr_root / f"config_asr_ls_{VOCAB_SIZE}.yaml")