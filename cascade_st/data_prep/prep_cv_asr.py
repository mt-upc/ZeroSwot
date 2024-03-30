from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
    gen_vocab,
    gen_config_yaml
)
import yaml
import os
from tempfile import NamedTemporaryFile
from constants import CV_NAME


COLUMNS = ["id", "audio", "n_frames", "src_text", "src_lang"]
DATA_ROOT = Path(os.environ["DATA_ROOT"])

VOCAB_TYPE = "unigram"
VOCAB_SIZE = 10000

siamese_root = DATA_ROOT / "siamese" / CV_NAME
asr_root = DATA_ROOT / "asr" / CV_NAME
asr_root.mkdir(parents=True, exist_ok=True)

train_file = siamese_root / "train_siamese.tsv"
dev_test_files = [
    siamese_root / "dev_siamese.tsv",
    siamese_root / "test_siamese.tsv",
]

df = load_df_from_tsv(train_file)

df = df[COLUMNS]
df = df.rename(columns={"src_text": "tgt_text", "src_lang": "tgt_lang"})
save_df_to_tsv(df, asr_root / "train_asr.tsv")
text = df["tgt_text"].tolist()

for i, file_name in enumerate(dev_test_files):
    df = load_df_from_tsv(file_name)
    df = df.rename(columns={"src_text": "tgt_text", "src_lang": "tgt_lang"})
    save_df_to_tsv(df, asr_root / f"{file_name.name.replace('siamese', 'asr')}")
    

spm_filename_prefix = f"spm_cv_{VOCAB_TYPE}_{VOCAB_SIZE}"
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
(asr_root / "config_asr.yaml").rename(asr_root / f"config_asr_cv_{VOCAB_SIZE}.yaml")