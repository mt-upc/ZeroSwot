import sys
from pathlib import Path

path_to_generation_file = Path(sys.argv[1])
path_to_hypotheses_file = Path(sys.argv[2])
path_to_references_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None

hypotheses, correct_order, references = [], [], []
with open(path_to_generation_file, "r", encoding="utf8") as f:
    for line in f.read().splitlines():
        if line[:2] == "T-":
            references.append(line.split(maxsplit=1)[1])
        if line[:2] == "D-":
            correct_order.append(int(line.split(maxsplit=1)[0].split("D-")[-1]))
            splits = line.split(maxsplit=2)
            if len(splits) == 3:
                hypotheses.append(splits[2])
            else:
                hypotheses.append("")

hypotheses = [gen for _, gen in sorted(zip(correct_order, hypotheses))]
with open(path_to_hypotheses_file, "w") as f:
    f.write("\n".join(hypotheses))

if path_to_references_file is not None:
    references = [gen for _, gen in sorted(zip(correct_order, references))]
    with open(path_to_references_file, "w") as f:
        f.write("\n".join(references))