"""Print 5 random cases that flipped from strict=incorrect → lenient=correct.

Usage:
    PYTHONPATH=src python3 scripts/inspect_flipped.py [path_to_lenient.json] [k]

Defaults to the baseline v20260415_002550 lenient file and k=5.
"""

import json
import random
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/experiments/baseline/baseline_v20260415_002550_lenient.json"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

with open(path) as f:
    d = json.load(f)

flipped = [c for c in d["cases"]
           if c["strict_correct"] is False and c["lenient_correct"] is True]

print(f"{len(flipped)} cases flipped to correct under lenient scoring. "
      f"Showing {min(k, len(flipped))} random examples:")

random.seed(42)
random.shuffle(flipped)

for c in flipped[:k]:
    g = c["gold_reference"] or {}
    print()
    print(f"subj={c['subject_id']}  hadm={c['hadm_id']}")
    print(f"  model diagnosis       : {c['model_diagnosis']!r}")
    print(f"  primary billed gold   : {g.get('long_title')!r} (ICD {g.get('icd9_code')})")
    print(f"  admission_icd_count   : {c['admission_icd_count']}")
    print(f"  lenient judge comment : {c['lenient_comment']}")
