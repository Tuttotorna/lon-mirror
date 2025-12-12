# OMNIA — Real Run (Colab, Reproducible Reference)

This Colab notebook provides a **reproducible and fixed execution path** for OMNIA.

Its purpose is **verification**, not exploration.

If you follow the steps below, you should obtain the **same structural metrics**
reported in this repository (within numerical tolerance).


---

## 1. Open in Colab

Open the official, canonical notebook:

https://colab.research.google.com/github/Tuttotorna/lon-mirror/blob/main/colab/OMNIA_REAL_RUN.ipynb


---

## 2. What This Run Does (Exactly)

The notebook performs the following steps in a fixed order:

1. Clones this repository
2. Installs dependencies from `requirements.txt`
3. Sets global random seeds (reproducibility lock)
4. Executes the real benchmark script
5. Produces machine-readable output artifacts

The executed command is:

```bash
python benchmarks/run_real_gsm8k.py


---

3. Reproducibility Lock (CRITICAL)

This run is seed-locked.

Inside the benchmark code, the following seeds are fixed:

SEED = 42

import random
import numpy as np

random.seed(SEED)
np.random.seed(SEED)

If PyTorch is used:

import torch
torch.manual_seed(SEED)

Changing the seed defines a new experiment.

Do not change the seed unless you explicitly intend to run a different experimental condition.


---

4. Fixed Environment

Python Version

Tested on:

Python 3.10 (Google Colab, December 2025)


Dependencies

All dependencies are fixed via:

requirements.txt

Dependency versions were captured using:

pip freeze

This ensures consistency in:

numerical behavior

random streams

library internals



---

5. Outputs (Artifacts)

After execution, the notebook produces the following files:

reports/
 ├─ real_gsm8k_report.json
 └─ real_gsm8k_worst_cases.jsonl

Artifact Description

real_gsm8k_report.json
Aggregated metrics (detection rate, precision, false positive rate, etc.)

real_gsm8k_worst_cases.jsonl
Hardest failure cases, intended for inspection and error analysis


These artifacts are part of the experiment and may be committed for audit and review.


---

6. Interpretation

This Colab run is intended to answer one question only:

> “Given the same input, the same code, and the same environment —
do we obtain the same structural results?”



If the answer is yes, OMNIA is:

reproducible

auditable

scientifically inspectable


If the answer is no, the system is not ready for claims and must be fixed.


---

7. Important Notes

This is not a model benchmark in the traditional ML sense.

OMNIA does not generate answers.

OMNIA measures structural instability, drift, and inconsistency.


The system is model-agnostic and can be placed:

after an LLM

after a tool chain

after any symbolic or numeric pipeline



---

8. Status

This Colab notebook represents a frozen reference run.

Any future changes must:

bump version numbers

explicitly change seeds

document output differences


Untracked changes make comparisons invalid.


---

9. 60-Second Run (No Setup)

Open the notebook and press Run:

https://colab.research.google.com/github/Tuttotorna/lon-mirror/blob/main/colab/OMNIA_REAL_RUN.ipynb

What you will see:

Ω_total

per-lens scores

ICE decision: PASS / ESCALATE / BLOCK

reproducible JSON artifacts


This is a reproducible, model-agnostic verification demo.


---

Author

Massimiliano Brighindi — MB-X.01
OMNIA / Omniabase± / ICE / LCR