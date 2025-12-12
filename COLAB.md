OMNIA — Real Run (Colab, Reproducible)

This Colab notebook provides a reproducible, fixed execution path for OMNIA.
Its purpose is not exploration, but verification.

If you follow these steps, you should obtain the same metrics reported in the repository (within numerical tolerance).


---

1. Open in Colab

Open the official notebook:

https://colab.research.google.com/github/Tuttotorna/lon-mirror/blob/main/colab/OMNIA_REAL_RUN.ipynb


---

2. What this run does (exactly)

The notebook performs the following steps:

1. Clones this repository


2. Installs fixed dependencies from requirements.txt


3. Sets global random seeds (reproducibility lock)


4. Executes the real benchmark script


5. Produces machine-readable reports



Specifically, it runs:

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

Do not change the seed unless you explicitly want a different experimental run.

Changing the seed = different experiment.


---

4. Fixed Environment

Python version

Tested on:

Python 3.10 (Google Colab, Dec 2025)

Dependencies

All dependencies are fixed via:

requirements.txt

These versions were captured using:

pip freeze

This ensures that:

numerical behavior

random streams

library internals


remain consistent across runs.


---

5. Outputs (Artifacts)

After execution, the notebook produces:

Reports

reports/
 ├─ real_gsm8k_report.json
 └─ real_gsm8k_worst_cases.jsonl

real_gsm8k_report.json
→ aggregated metrics (detection rate, precision, FPR, etc.)

real_gsm8k_worst_cases.jsonl
→ hardest failure cases for analysis


These files are part of the experiment and can be committed for audit.


---

6. Interpretation

This Colab run is intended to answer one question only:

> “Given the same input, the same code, and the same environment —
do we get the same structural results?”



If the answer is yes, OMNIA is:

reproducible

auditable

scientifically inspectable


If no, something is broken and must be fixed before further claims.


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

This Colab represents a frozen reference run.

Future changes must:

bump version numbers

change seeds explicitly

document differences in outputs


Untracked changes = invalid comparisons.

## 60-Second Run (No Setup)

Click → Run → Read the decision.

Open this notebook:
https://colab.research.google.com/github/Tuttotorna/lon-mirror/blob/main/colab/OMNIA_REAL_RUN.ipynb

What you will see:
- Ω_total
- per-lens scores
- ICE decision: PASS / ESCALATE / BLOCK

This is a reproducible, model-agnostic demo.


---

Author:
Massimiliano Brighindi — MB-X.01
OMNIA / Omniabase± / ICE / LCR
