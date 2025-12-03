# Omniabase Prime Classification Micro-Experiment

**File:** `omniabase_experiment.py`  
**Author:** Massimiliano Brighindi — brighissimo@gmail.com  
**Project:** MB-X.01 / Omniabase± — L.O.N. (Logical Origin Node)

This is a minimal, fully reproducible experiment designed to test a simple claim:

> A multi-base numeric representation (Omniabase-style) provides more useful
> structural information to a machine learning model than a naive single-number
> representation.

We use a toy but non-trivial task: **prime vs non-prime classification** over the
integers \\(n \in [2, 5000]\\).

---

## 1. Idea

We compare two feature spaces:

1. **Baseline features**
   - raw integer `n`
   - `log10(n)`

2. **Omniabase features**
   For each integer `n` and for multiple bases (e.g. 2, 3, 4, 5, 7, 10), we compute:

   - length of the representation in that base  
   - digit sum  
   - first digit  
   - last digit  
   - count of digit `0`  
   - count of digit `b-1` (max digit in that base)

All base-specific vectors are concatenated into a single feature vector.
The task is to predict whether `n` is prime or not.

If models trained on the **Omniabase feature space** consistently outperform models
trained on the **baseline feature space**, this is empirical evidence that
multi-base structure carries additional useful information for pattern recognition.

---

## 2. Files

- `omniabase_experiment.py` — main experiment script
- `requirements.txt` — Python dependencies (minimal)

---

## 3. How to run

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the experiment
python omniabase_experiment.py