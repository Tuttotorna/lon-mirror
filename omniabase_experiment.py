#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MB-X.01 / Omniabase±
Massimiliano Brighindi
brighissimo@gmail.com

Micro-experiment: Does an Omniabase multi-base numeric representation
provide more useful structural information to an ML model than a 
traditional single-number representation?

This is a minimal, concrete, reproducible experiment.
Researchers can run it, compare results, and extend it.
"""

import math
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ================================================================
# 1. BASIC NUMBER UTILITIES
# ================================================================

def is_prime(n: int) -> bool:
    """Simple primality test (sufficient for n up to ~1e6 here)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(n ** 0.5)
    for k in range(3, r + 1, 2):
        if n % k == 0:
            return False
    return True


def to_base_digits(n: int, base: int) -> List[int]:
    """
    Represent n in the given base as a list of digits [d_k, ..., d_0],
    most significant digit first.
    """
    if n == 0:
        return [0]
    digits = []
    x = n
    while x > 0:
        x, r = divmod(x, base)
        digits.append(r)
    return digits[::-1]


# ================================================================
# 2. OMNIABASE FEATURE ENGINE
# ================================================================

def omniabase_features(n: int, bases: Sequence[int]) -> np.ndarray:
    """
    Compute a multi-base numeric signature for integer n.

    For each base b in `bases`, we extract:

        - length: number of digits
        - digit_sum: sum of digits
        - first_digit
        - last_digit
        - zero_count: count of digit 0
        - maxdigit_count: count of digit (b-1), the maximum digit

    All base-specific vectors are concatenated.
    """
    feats: List[float] = []
    for b in bases:
        digits = to_base_digits(n, b)
        length = len(digits)
        digit_sum = sum(digits)
        first = digits[0]
        last = digits[-1]
        zeros = digits.count(0)
        maxdig = digits.count(b - 1)
        feats.extend([length, digit_sum, first, last, zeros, maxdig])
    return np.array(feats, dtype=float)


def baseline_features(n: int) -> np.ndarray:
    """
    Simple baseline:
        - raw integer n
        - log10(n)

    This mimics a naive numeric representation with low structure.
    """
    return np.array([float(n), math.log10(n)], dtype=float)


# ================================================================
# 3. DATASET CONSTRUCTION
# ================================================================

def build_dataset(
    n_min: int = 2,
    n_max: int = 5000,
    bases: Sequence[int] = (2, 3, 4, 5, 7, 10)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    xs = list(range(n_min, n_max + 1))
    y = np.array([1 if is_prime(n) else 0 for n in xs], dtype=int)

    X_base_list = []
    X_omni_list = []

    for n in xs:
        X_base_list.append(baseline_features(n))
        X_omni_list.append(omniabase_features(n, bases))

    return (
        np.vstack(X_base_list),
        np.vstack(X_omni_list),
        y,
    )


# ================================================================
# 4. MODEL EVALUATION
# ================================================================

def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray) -> None:
    """
    Train/test split and metrics for a given model and feature matrix.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    print(f"=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print()


def main():
    bases = (2, 3, 4, 5, 7, 10)

    X_base, X_omni, y = build_dataset(
        n_min=2,
        n_max=5000,
        bases=bases
    )

    print("Dataset ready.")
    print(f"Baseline feature shape : {X_base.shape}")
    print(f"Omniabase feature shape: {X_omni.shape}\n")

    logreg = LogisticRegression(max_iter=2000)
    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    evaluate_model("LogReg + baseline", logreg, X_base, y)
    evaluate_model("LogReg + Omniabase", logreg, X_omni, y)

    evaluate_model("RandomForest + baseline", forest, X_base, y)
    evaluate_model("RandomForest + Omniabase", forest, X_omni, y)

    scores = cross_val_score(forest, X_omni, y, cv=5, scoring="f1")
    print("RandomForest + Omniabase (5-fold F1): "
          f"mean={scores.mean():.4f}, std={scores.std():.4f}")


if __name__ == "__main__":
    main()