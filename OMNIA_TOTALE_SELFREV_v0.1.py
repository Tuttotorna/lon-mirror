"""
OMNIA_TOTALE_SELFREV v0.1
Prototype for Ω-guided self-revision loop for LLMs.

Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

This module defines a simple pattern:

    prompt -> generate() -> Ω-score
           -> if Ω < threshold: add feedback and regenerate
           -> return best answer + Ω-trajectory

It is designed to be integrated as a lightweight plug-in around
an existing LLM API.

NOTE:
    Adjust the import below so that omnia_totale_score is imported
    from your OMNIA_TOTALE v0.5 core file.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# Import the fused Ω-score from your core module.
# Example if renamed:
# from omnia_totale_core import omnia_totale_score
from OMNIA_TOTALE_v0_5 import omnia_totale_score  # adjust to your setup


# =========================
# 1. FEATURE EXTRACTORS
# =========================

def simple_tokenize(text: str) -> List[str]:
    return [t for t in text.strip().split() if t]


def text_to_int(text: str, modulus: int = 10**9 + 7) -> int:
    """
    Hash the whole answer into an integer for the BASE lens.
    """
    return abs(hash(text)) % modulus


def build_series_for_text(text: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build simple structural series for the TIME and CAUSA lenses.
    For real integration, replace with logs: per-step loss, logprobs, etc.
    """
    tokens = simple_tokenize(text)
    if not tokens:
        arr = np.zeros(1, dtype=float)
        return arr, {"len": arr}

    lens = np.array([len(t) for t in tokens], dtype=float)
    alpha_ratio = np.array(
        [
            sum(ch.isalpha() for ch in t) / max(1, len(t))
            for t in tokens
        ],
        dtype=float,
    )
    digit_ratio = np.array(
        [
            sum(ch.isdigit() for ch in t) / max(1, len(t))
            for t in tokens
        ],
        dtype=float,
    )

    return lens, {
        "len": lens,
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
    }


# =========================
# 2. SELF-REVISION CORE
# =========================

@dataclass
class RevisionStep:
    iteration: int
    omega: float
    components: Dict[str, float]
    text: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SelfRevisionResult:
    final_text: str
    steps: List[RevisionStep]

    def to_dict(self) -> Dict:
        return {
            "final_text": self.final_text,
            "steps": [s.to_dict() for s in self.steps],
        }


class OmniaSelfRevisor:
    """
    Ω-guided self-revision wrapper.

    generate_fn: Callable[[str], str]
        A function that, given a prompt, returns a model answer.
        (In practice: wrapper around Grok / xAI / OpenAI / etc.)
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        omega_threshold: float = 0.0,
        max_iterations: int = 3,
    ) -> None:
        self.generate_fn = generate_fn
        self.omega_threshold = omega_threshold
        self.max_iterations = max_iterations

    def _compute_omega(self, answer: str) -> RevisionStep:
        n = text_to_int(answer)
        series, series_dict = build_series_for_text(answer)

        res = omnia_totale_score(
            n=n,
            series=series,
            series_dict=series_dict,
        )
        return RevisionStep(
            iteration=0,
            omega=res.omega_score,
            components=res.components,
            text=answer,
        )

    def revise(self, prompt: str) -> SelfRevisionResult:
        """
        Run the Ω-guided self-revision loop.

        1. Generate initial answer.
        2. Compute Ω.
        3. If Ω >= threshold or max_iterations reached -> return.
        4. Else, append structured feedback to the prompt and regenerate.
        """
        steps: List[RevisionStep] = []

        current_prompt = prompt
        for it in range(1, self.max_iterations + 1):
            answer = self.generate_fn(current_prompt)
            step = self._compute_omega(answer)
            step.iteration = it
            steps.append(step)

            if step.omega >= self.omega_threshold:
                break

            # Structured feedback for the next iteration
            feedback = (
                "\n\n[OMNIA_TOTALE feedback] "
                f"The previous answer had low structural Ω-coherence (Ω={step.omega:.4f}). "
                "Please regenerate a more consistent, self-checked answer, "
                "reducing contradictions and spurious details."
            )
            current_prompt = prompt + feedback

        final_text = steps[-1].text if steps else ""
        return SelfRevisionResult(final_text=final_text, steps=steps)


# =========================
# 3. DEMO WITH A MOCK MODEL
# =========================

def _mock_generate(prompt: str) -> str:
    """
    Toy generator for demo purposes only.
    Simulates an answer that becomes more coherent when feedback is added.
    """
    if "[OMNIA_TOTALE feedback]" not in prompt:
        # "Messy" first answer
        return (
            "The answer is probably 42, or maybe 13, but I am not sure. "
            "In any case, the result changes depending on the day of the week."
        )
    else:
        # Cleaner second answer
        return (
            "The stable answer is 42 under the given assumptions. "
            "Earlier contradictions have been removed."
        )


def demo():
    """
    Minimal demo:
        python OMNIA_TOTALE_SELFREV_v0.1.py
    (after making OMNIA_TOTALE v0.5 importable)
    """
    prompt = "Explain the result of this simple thought experiment."
    revisor = OmniaSelfRevisor(
        generate_fn=_mock_generate,
        omega_threshold=0.0,   # in real use, tune based on evals
        max_iterations=3,
    )

    result = revisor.revise(prompt)

    print("=== OMNIA_TOTALE_SELFREV v0.1 demo ===")
    for step in result.steps:
        print(f"\n[Iteration {step.iteration}] Ω={step.omega:.4f}")
        print("Components:", step.components)
        print("Answer:", step.text)

    print("\nFinal answer used by the pipeline:")
    print(result.final_text)


if __name__ == "__main__":
    demo()