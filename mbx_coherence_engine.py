from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re
import json

@dataclass
class Proposition:
    raw: str
    normalized: str
    polarity: int  # +1 = affirmative, -1 = negated

@dataclass
class CoherenceReport:
    text: str
    propositions: List[Proposition]
    contradictions: List[Tuple[Proposition, Proposition]]
    redundancy: List[Tuple[Proposition, Proposition]]
    coherence: float
    noise: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "propositions": [
                {
                    "raw": p.raw,
                    "normalized": p.normalized,
                    "polarity": p.polarity,
                } for p in self.propositions
            ],
            "contradictions": [
                (
                    {"raw": a.raw, "normalized": a.normalized, "polarity": a.polarity},
                    {"raw": b.raw, "normalized": b.normalized, "polarity": b.polarity},
                ) for a, b in self.contradictions
            ],
            "redundancy": [
                (
                    {"raw": a.raw, "normalized": a.normalized, "polarity": a.polarity},
                    {"raw": b.raw, "normalized": b.normalized, "polarity": b.polarity},
                ) for a, b in self.redundancy
            ],
            "coherence": self.coherence,
            "noise": self.noise,
        }

class MBXCoherenceEngine:
    """
    Minimal MBX-style coherence engine.
    Non narrativo, rule-based, estendibile.
    """

    NEGATION_PATTERNS = [
        r"\bnon\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bnessuno\b",
    ]

    SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")

    def analyze(self, text: str) -> CoherenceReport:
        sentences = self._split_sentences(text)
        propositions = [p for s in sentences for p in self._extract_propositions(s)]
        contradictions = self._find_contradictions(propositions)
        redundancy = self._find_redundancy(propositions)

        # Coherence score: 1 - (contradictions + 0.5*noise) / max(1, len(props))
        n_props = max(1, len(propositions))
        n_contr = len(contradictions)
        n_redund = len(redundancy)
        noise = max(0.0, min(1.0, n_redund / n_props))

        coherence_raw = 1.0 - (n_contr + 0.5 * noise * n_props) / n_props
        coherence = max(0.0, min(1.0, coherence_raw))

        return CoherenceReport(
            text=text,
            propositions=propositions,
            contradictions=contradictions,
            redundancy=redundancy,
            coherence=coherence,
            noise=noise,
        )

    def _split_sentences(self, text: str) -> List[str]:
        parts = [s.strip() for s in self.SENTENCE_SPLIT.split(text) if s.strip()]
        return parts

    def _extract_propositions(self, sentence: str) -> List[Proposition]:
        # Normalizzazione minima: lower, rimozione punteggiatura semplice
        base = re.sub(r"[^\w\s]", " ", sentence.lower())
        base = re.sub(r"\s+", " ", base).strip()

        polarity = self._polarity(base)
        normalized = self._normalize_statement(base)

        if not normalized:
            return []

        return [Proposition(raw=sentence.strip(), normalized=normalized, polarity=polarity)]

    def _polarity(self, text: str) -> int:
        for pat in self.NEGATION_PATTERNS:
            if re.search(pat, text):
                return -1
        return +1

    def _normalize_statement(self, text: str) -> str:
        # Rimuove negazioni e stopwords banali per isolare il nucleo concettuale.
        t = text
        for pat in self.NEGATION_PATTERNS:
            t = re.sub(pat, " ", t)
        t = re.sub(
            r"\b(io|tu|lui|lei|noi|voi|loro|sono|sei|era|ero|siete|siamo|sarò|sarai|sará|saranno)\b",
            " ",
            t,
        )
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _find_contradictions(self, props: List[Proposition]) -> List[Tuple[Proposition, Proposition]]:
        contradictions = []
        by_key: Dict[str, List[Proposition]] = {}
        for p in props:
            if not p.normalized:
                continue
            by_key.setdefault(p.normalized, []).append(p)

        for key, group in by_key.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if group[i].polarity != group[j].polarity:
                        contradictions.append((group[i], group[j]))
        return contradictions

    def _find_redundancy(self, props: List[Proposition]) -> List[Tuple[Proposition, Proposition]]:
        redundancy = []
        seen_pairs = set()
        for i in range(len(props)):
            for j in range(i + 1, len(props)):
                if (
                    props[i].normalized
                    and props[i].normalized == props[j].normalized
                    and props[i].polarity == props[j].polarity
                ):
                    key = (i, j)
                    if key not in seen_pairs:
                        redundancy.append((props[i], props[j]))
                        seen_pairs.add(key)
        return redundancy

if __name__ == "__main__":
    engine = MBXCoherenceEngine()
    sample = "Tutti gli uomini sono mortali. Socrate è un uomo. Socrate non è mortale."
    report = engine.analyze(sample)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))