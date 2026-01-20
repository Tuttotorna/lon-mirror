from __future__ import annotations

from typing import List, Dict, Any
from .state import InferenceState


class InferenceTrajectory:
    """
    Tracks the sequence of pre-limit inference states.
    This is a sensor, not a decision mechanism.
    """

    def __init__(self) -> None:
        self.states: List[InferenceState] = []
        self.telemetry: List[Dict[str, Any]] = []

    def append(self, state: InferenceState, record: Dict[str, Any] | None = None) -> None:
        self.states.append(state)
        if record is not None:
            self.telemetry.append(record)

    def last(self) -> InferenceState | None:
        if not self.states:
            return None
        return self.states[-1]