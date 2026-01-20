from typing import List
from .state import InferenceState

class InferenceTrajectory:
    def __init__(self):
        self.states: List[InferenceState] = []

    def append(self, state: InferenceState):
        self.states.append(state)

    def last(self) -> InferenceState | None:
        return self.states[-1] if self.states else None

    def irreversible(self) -> bool:
        return InferenceState.COHERENT_DRIFT in self.states

    def approaching_limit(self) -> bool:
        return self.states.count(InferenceState.PRE_LIMIT_FRAGMENTATION) >= 1