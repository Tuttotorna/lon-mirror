from enum import Enum, auto

class InferenceState(Enum):
    RIGID_INVARIANCE = auto()        # S1
    ELASTIC_INVARIANCE = auto()      # S2
    META_STABLE = auto()             # S3
    COHERENT_DRIFT = auto()          # S4
    PRE_LIMIT_FRAGMENTATION = auto() # S5