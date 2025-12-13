# Legacy compatibility shim for older imports:
# from ICE.OMNIA_ICE_v0_1 import ICEInput, ice_gate

from omnia.ice import ICEInput, ice_gate

__all__ = ["ICEInput", "ice_gate"]