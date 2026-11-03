"""Read-only state snapshots. Optional torch dependency is loaded on use."""
from .snapshot import StateSnapshot, capture_states

__all__ = ["StateSnapshot", "capture_states"]
