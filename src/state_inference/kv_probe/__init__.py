"""Minimal paper-based single-token verifier; not an official OTV reproduction."""
from .single_probe import SingleKVProbe, linear_pseudo_target

__all__ = ["SingleKVProbe", "linear_pseudo_target"]
