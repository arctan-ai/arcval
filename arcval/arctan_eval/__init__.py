"""Arctan-vs-baseline STT comparison module."""

from arcval.arctan_eval.benchmark import run
from arcval.arctan_eval.leaderboard import generate_leaderboard

__all__ = [
    "run",
    "generate_leaderboard",
]
