"""Core modules for pharmacophore alignment."""

from .features import identify_features
from .geometry import get_coords, apply_transform
from .scoring import compute_score, has_clash, clash_penalty, excl_arrays
from .objectives import objective_full, objective_score_only
from .alignment import (
    centroid_translation,
    weighted_svd_alignment,
    icp_svd_starts,
    anchor_starts,
    three_point_starts,
)
from .conformers import generate_conformers
from .optimiser import optimise_guess, score_only_then_refine
from .solver import solve_target
from .logger import log, configure

__all__ = [
    "identify_features",
    "get_coords",
    "apply_transform",
    "compute_score",
    "has_clash",
    "clash_penalty",
    "excl_arrays",
    "objective_full",
    "objective_score_only",
    "centroid_translation",
    "weighted_svd_alignment",
    "icp_svd_starts",
    "anchor_starts",
    "three_point_starts",
    "generate_conformers",
    "optimise_guess",
    "score_only_then_refine",
    "solve_target",
    "log",
    "configure",
]
