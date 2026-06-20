"""
Objective functions for rigid-body optimisation.

Two variants:
- ``objective_full``: standard objective with clash penalty.
- ``objective_score_only``: ignores clashes (for basin discovery).
"""

from .geometry import apply_transform
from .scoring import compute_score, clash_penalty


def objective_full(params, coords, feat_atoms, sites, ec, er):
    """Minimise: -score + clash_penalty."""
    t = apply_transform(coords, params)
    return -compute_score(t, feat_atoms, sites) + clash_penalty(t, ec, er)


def objective_score_only(params, coords, feat_atoms, sites):
    """Minimise: -score (ignores exclusion volumes)."""
    t = apply_transform(coords, params)
    return -compute_score(t, feat_atoms, sites)
