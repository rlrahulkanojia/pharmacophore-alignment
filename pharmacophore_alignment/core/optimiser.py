"""
Optimisation wrappers.

Provides two strategies:
- ``optimise_guess``: direct Nelder-Mead with full objective.
- ``score_only_then_refine``: two-pass (score basin discovery → clash
  refinement).
"""

from scipy.optimize import minimize

from .geometry import apply_transform
from .scoring import compute_score, has_clash
from .objectives import objective_full, objective_score_only


def optimise_guess(coords, guess, feat_atoms, sites, ec, er):
    """
    Direct optimisation with full objective (score + clash penalty).

    Returns
    -------
    (score, params, has_clash_bool)
    """
    res = minimize(
        objective_full,
        guess,
        args=(coords, feat_atoms, sites, ec, er),
        method="Nelder-Mead",
        options={"maxiter": 2500, "xatol": 1e-5, "fatol": 1e-7},
    )
    tc = apply_transform(coords, res.x)
    sc = compute_score(tc, feat_atoms, sites)
    return sc, res.x.copy(), has_clash(tc, ec, er)


def score_only_then_refine(coords, guess, feat_atoms, sites, ec, er):
    """
    Two-pass optimisation:
      1. Minimise score-only (ignores clashes) to find a high-score basin.
      2. Refine with full objective to resolve clashes.

    Used sparingly — once per conformer as an alternative path.

    Returns
    -------
    (score, params, has_clash_bool)
    """
    res1 = minimize(
        objective_score_only,
        guess,
        args=(coords, feat_atoms, sites),
        method="Nelder-Mead",
        options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-6},
    )
    res2 = minimize(
        objective_full,
        res1.x,
        args=(coords, feat_atoms, sites, ec, er),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-7},
    )
    tc = apply_transform(coords, res2.x)
    sc = compute_score(tc, feat_atoms, sites)
    return sc, res2.x.copy(), has_clash(tc, ec, er)
