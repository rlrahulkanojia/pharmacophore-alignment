"""
Optimisation wrappers using ScoringContext for fused hot-path.
"""

from scipy.optimize import minimize

from .scoring import ScoringContext


def optimise_guess(coords, guess, ctx):
    """
    Direct Nelder-Mead with fused objective.

    Parameters
    ----------
    coords : np.ndarray (n, 3)
    guess : array-like (6,)
    ctx : ScoringContext

    Returns
    -------
    (score, params, has_clash_bool)
    """
    res = minimize(
        ctx.objective,
        guess,
        args=(coords,),
        method="Nelder-Mead",
        options={"maxiter": 2500, "xatol": 1e-5, "fatol": 1e-7},
    )
    from .scoring import _fast_transform
    tc = _fast_transform(coords, res.x)
    sc = ctx.score(tc)
    return sc, res.x.copy(), ctx.clashes(tc)


def score_only_then_refine(coords, guess, ctx):
    """
    Two-pass: score-only → refine with clash penalty.

    Returns
    -------
    (score, params, has_clash_bool)
    """
    res1 = minimize(
        ctx.objective_score_only,
        guess,
        args=(coords,),
        method="Nelder-Mead",
        options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-6},
    )
    res2 = minimize(
        ctx.objective,
        res1.x,
        args=(coords,),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-7},
    )
    from .scoring import _fast_transform
    tc = _fast_transform(coords, res2.x)
    sc = ctx.score(tc)
    return sc, res2.x.copy(), ctx.clashes(tc)
