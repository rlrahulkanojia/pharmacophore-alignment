"""
Scoring and steric clash detection.

Implements the pharmacophore alignment score:

    score = Σ  w_i · exp(-(d_i / 1.25)²)

where d_i is the minimum distance from interaction site i to the nearest
ligand atom whose chemical feature matches the site's family.
"""

import numpy as np
from scipy.spatial.distance import cdist


def compute_score(transformed, feat_atoms, sites):
    """
    Compute pharmacophore alignment score.

    Parameters
    ----------
    transformed : np.ndarray, shape (n_atoms, 3)
        Ligand atom coordinates after rigid-body transform.
    feat_atoms : dict[str, list[int]]
        Family → atom indices.
    sites : list[dict]
        Interaction sites from targets.json.

    Returns
    -------
    float
    """
    score = 0.0
    for site in sites:
        indices = feat_atoms.get(site["family"], [])
        if not indices:
            continue
        dists = np.linalg.norm(
            transformed[indices] - [site["x"], site["y"], site["z"]],
            axis=1,
        )
        d_min = dists.min()
        score += site["weight"] * np.exp(-((d_min / 1.25) ** 2))
    return score


def excl_arrays(excl_vols):
    """
    Pre-compute exclusion volume arrays for fast clash checking.

    Returns
    -------
    (centers, radii) or (None, None) if no exclusion volumes.
    """
    if not excl_vols:
        return None, None
    centers = np.array([[e["x"], e["y"], e["z"]] for e in excl_vols])
    radii = np.array([e.get("radius", 1.2) for e in excl_vols])
    return centers, radii


def clash_penalty(transformed, ec, er, tolerance=0.1, weight=100.0):
    """
    Smooth quadratic penalty for exclusion-sphere violations.

    Used inside the objective function to push the optimiser away
    from clashing configurations.
    """
    if ec is None:
        return 0.0
    dists = cdist(transformed, ec)
    violations = np.maximum(0, (er - tolerance)[None, :] - dists)
    return weight * np.sum(violations ** 2)


def has_clash(transformed, ec, er, tolerance=0.1):
    """
    Boolean clash check.

    Returns True if any ligand atom is within (radius - tolerance) of
    an exclusion centre.  Default: 1.2 - 0.1 = 1.1 Å.
    """
    if ec is None:
        return False
    dists = cdist(transformed, ec)
    return bool(np.any(dists < (er - tolerance)[None, :]))
