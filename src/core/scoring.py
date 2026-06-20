"""
Scoring and steric clash detection.

Implements the pharmacophore alignment score:

    score = Σ  w_i · exp(-(d_i / 1.25)²)

where d_i is the minimum distance from interaction site i to the nearest
ligand atom whose chemical feature matches the site's family.

Performance-critical: the ``ScoringContext`` class pre-computes all arrays
and provides a fused ``objective()`` method that avoids repeated dict
lookups, object allocation, and Python loops in the hot path.
"""

import numpy as np
from scipy.spatial.distance import cdist


class ScoringContext:
    """
    Pre-computed arrays for fast scoring + clash evaluation.

    Build once per conformer-set, re-use across millions of objective calls.
    """

    __slots__ = (
        "site_coords", "site_weights", "site_family_masks",
        "ec", "er", "min_allowed",
        "_inv_sigma_sq", "families",
    )

    _SIGMA = 1.25

    def __init__(self, feat_atoms, sites, excl_vols):
        n_sites = len(sites)

        # Site coordinates and weights as contiguous arrays
        self.site_coords = np.array(
            [[s["x"], s["y"], s["z"]] for s in sites], dtype=np.float64,
        )
        self.site_weights = np.array(
            [s["weight"] for s in sites], dtype=np.float64,
        )

        # Per-site: numpy array of matching atom indices
        # Stored as a list of int arrays (ragged — can't be a 2-D array)
        self.families = []
        for s in sites:
            idxs = feat_atoms.get(s["family"], [])
            self.families.append(np.array(idxs, dtype=np.intp))

        self._inv_sigma_sq = 1.0 / (self._SIGMA ** 2)

        # Exclusion volumes
        if excl_vols:
            self.ec = np.array(
                [[e["x"], e["y"], e["z"]] for e in excl_vols], dtype=np.float64,
            )
            self.er = np.array(
                [e.get("radius", 1.2) for e in excl_vols], dtype=np.float64,
            )
            self.min_allowed = self.er - 0.1  # tolerance baked in
        else:
            self.ec = None
            self.er = None
            self.min_allowed = None

    # ── core methods (hot path) ──────────────────────────────────────────

    def score(self, transformed):
        """Compute pharmacophore alignment score (vectorised)."""
        total = 0.0
        for i, idx_arr in enumerate(self.families):
            if len(idx_arr) == 0:
                continue
            # Squared distances from matching atoms to this site
            diff = transformed[idx_arr] - self.site_coords[i]
            sq_dists = np.einsum("ij,ij->i", diff, diff)
            d_min_sq = sq_dists.min()
            total += self.site_weights[i] * np.exp(-d_min_sq * self._inv_sigma_sq)
        return total

    def penalty(self, transformed, weight=100.0):
        """Smooth quadratic clash penalty."""
        if self.ec is None:
            return 0.0
        dists = cdist(transformed, self.ec)
        violations = np.maximum(0, self.min_allowed[None, :] - dists)
        return weight * np.sum(violations * violations)

    def clashes(self, transformed):
        """Boolean clash check."""
        if self.ec is None:
            return False
        dists = cdist(transformed, self.ec)
        return bool(np.any(dists < self.min_allowed[None, :]))

    def objective(self, params, coords):
        """
        Fused objective for scipy.optimize: -score + penalty.
        Inlines the rotation to avoid function-call overhead.
        """
        transformed = _fast_transform(coords, params)
        return -self.score(transformed) + self.penalty(transformed)

    def objective_score_only(self, params, coords):
        """Score-only objective (no clash penalty)."""
        transformed = _fast_transform(coords, params)
        return -self.score(transformed)


# ── standalone functions (backward compat) ───────────────────────────────

def compute_score(transformed, feat_atoms, sites):
    """Legacy wrapper — prefer ScoringContext.score()."""
    score = 0.0
    for site in sites:
        indices = feat_atoms.get(site["family"], [])
        if not indices:
            continue
        diff = transformed[indices] - [site["x"], site["y"], site["z"]]
        sq_dists = np.einsum("ij,ij->i", diff, diff)
        d_min_sq = sq_dists.min()
        score += site["weight"] * np.exp(-d_min_sq / (1.25 ** 2))
    return score


def excl_arrays(excl_vols):
    """Pre-compute exclusion volume arrays."""
    if not excl_vols:
        return None, None
    centers = np.array([[e["x"], e["y"], e["z"]] for e in excl_vols])
    radii = np.array([e.get("radius", 1.2) for e in excl_vols])
    return centers, radii


def clash_penalty(transformed, ec, er, tolerance=0.1, weight=100.0):
    """Smooth quadratic penalty for exclusion-sphere violations."""
    if ec is None:
        return 0.0
    dists = cdist(transformed, ec)
    violations = np.maximum(0, (er - tolerance)[None, :] - dists)
    return weight * np.sum(violations * violations)


def has_clash(transformed, ec, er, tolerance=0.1):
    """Boolean clash check."""
    if ec is None:
        return False
    dists = cdist(transformed, ec)
    return bool(np.any(dists < (er - tolerance)[None, :]))


# ── inlined fast transform (avoids import cycle) ────────────────────────

def _fast_transform(coords, params):
    """Rodrigues rotation + translation — inlined for zero overhead."""
    rx, ry, rz = params[0], params[1], params[2]
    theta_sq = rx * rx + ry * ry + rz * rz
    if theta_sq < 1e-30:
        return coords + params[3:6]

    theta = np.sqrt(theta_sq)
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1.0 - c
    kx, ky, kz = rx / theta, ry / theta, rz / theta

    R = np.array([
        [c + kx*kx*t,      kx*ky*t - kz*s,  kx*kz*t + ky*s],
        [ky*kx*t + kz*s,   c + ky*ky*t,      ky*kz*t - kx*s],
        [kz*kx*t - ky*s,   kz*ky*t + kx*s,   c + kz*kz*t    ],
    ])
    return coords @ R.T + params[3:6]
