"""
Initial-guess generators for rigid-body alignment.

All functions return lists of 6-element parameter vectors
``[rx, ry, rz, tx, ty, tz]`` suitable as starting points for
``scipy.optimize.minimize``.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ── helpers ──────────────────────────────────────────────────────────────────

def _svd_rigid(A, B, W=None):
    """
    Weighted SVD rigid alignment: find R, t such that R @ A + t ≈ B.

    Returns (rotvec, translation) or None if degenerate.
    """
    if W is None:
        W = np.ones(len(A)) / len(A)
    else:
        W = W / W.sum()
    cA = (W[:, None] * A).sum(0)
    cB = (W[:, None] * B).sum(0)
    H = ((A - cA) * W[:, None]).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    rv = Rotation.from_matrix(R).as_rotvec()
    return np.concatenate([rv, t])


# ── public generators ────────────────────────────────────────────────────────

def centroid_translation(coords, feat_atoms, sites):
    """Translation mapping feature centroid → site centroid."""
    site_ctr = np.mean([[s["x"], s["y"], s["z"]] for s in sites], axis=0)
    all_idx = sorted(set().union(*feat_atoms.values()))
    lig_ctr = coords[all_idx].mean(0) if all_idx else coords.mean(0)
    return site_ctr - lig_ctr


def weighted_svd_alignment(coords, feat_atoms, sites):
    """Per-family centroid weighted SVD alignment."""
    starts = []
    lig_pts, site_pts, weights = [], [], []
    by_fam = {}
    for s in sites:
        by_fam.setdefault(s["family"], []).append(s)
    for fam, site_list in by_fam.items():
        idxs = feat_atoms.get(fam, [])
        if not idxs:
            continue
        lig_ctr = coords[idxs].mean(0)
        for s in site_list:
            lig_pts.append(lig_ctr)
            site_pts.append([s["x"], s["y"], s["z"]])
            weights.append(s["weight"])
    if len(lig_pts) >= 2:
        p = _svd_rigid(
            np.array(lig_pts), np.array(site_pts), np.array(weights),
        )
        if p is not None:
            starts.append(p)
    return starts


def icp_svd_starts(coords, feat_atoms, sites, n_iters=3):
    """
    ICP-like nearest-atom SVD alignment.

    Each iteration:
      1. For each site, find the nearest same-family ligand atom.
      2. Compute SVD rigid transform from matched pairs.
      3. Apply and repeat.

    Every iteration produces a starting guess.
    """
    starts = []
    t0 = centroid_translation(coords, feat_atoms, sites)
    current_coords = coords + t0

    for _ in range(n_iters):
        lig_pts, site_pts, wts = [], [], []
        for s in sites:
            idxs = feat_atoms.get(s["family"], [])
            if not idxs:
                continue
            sp = np.array([s["x"], s["y"], s["z"]])
            dists = np.linalg.norm(current_coords[idxs] - sp, axis=1)
            nearest_idx = idxs[np.argmin(dists)]
            lig_pts.append(coords[nearest_idx])  # original coords
            site_pts.append(sp)
            wts.append(s["weight"])

        if len(lig_pts) < 2:
            break

        p = _svd_rigid(
            np.array(lig_pts), np.array(site_pts), np.array(wts),
        )
        if p is None:
            break
        starts.append(p)

        # Update positions for next iteration
        current_coords = (
            Rotation.from_rotvec(p[:3]).apply(coords) + p[3:6]
        )

    return starts


def anchor_starts(coords, feat_atoms, sites, rng, n=15):
    """
    Single-atom anchoring: translate so one feature atom sits on a site,
    optionally with a random rotation.
    """
    starts = []
    for s in sites:
        for aidx in feat_atoms.get(s["family"], []):
            tgt = np.array([s["x"], s["y"], s["z"]])
            # Pure translation
            starts.append(np.concatenate([[0, 0, 0], tgt - coords[aidx]]))
            # With random rotation
            rv = rng.randn(3) * 0.8
            rot_pos = Rotation.from_rotvec(rv).apply(
                coords[aidx:aidx + 1]
            ).squeeze()
            starts.append(np.concatenate([rv, tgt - rot_pos]))
            if len(starts) >= n:
                return starts
    return starts[:n]


def three_point_starts(coords, feat_atoms, sites, rng, n=12):
    """
    Match random triplets of (ligand atom, site) with same family,
    compute the SVD rigid transform as a starting guess.
    """
    pairs = []
    for s in sites:
        for aidx in feat_atoms.get(s["family"], []):
            pairs.append((aidx, np.array([s["x"], s["y"], s["z"]])))
    if len(pairs) < 3:
        return []

    starts = []
    for _ in range(n * 10):
        if len(starts) >= n:
            break
        ch = rng.choice(len(pairs), 3, replace=False)
        A = np.array([coords[pairs[c][0]] for c in ch])
        B = np.array([pairs[c][1] for c in ch])
        v1, v2 = A[1] - A[0], A[2] - A[0]
        if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
            continue  # collinear — skip
        p = _svd_rigid(A, B)
        if p is not None:
            starts.append(p)
    return starts
