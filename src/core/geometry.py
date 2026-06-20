"""
Low-level geometry helpers.

Coordinate extraction from RDKit conformers and rigid-body transforms
parameterised as axis-angle rotation + translation (6 DOF).

The rotation uses the Rodrigues formula directly in numpy — ~10×
faster than ``scipy.spatial.transform.Rotation.from_rotvec()`` because
it avoids object construction on every call.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def get_coords(mol, conf_id):
    """
    Extract atom coordinates from a conformer.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
    conf_id : int
        Conformer ID.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
    """
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    coords = np.empty((n, 3))
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        coords[i, 0] = pos.x
        coords[i, 1] = pos.y
        coords[i, 2] = pos.z
    return coords


def apply_transform(coords, params):
    """
    Apply a rigid-body transform to coordinates using the Rodrigues
    rotation formula (pure numpy — no scipy object overhead).

    Parameters
    ----------
    coords : np.ndarray, shape (n, 3)
    params : array-like, length 6
        [rx, ry, rz, tx, ty, tz] — axis-angle rotation followed by
        translation.

    Returns
    -------
    np.ndarray, shape (n, 3)
    """
    return _rodrigues_rotate(coords, params[0], params[1], params[2]) + params[3:6]


def _rodrigues_rotate(v, rx, ry, rz):
    """
    Rotate array of 3-D vectors by axis-angle (rx, ry, rz) using
    Rodrigues' formula.  Pure numpy, no object allocation.

    v_rot = v·cos(θ) + (k × v)·sin(θ) + k·(k·v)·(1 - cos(θ))
    where k = (rx,ry,rz)/θ, θ = ||(rx,ry,rz)||.
    """
    theta_sq = rx * rx + ry * ry + rz * rz
    if theta_sq < 1e-30:
        return v.copy()  # identity rotation

    theta = np.sqrt(theta_sq)
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1.0 - c

    # Unit axis
    kx, ky, kz = rx / theta, ry / theta, rz / theta

    # Rotation matrix (row-major)
    R = np.array([
        [c + kx*kx*t,      kx*ky*t - kz*s,  kx*kz*t + ky*s],
        [ky*kx*t + kz*s,   c + ky*ky*t,      ky*kz*t - kx*s],
        [kz*kx*t - ky*s,   kz*ky*t + kx*s,   c + kz*kz*t    ],
    ])

    return v @ R.T


def rotvec_to_matrix(params):
    """Convert axis-angle params[:3] to a 3×3 rotation matrix (for SVD starts)."""
    return Rotation.from_rotvec(params[:3]).as_matrix()
