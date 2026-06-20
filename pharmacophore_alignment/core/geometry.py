"""
Low-level geometry helpers.

Coordinate extraction from RDKit conformers and rigid-body transforms
parameterised as axis-angle rotation + translation (6 DOF).
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
    return np.array(
        [[p.x, p.y, p.z]
         for p in (conf.GetAtomPosition(i)
                   for i in range(mol.GetNumAtoms()))]
    )


def apply_transform(coords, params):
    """
    Apply a rigid-body transform to coordinates.

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
    rot = Rotation.from_rotvec(params[:3])
    return rot.apply(coords) + params[3:6]
