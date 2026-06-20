"""
3D conformer generation with diversity enhancement.

Pipeline:
  1. ETKDG v3 embedding (up to 200 conformers).
  2. Fallback with relaxed pruning / random coords for rigid molecules.
  3. Feature identification BEFORE MMFF (prevents aromaticity corruption).
  4. MMFF94 force-field optimisation.
  5. Torsion perturbation for molecules with < 10 unique conformers.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from .features import identify_features
from .logger import log

# SMARTS for rotatable bonds (excludes amide bonds and terminal atoms)
_ROT_BOND_SMARTS = Chem.MolFromSmarts(
    "[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]"
)


def generate_conformers(mol_h, rng):
    """
    Generate diverse 3D conformers.

    Parameters
    ----------
    mol_h : rdkit.Chem.Mol
        Molecule with explicit hydrogens.
    rng : np.random.RandomState
        Random state for torsion perturbation.

    Returns
    -------
    (mol_h, cids, feat_atoms)
        mol_h with embedded conformers, list of conformer IDs,
        and feature-to-atom mapping dict.
    """
    # ── Pass 1: standard ETKDG ──
    log.debug("  Conformer pass 1: ETKDG v3 (pruneRms=0.35)...")
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    ps.numThreads = 0
    ps.pruneRmsThresh = 0.35
    ps.useSmallRingTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=200, params=ps))
    log.debug("  Conformer pass 1: %d conformers", len(cids))

    # ── Pass 2: relaxed params for rigid molecules ──
    if len(cids) < 5:
        log.debug("  Conformer pass 2: relaxed (pruneRms=0.05, randomCoords)...")
        ps2 = AllChem.ETKDGv3()
        ps2.randomSeed = 123
        ps2.numThreads = 0
        ps2.pruneRmsThresh = 0.05
        ps2.useRandomCoords = True
        ps2.useSmallRingTorsions = True
        AllChem.EmbedMultipleConfs(mol_h, numConfs=200, params=ps2)
        cids = list(range(mol_h.GetNumConformers()))
        log.debug("  Conformer pass 2: total %d conformers", len(cids))

    # ── Pass 3: last-resort random coords ──
    if not cids:
        log.debug("  Conformer pass 3: last-resort random coords...")
        ps.useRandomCoords = True
        ps.pruneRmsThresh = 0.01
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=200, params=ps))

    if not cids:
        return mol_h, [], {}

    # ── Feature identification BEFORE MMFF ──
    feat_atoms = identify_features(mol_h)
    log.debug("  Features identified (pre-MMFF): %s",
              {k: len(v) for k, v in feat_atoms.items()})

    # ── Force-field optimisation ──
    log.debug("  Running MMFF94 optimisation on %d conformers...", len(cids))
    AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0, maxIters=400)

    # ── Torsion perturbation for low-diversity sets ──
    n_before = len(cids)
    rot_bonds = list(mol_h.GetSubstructMatches(_ROT_BOND_SMARTS))
    if len(cids) < 10 and rot_bonds:
        log.debug("  Torsion perturbation: %d rotatable bonds on %d conformers",
                  len(rot_bonds), len(cids))
        cids = _perturb_torsions(mol_h, cids, rot_bonds, rng)

    log.info("  %d conformers (%d ETKDG + %d torsion)",
             len(cids), n_before, len(cids) - n_before)
    return mol_h, cids, feat_atoms


def _perturb_torsions(mol_h, cids, rot_bonds, rng):
    """Add conformers by perturbing rotatable-bond torsion angles."""
    angles = [60, 120, 180, -60, -120]
    new_cids = []

    for cid in list(cids)[:3]:
        for bond in rot_bonds[:4]:
            for angle_deg in rng.choice(angles, size=2, replace=False):
                new_cid = mol_h.AddConformer(
                    mol_h.GetConformer(cid), assignId=True
                )
                try:
                    i, j = bond
                    nbrs_i = [
                        n.GetIdx()
                        for n in mol_h.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetIdx() != j
                    ]
                    nbrs_j = [
                        n.GetIdx()
                        for n in mol_h.GetAtomWithIdx(j).GetNeighbors()
                        if n.GetIdx() != i
                    ]
                    if nbrs_i and nbrs_j:
                        rdMolTransforms.SetDihedralDeg(
                            mol_h.GetConformer(new_cid),
                            nbrs_i[0], i, j, nbrs_j[0],
                            float(angle_deg),
                        )
                        new_cids.append(new_cid)
                except Exception:
                    mol_h.RemoveConformer(new_cid)

    if new_cids:
        log.debug("  Optimising %d torsion-perturbed conformers...", len(new_cids))
        for nc in new_cids:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, confId=nc, maxIters=300)
            except Exception:
                pass

    return list(range(mol_h.GetNumConformers()))
