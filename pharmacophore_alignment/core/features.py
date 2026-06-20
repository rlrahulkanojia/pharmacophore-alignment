"""
Pharmacophore feature identification.

Maps ligand atoms to pharmacophore families (Donor, Acceptor, Hydrophobe,
Aromatic) using RDKit's BaseFeatures.fdef factory with broadened SMARTS
fallbacks.

IMPORTANT: Must be called BEFORE force-field optimisation (MMFF) to avoid
aromaticity flag corruption.
"""

import os

from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

# Build the factory once at import time
_FDEF_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)

# RDKit family names → task family names
_FAMILY_MAP = {
    "Donor": "Donor",
    "Acceptor": "Acceptor",
    "Hydrophobe": "Hydrophobe",
    "LumpedHydrophobe": "Hydrophobe",
    "Aromatic": "Aromatic",
}

# Broadened SMARTS fallbacks per family.
# Applied only when the factory finds zero atoms for a given family.
_SMARTS_FALLBACKS = {
    "Donor": ["[#7;!H0]", "[#8;!H0]", "[#7;H1;+0]", "[#7;H2]"],
    "Acceptor": ["[#7]", "[#8]", "[#9]"],
    "Aromatic": ["[a]"],
    "Hydrophobe": [
        "[CH3]", "[CH2]", "[CH1;!$(C=O)]", "[c]",
        "[S;X2]", "[Cl,Br,I]",
    ],
}


def identify_features(mol):
    """
    Classify ligand atoms into pharmacophore families.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with explicit hydrogens (call ``Chem.AddHs`` first).

    Returns
    -------
    dict[str, list[int]]
        Mapping of family name to sorted atom indices.
    """
    feats = {k: set() for k in ("Donor", "Acceptor", "Hydrophobe", "Aromatic")}

    # Primary: RDKit feature factory
    for feat in _FACTORY.GetFeaturesForMol(mol):
        family = _FAMILY_MAP.get(feat.GetFamily())
        if family:
            feats[family].update(feat.GetAtomIds())

    # Secondary: SMARTS fallbacks for empty families
    for family, smarts_list in _SMARTS_FALLBACKS.items():
        if not feats[family]:
            for sma in smarts_list:
                pat = Chem.MolFromSmarts(sma)
                if pat:
                    for match in mol.GetSubstructMatches(pat):
                        feats[family].add(match[0])

    return {k: sorted(v) for k, v in feats.items()}
