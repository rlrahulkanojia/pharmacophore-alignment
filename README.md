# Geometric Pharmacophore Alignment

A cross-docking solver that places small molecules into protein binding pockets defined by pharmacophore interaction sites and exclusion spheres (no explicit protein structure required).

## Problem

Given a ligand SMILES string and a set of pharmacophore interaction sites (Donor, Acceptor, Hydrophobe, Aromatic) with 3D coordinates and weights, find the 3D conformer and rigid-body pose that maximises:

```
score = Σ  w_i · exp(-(d_i / 1.25)²)
```

where `d_i` is the minimum distance from interaction site `i` to the nearest matching-family ligand atom, subject to no steric clashes with exclusion spheres.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run (auto-detects local targets.json)
python -m pharmacophore_alignment

# With explicit paths
python -m pharmacophore_alignment --input targets.json --output poses.sdf

# Verbose mode (per-conformer details, timing, debug info)
python -m pharmacophore_alignment --verbose
```

## Project Structure

```
.
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── targets.json                      # Input data (5 targets)
├── docked_poses.sdf                  # Output (generated)
│
└── pharmacophore_alignment/          # Main package
    ├── __init__.py                   # Package version
    ├── __main__.py                   # python -m entry point
    ├── cli.py                        # CLI, argument parsing, I/O, summary table
    │
    └── core/                         # Algorithm modules
        ├── README.md                 # Module dependency graph & details
        ├── __init__.py               # Public API re-exports
        ├── logger.py                 # Centralised logging (INFO / DEBUG)
        ├── features.py               # Pharmacophore feature identification
        ├── geometry.py               # Coordinate extraction & rigid transforms
        ├── scoring.py                # Score function & clash detection
        ├── objectives.py             # Objective functions for optimisation
        ├── alignment.py              # Initial-guess generators (SVD, ICP, anchors)
        ├── conformers.py             # 3D conformer generation & diversity
        ├── optimiser.py              # Optimisation wrappers (Nelder-Mead, two-pass)
        └── solver.py                 # Per-target solver (orchestrates full pipeline)
```

## Algorithm

### Pipeline Overview

```
SMILES → 3D Conformers → Feature ID → Phase 1 Screen → Phase 2 Deep Opt → Output SDF
```

### 1. Conformer Generation (`core/conformers.py`)

- ETKDG v3 with up to 200 conformers
- Fallback with relaxed pruning for rigid molecules (e.g. caffeine)
- Torsion perturbation for low-diversity sets
- MMFF94 force-field optimisation

### 2. Feature Identification (`core/features.py`)

Maps atoms to pharmacophore families using RDKit's `BaseFeatures.fdef` factory with broadened SMARTS fallbacks. Critically, this runs **before** MMFF optimisation to prevent aromaticity flag corruption.

### 3. Alignment (`core/alignment.py`)

Multiple initial-guess strategies:
- **Centroid translation**: feature centroid → site centroid
- **Weighted SVD**: per-family centroid matching with site-weight-aware SVD
- **ICP-SVD**: iterative closest-point refinement (3–4 iterations)
- **Anchor starts**: single-atom anchoring with random rotations
- **Three-point SVD**: random triplet matching

### 4. Optimisation (`core/optimiser.py`, `core/solver.py`)

Two-phase search:
- **Phase 1**: quick screen of all conformers with cheap starts
- **Phase 2**: deep optimisation of top-20 with all start generators + differential evolution + basin-hopping

### 5. Scoring (`core/scoring.py`)

Exact implementation of `w_i · exp(-(d_i / 1.25)²)` with steric clash checking at 1.2 Å radius and 0.1 Å tolerance.

## Input Format

`targets.json` — a JSON object with target names as keys:

```json
{
  "target_1": {
    "smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "interaction_sites": [
      {"family": "Acceptor", "x": 2.45, "y": -1.32, "z": 0.87, "weight": 1.2}
    ],
    "excluded_volumes": [
      {"x": 1.5, "y": 3.2, "z": -1.8, "radius": 1.2}
    ]
  }
}
```

## Output Format

A single SDF file with one best-pose conformer per target, preserving JSON key order and original SMILES atom count/topology. Each entry has `_Name` and `Score` properties.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| rdkit   | ≥ 2022.03 | Cheminformatics, conformer generation |
| numpy   | ≥ 1.21 | Numerical arrays |
| scipy   | ≥ 1.7 | Optimisation (Nelder-Mead, DE, basin-hopping) |
