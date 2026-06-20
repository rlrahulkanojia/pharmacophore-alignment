# Core Modules

This directory contains the algorithm implementation, split by responsibility.

## Module Dependency Graph

```
cli.py
  └─ solver.py          ← orchestration
       ├─ conformers.py  ← 3D generation + feature ID
       │    └─ features.py
       ├─ optimiser.py   ← Nelder-Mead / two-pass wrappers
       │    ├─ objectives.py
       │    ├─ scoring.py
       │    └─ geometry.py
       └─ alignment.py   ← initial-guess generators
            └─ geometry.py (centroid_translation uses it implicitly)
```

## Module Details

### `features.py`
Pharmacophore feature classification. Uses RDKit's `BaseFeatures.fdef` factory as primary source, with broadened SMARTS patterns as fallbacks for molecules where the factory misses families. Must be called before MMFF optimisation.

### `geometry.py`
Two pure functions: `get_coords` (extract coordinates from an RDKit conformer) and `apply_transform` (apply an axis-angle rotation + translation). These are the hot-path functions called thousands of times during optimisation.

### `scoring.py`
The pharmacophore score formula and steric clash detection. `compute_score` implements the exact `w_i · exp(-(d/1.25)²)` formula. `clash_penalty` provides a smooth quadratic penalty for use inside optimisers. `has_clash` is the hard boolean check.

### `objectives.py`
Thin wrappers combining scoring + geometry into objective functions suitable for `scipy.optimize.minimize`. Two variants: `objective_full` (with clash penalty) and `objective_score_only` (for basin discovery).

### `alignment.py`
Five initial-guess generators, all producing 6-element parameter vectors `[rx, ry, rz, tx, ty, tz]`:
- `centroid_translation` — baseline shift
- `weighted_svd_alignment` — per-family centroid SVD
- `icp_svd_starts` — iterative closest-point with SVD
- `anchor_starts` — single-atom anchoring
- `three_point_starts` — random triplet SVD matching

### `conformers.py`
Conformer generation pipeline: ETKDG → fallbacks → feature ID → MMFF → torsion perturbation. Returns the molecule, conformer IDs, and feature mapping as a tuple.

### `optimiser.py`
Wrappers for `scipy.optimize.minimize` with pre-configured settings. `optimise_guess` is the standard path; `score_only_then_refine` is an alternative two-pass strategy used sparingly.

### `solver.py`
The main orchestrator. `solve_target` runs the full pipeline for one target: conformer generation → Phase 1 screening → Phase 2 deep optimisation → local refinement → fallback → output construction.
