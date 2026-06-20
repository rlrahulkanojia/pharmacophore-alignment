"""
Per-target solver.

Orchestrates the full pipeline for a single target:
  Phase 1 — Quick screen every conformer.
  Phase 2 — Deep optimise top-K with diverse starts + DE + basin-hopping.
  Local refinement on the winner.
  Fallback for clash-free failure.
  Output molecule construction.
"""

import time

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from rdkit import Chem

from .geometry import get_coords, apply_transform
from .scoring import ScoringContext, _fast_transform, compute_score, has_clash, excl_arrays
from .alignment import (
    centroid_translation,
    weighted_svd_alignment,
    icp_svd_starts,
    anchor_starts,
    three_point_starts,
)
from .conformers import generate_conformers
from .optimiser import optimise_guess, score_only_then_refine
from .logger import log


def solve_target(name, data):
    """
    Solve one target: generate conformers, align, score, return best pose.

    Parameters
    ----------
    name : str
        Target identifier.
    data : dict
        Target data with keys ``smiles``, ``interaction_sites``,
        ``excluded_volumes``.

    Returns
    -------
    (mol, score) or (None, -1) on failure.
        mol is an RDKit Mol with a single 3D conformer, hydrogens removed.
    """
    t_start = time.time()
    smiles = data["smiles"]
    sites = data["interaction_sites"]
    excl = data.get("excluded_volumes", [])

    log.info("  SMILES: %s", smiles)
    log.info("  Interaction sites: %d  |  Exclusion volumes: %d",
             len(sites), len(excl))

    fam_counts = {}
    for s in sites:
        fam_counts[s["family"]] = fam_counts.get(s["family"], 0) + 1
    log.info("  Site families: %s", fam_counts)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.error("  Bad SMILES: %s", smiles)
        return None, -1
    log.info("  Heavy atoms: %d", mol.GetNumAtoms())
    mol_h = Chem.AddHs(mol)
    log.debug("  Atoms (with H): %d", mol_h.GetNumAtoms())

    rng = np.random.RandomState(42)

    # ── Conformer generation ──
    mol_h, cids, feat_atoms = generate_conformers(mol_h, rng)
    if not cids:
        log.error("  Conformer generation failed")
        return None, -1

    needed = {s["family"] for s in sites}
    feat_summary = {k: len(v) for k, v in feat_atoms.items() if k in needed}
    log.info("  Matched features: %s", feat_summary)

    unmatchable = [
        (s["family"], s["weight"])
        for s in sites if not feat_atoms.get(s["family"])
    ]
    if unmatchable:
        log.warning("  Unmatchable sites (no ligand atoms): %s", unmatchable)

    max_score = sum(s["weight"] for s in sites)
    achievable = max_score - sum(w for _, w in unmatchable)
    log.info("  Max possible score: %.3f  (achievable: %.3f)", max_score, achievable)

    # ── Build ScoringContext (pre-compute arrays once) ──
    ctx = ScoringContext(feat_atoms, sites, excl)
    # Legacy arrays for DE/BH that still use the old API
    ec, er = excl_arrays(excl)

    # ── Phase 1: quick screen ──
    t_p1 = time.time()
    log.info("  Phase 1: screening %d conformers...", len(cids))
    screen_results = _phase1_screen(mol_h, cids, feat_atoms, sites, ctx, rng)

    if screen_results:
        log.info("  Phase 1 done [%.1fs]: %d clash-free poses, top=%.4f (%.1f%%)",
                 time.time() - t_p1,
                 len(screen_results),
                 screen_results[0][0],
                 100 * screen_results[0][0] / max_score)
    else:
        log.warning("  Phase 1: no clash-free poses found")

    # ── Phase 2: deep optimise ──
    t_p2 = time.time()
    best_score, best_cid, best_params = _phase2_deep(
        mol_h, cids, screen_results, feat_atoms, sites, ctx, ec, er, rng,
    )
    log.info("  Phase 2 done [%.1fs]", time.time() - t_p2)

    # ── Local refinement ──
    if best_params is not None:
        pre_refine = best_score
        best_score, best_params = _local_refine(
            mol_h, best_cid, best_params, best_score, ctx,
        )
        if best_score > pre_refine:
            log.info("  Local refinement improved: %.4f → %.4f", pre_refine, best_score)
        else:
            log.debug("  Local refinement: no improvement")

    # ── Fallback ──
    if best_params is None:
        best_score, best_cid, best_params = _fallback(
            mol_h, cids, feat_atoms, sites, ctx,
        )

    pct = 100 * best_score / max_score if max_score > 0 else 0
    elapsed = time.time() - t_start
    log.info("  ✓ Best score: %.4f / %.3f (%.1f%%) in %.1fs",
             best_score, max_score, pct, elapsed)

    return _build_output(mol_h, best_cid, best_params, name, best_score), best_score


# ── internal helpers ─────────────────────────────────────────────────────────

def _phase1_screen(mol_h, cids, feat_atoms, sites, ctx, rng):
    """Quick screen every conformer with fewer guesses for speed."""
    results = []
    n_clashing = 0
    for idx, cid in enumerate(cids):
        coords = get_coords(mol_h, cid)
        t0 = centroid_translation(coords, feat_atoms, sites)
        guesses = [np.concatenate([[0, 0, 0], t0])]
        guesses.extend(anchor_starts(coords, feat_atoms, sites, rng, n=4))
        guesses.extend(icp_svd_starts(coords, feat_atoms, sites, n_iters=2))

        local_best_sc = -np.inf
        local_best_p = None
        for g in guesses:
            sc, px, clash = optimise_guess(coords, g, ctx)
            if not clash and sc > local_best_sc:
                local_best_sc = sc
                local_best_p = px.copy()

        if local_best_p is not None:
            results.append((local_best_sc, cid, local_best_p))
            log.debug("    conf %3d/%d: score=%.4f ✓", idx + 1, len(cids), local_best_sc)
        else:
            n_clashing += 1
            log.debug("    conf %3d/%d: all poses clashed ✗", idx + 1, len(cids))

    if n_clashing:
        log.debug("  Phase 1: %d conformers had no clash-free pose", n_clashing)

    results.sort(key=lambda x: -x[0])
    return results


def _phase2_deep(mol_h, cids, screen_results, feat_atoms, sites, ctx, ec, er, rng):
    """Deep optimise top-K conformers."""
    TOP_K = min(15, len(screen_results)) if screen_results else min(15, len(cids))
    top_cids = (
        [r[1] for r in screen_results[:TOP_K]]
        if screen_results
        else cids[:TOP_K]
    )
    top_params = (
        {r[1]: r[2] for r in screen_results[:TOP_K]}
        if screen_results
        else {}
    )
    log.info("  Phase 2: deep-optimising top %d conformers...", len(top_cids))

    best_score = screen_results[0][0] if screen_results else -np.inf
    best_cid = screen_results[0][1] if screen_results else cids[0]
    best_params = screen_results[0][2] if screen_results else None

    for ci, cid in enumerate(top_cids):
        coords = get_coords(mol_h, cid)
        t0 = centroid_translation(coords, feat_atoms, sites)

        # Build diverse starting guesses (trimmed from ~80 → ~50)
        guesses = [np.concatenate([[0, 0, 0], t0])]
        guesses.extend(weighted_svd_alignment(coords, feat_atoms, sites))
        guesses.extend(icp_svd_starts(coords, feat_atoms, sites, n_iters=3))
        guesses.extend(anchor_starts(coords, feat_atoms, sites, rng, n=12))
        guesses.extend(three_point_starts(coords, feat_atoms, sites, rng, n=10))
        for _ in range(10):
            rv = rng.randn(3) * np.pi
            guesses.append(np.concatenate([rv, t0 + rng.randn(3) * 3.0]))
        # Reduced rotation grid (12 instead of 24)
        for rx in [0, np.pi / 2, np.pi]:
            for ry in [0, np.pi / 2]:
                for rz in [0, np.pi / 2]:
                    guesses.append(np.concatenate([[rx, ry, rz], t0]))

        log.debug("    conf %d/%d: %d guesses...",
                  ci + 1, len(top_cids), len(guesses))

        conf_best = -np.inf
        for g in guesses:
            sc, px, clash = optimise_guess(coords, g, ctx)
            if not clash and sc > best_score:
                best_score, best_cid, best_params = sc, cid, px.copy()
            if not clash and sc > conf_best:
                conf_best = sc

        # Score-only-then-refine: one extra attempt via ICP start
        icp = icp_svd_starts(coords, feat_atoms, sites, n_iters=3)
        if icp:
            sc, px, clash = score_only_then_refine(coords, icp[0], ctx)
            if not clash and sc > best_score:
                best_score, best_cid, best_params = sc, cid, px.copy()
            if not clash and sc > conf_best:
                conf_best = sc

        # Differential evolution (still uses legacy API — acceptable overhead)
        bounds = [(-np.pi, np.pi)] * 3 + [
            (t0[i] - 12, t0[i] + 12) for i in range(3)
        ]
        try:
            res = differential_evolution(
                ctx.objective,
                bounds,
                args=(coords,),
                seed=42 + ci,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.9,
            )
            tc = _fast_transform(coords, res.x)
            sc = ctx.score(tc)
            if not ctx.clashes(tc) and sc > best_score:
                best_score, best_cid, best_params = sc, cid, res.x.copy()
            log.debug("    conf %d DE: score=%.4f", ci + 1, sc)
        except Exception:
            log.debug("    conf %d DE: failed", ci + 1)

        # Basin-hopping on top-3 conformers only
        if ci < 3:
            bh_start = top_params.get(cid, best_params)
            if bh_start is not None:
                try:
                    bh_res = basinhopping(
                        ctx.objective,
                        bh_start,
                        minimizer_kwargs=dict(
                            args=(coords,),
                            method="Nelder-Mead",
                            options={"maxiter": 1500},
                        ),
                        niter=25,
                        seed=42 + ci,
                        stepsize=1.5,
                        T=2.0,
                    )
                    tc = _fast_transform(coords, bh_res.x)
                    sc = ctx.score(tc)
                    if not ctx.clashes(tc) and sc > best_score:
                        best_score, best_cid = sc, cid
                        best_params = bh_res.x.copy()
                    log.debug("    conf %d BH: score=%.4f", ci + 1, sc)
                except Exception:
                    log.debug("    conf %d BH: failed", ci + 1)

        log.debug("    conf %d/%d best=%.4f  (global best=%.4f)",
                  ci + 1, len(top_cids), conf_best, best_score)

    return best_score, best_cid, best_params


def _local_refine(mol_h, best_cid, best_params, best_score, ctx):
    """Polish the best pose with multiple local optimisers."""
    coords = get_coords(mol_h, best_cid)
    for method in ("Powell", "Nelder-Mead"):
        res = minimize(
            ctx.objective,
            best_params,
            args=(coords,),
            method=method,
            options={"maxiter": 5000},
        )
        tc = _fast_transform(coords, res.x)
        sc = ctx.score(tc)
        if not ctx.clashes(tc) and sc > best_score:
            log.debug("  Refinement (%s): %.4f → %.4f", method, best_score, sc)
            best_score = sc
            best_params = res.x.copy()
    return best_score, best_params


def _fallback(mol_h, cids, feat_atoms, sites, ctx):
    """Last resort: optimise score only (ignore clashes)."""
    log.warning("  No clash-free pose — relaxing constraint")
    best_score = -np.inf
    best_cid = cids[0]
    best_params = None

    for cid in cids[:30]:
        coords = get_coords(mol_h, cid)
        t0 = centroid_translation(coords, feat_atoms, sites)
        g = np.concatenate([[0, 0, 0], t0])
        res = minimize(
            ctx.objective_score_only,
            g,
            args=(coords,),
            method="Nelder-Mead",
            options={"maxiter": 2000},
        )
        tc = _fast_transform(coords, res.x)
        sc = ctx.score(tc)
        if sc > best_score:
            best_score, best_cid, best_params = sc, cid, res.x.copy()

    return best_score, best_cid, best_params


def _build_output(mol_h, best_cid, best_params, name, score):
    """Create output Mol with single transformed conformer, Hs removed."""
    coords = get_coords(mol_h, best_cid)
    tc = apply_transform(coords, best_params)

    out_h = Chem.RWMol(mol_h)
    ids_to_remove = [
        c.GetId() for c in out_h.GetConformers() if c.GetId() != best_cid
    ]
    for rid in ids_to_remove:
        out_h.RemoveConformer(rid)

    conf = out_h.GetConformer(best_cid)
    for i in range(out_h.GetNumAtoms()):
        conf.SetAtomPosition(i, tc[i].tolist())

    out = Chem.RemoveHs(out_h.GetMol())
    out.SetProp("_Name", name)
    out.SetProp("Score", f"{score:.4f}")
    log.debug("  Output mol: %d atoms, 1 conformer", out.GetNumAtoms())
    return out
