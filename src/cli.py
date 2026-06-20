"""
Command-line interface.

Usage::

    python -m pharmacophore_alignment
    python -m pharmacophore_alignment --input targets.json --output poses.sdf
    python -m pharmacophore_alignment --verbose

In the competition container the defaults resolve to
``/root/data/targets.json`` → ``/root/results/docked_poses.sdf``.
"""

import argparse
import json
import os
import sys
import time
import warnings

from rdkit import Chem, RDLogger

from .core.solver import solve_target
from .core.logger import log, configure

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# Default paths (competition environment)
_DEFAULT_INPUT = "/root/data/targets.json"
_DEFAULT_OUTPUT = "/root/results/docked_poses.sdf"


def _resolve_paths(input_path, output_path):
    """Resolve paths, falling back to the package's parent dir for dev."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.normpath(os.path.join(pkg_dir, ".."))

    if not os.path.exists(input_path):
        local = os.path.join(parent_dir, "targets.json")
        if os.path.exists(local):
            input_path = local
        else:
            log.error("Input not found: %s", input_path)
            sys.exit(1)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        output_path = os.path.join(parent_dir, "docked_poses.sdf")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    return input_path, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Geometric Pharmacophore Alignment — cross-docking solver",
    )
    parser.add_argument(
        "--input", "-i",
        default=_DEFAULT_INPUT,
        help="Path to targets.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o",
        default=_DEFAULT_OUTPUT,
        help="Path for output SDF file (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )
    args = parser.parse_args()

    configure(verbose=args.verbose)

    input_path, output_path = _resolve_paths(args.input, args.output)

    log.info("=" * 60)
    log.info("Geometric Pharmacophore Alignment")
    log.info("=" * 60)
    log.info("Input:  %s", input_path)
    log.info("Output: %s", output_path)
    log.info("")

    with open(input_path) as f:
        targets = json.load(f, object_pairs_hook=lambda pairs: dict(pairs))

    log.info("Loaded %d targets\n", len(targets))

    t_total = time.time()
    writer = Chem.SDWriter(output_path)
    scores = {}

    for idx, (target_name, target_data) in enumerate(targets.items(), 1):
        log.info("━" * 50)
        log.info("Target %d/%d: %s", idx, len(targets), target_name)
        log.info("━" * 50)

        mol, score = solve_target(target_name, target_data)
        if mol is not None:
            writer.write(mol)
            scores[target_name] = score
        else:
            log.error("  FAILED — no pose produced")
        log.info("")

    writer.close()

    # ── Summary ──
    elapsed = time.time() - t_total
    total = sum(scores.values())
    max_total = sum(
        sum(s["weight"] for s in t["interaction_sites"]) for t in targets.values()
    )

    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    log.info("")
    log.info("  %-12s  %8s  %8s  %6s", "Target", "Score", "Max", "%")
    log.info("  %-12s  %8s  %8s  %6s", "─" * 12, "─" * 8, "─" * 8, "─" * 6)
    for tname, tdata in targets.items():
        sc = scores.get(tname, 0)
        mx = sum(s["weight"] for s in tdata["interaction_sites"])
        pct = 100 * sc / mx if mx > 0 else 0
        log.info("  %-12s  %8.4f  %8.3f  %5.1f%%", tname, sc, mx, pct)
    log.info("  %-12s  %8s  %8s  %6s", "─" * 12, "─" * 8, "─" * 8, "─" * 6)
    log.info("  %-12s  %8.4f  %8.3f  %5.1f%%", "TOTAL", total, max_total,
             100 * total / max_total)
    log.info("")
    log.info("Output: %s", output_path)
    log.info("Time:   %.1fs", elapsed)
    log.info("=" * 60)
