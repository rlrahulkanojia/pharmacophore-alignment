"""
Centralised logging for the pharmacophore alignment pipeline.

Provides a pre-configured logger ``log`` and a ``configure()`` helper
that sets the verbosity level from the CLI.

Usage in any module::

    from .logger import log

    log.info("Starting Phase 1...")
    log.debug("Conformer %d: score=%.4f", cid, score)
"""

import logging
import sys

# Package-wide logger
log = logging.getLogger("pharmacophore_alignment")

# Guard against duplicate handlers when reimported
if not log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            fmt="[%(levelname).1s] %(message)s",
        )
    )
    log.addHandler(_handler)
    log.setLevel(logging.INFO)


def configure(verbose=False):
    """
    Set log verbosity.

    Parameters
    ----------
    verbose : bool
        If True, set level to DEBUG (shows per-conformer details).
        Otherwise INFO (shows phase summaries only).
    """
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
