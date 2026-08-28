"""losscolumn -- falsifiable performance claims for LLM systems.

The package is organised in three layers:

``losscolumn.core``
    Hardware-free analysis: measurement envelopes, paired statistics,
    loss-region extraction, cliff detection, Pareto frontiers, tuning-budget
    parity accounting and pre-registration sealing.  Depends on numpy only, so
    it installs and runs on a laptop, in CI, and on a login node.

``losscolumn.validate`` / ``losscolumn.report``
    The reporting standard as an executable conformance checker, plus the
    renderers that turn a conforming claim into a human-readable artifact.

``losscolumn.thrusts``
    The three empirical harnesses (overlap envelope, equal-tuning engine audit,
    attention-kernel reproduction).  Every harness is written against a backend
    protocol with a deterministic ``synthetic`` implementation, so the entire
    analysis path is exercised before any GPU is rented.
"""

from losscolumn.version import __version__

__all__ = ["__version__"]
