"""How many bytes a collective actually moves.

One definition, in a module both `core` and `thrusts` can import, because it
was written out by hand in three places and two of them were wrong in the same
way for the entire life of the project.

The convention is nccl-tests' bus bandwidth, so a number here is comparable with
one from `nccl-tests` rather than merely internally consistent.
"""

from __future__ import annotations

__all__ = ["effective_bytes"]


def effective_bytes(collective: str, world: int, nbytes: int) -> float:
    """Bytes crossing the link for one call, from the per-rank INPUT size.

    That input/output distinction is the whole of it. all_reduce's output is the
    same size as its input, so the standard figure is ``2 (w-1)/w`` times it.
    all_gather's output is ``w`` times its input, and the convention is stated
    against the output -- so expressed in the input it is ``(w-1)`` times, not
    ``(w-1)/w``.

    Reading the second as the first understates all_gather by a factor of the
    world size: 2x at world 2, 8x at world 8. It is how a PCIe 4.0 x16 link came
    to be reported at 0.5 GB/s, and it stood because the number was never
    checked against a figure from outside this codebase.

    What it did and did not corrupt is worth being exact about. World size is
    fixed within a group, so the error was one constant factor per group. It
    rescales a fitted bandwidth term and leaves every relative quantity alone --
    held-out error, coverage, and the detected thresholds, which rest on timings
    rather than on bytes.
    """
    if world <= 1:
        return float(nbytes)
    if collective == "all_reduce":
        return 2.0 * (world - 1) / world * nbytes
    if collective == "all_gather":
        return float((world - 1) * nbytes)
    # An unknown collective is counted as moving its buffer once, which is the
    # least misleading guess, and the caller is not told it was a guess -- so
    # anything added here should be given its own case.
    return float(nbytes)
