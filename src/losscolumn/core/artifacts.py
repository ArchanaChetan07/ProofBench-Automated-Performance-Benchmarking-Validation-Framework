"""Which campaign artifact the analysis reads.

One place, because three reports have to agree on it: a coverage matrix built
from one campaign and a noise budget built from another would describe a machine
state that never existed.

In the library rather than beside the scripts because a script run directly puts
its own directory on the path and not the repository root, so a module shared
between scripts is importable from some invocations and not others.
"""
from pathlib import Path

# Newest first. Each supersedes the one after it, and every one is kept: the
# earlier artifacts are the record of what the corrections changed.
CANDIDATES = (
    "campaign-v2.json",                        # corrected gates and harness
    "recampaign-thrust1-communication.json",   # 21 repeats, original harness
    "campaign-thrust1-communication.json",     # 2 repeats
)


def campaign_path(art: Path = Path("artifacts")) -> Path:
    for name in CANDIDATES:
        p = art / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "no campaign artifact found; run scripts/campaign_v2.py first"
    )
