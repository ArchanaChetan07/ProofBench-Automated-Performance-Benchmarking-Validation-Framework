# Evidence reclassification under LC-1.2

Criterion **1.48x**, derived from this machine's own restart-level repeatability. Drift classified `restart_level`.

## The campaign's two passes

They ran back to back inside one process invocation and one measurement lock, which makes them *likely* to share a machine state. Tested rather than assumed:

- worst shared probe: **2.90x**, against a noise floor of 4.91x measured from the campaign's own repeats at this probe count
- shape divergence: **7.21x**, against 15.35x
- shared probes: 276
- verdict: **COMPARABLE**

the surface did not move (0.998x median shift), every shared probe agrees within 2.90x, and the probes moved together (shape divergence 7.21x), after widening to 4.91x because measurement noise alone reaches that across 276 probes

**The campaign stands.** Its two passes differ by less than two repeats within a single launch do, so whatever separates them is the instrument and not the machine. Its pooled analysis was legitimate and nothing in it needs withdrawing.

This is a result rather than a reprieve, and it was nearly the opposite one. Judged against the registered 1.48x criterion the passes read as clearly incomparable at 2.90x -- but that criterion bounds *one* comparison of medians-of-seven, and it was being applied to the worst of 276 comparisons of medians-of-two. Splitting the campaign's own repeats within a single pass, which is the same machine state by construction, reaches 4.91x. The passes are more alike than the instrument is with itself.

## The campaign against the medium probe

- worst shared probe: **nanx**
- shape divergence: **nanx**
- shared probes: 0
- verdict: **INVALID**

only 0 shared probe(s), below the 3 the protocol requires. Two sessions that measured different things cannot be shown comparable by measuring neither.

This is the pooling that produced the impossible result. Under LC-8.1 it is now refused at the point where it would happen, rather than discovered afterwards from a number that could not be true.

## What this does and does not license

| | |
|---|---|
| Campaign-internal comparisons | permitted |
| Campaign pooled with any later session | **refused** until that session is shown comparable |
| Local gloo parameters as A100 fabric parameters | **refused** unconditionally, and not for a stability reason: a CPU-side gloo transport on one host is not an NVLink or InfiniBand fabric, and no amount of stability makes it one |