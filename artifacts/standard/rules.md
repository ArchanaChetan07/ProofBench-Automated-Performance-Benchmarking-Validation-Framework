# LC-1.1 rule registry (frozen)

54 rules, 34 fatal. Digest `sha256:c167b1a695a81a4cb388142f277c42cdc5667ac31661c2ff92947c6a36ab5684`.

Supersedes LC-1.0 (`sha256:1483dc1b908fc25155cdf160dd85c04a6d2b995bac3d61ab074446dcc9468df3`), which is unchanged: LC-1.1 adds 7 rules (LC-6.1, LC-6.2, LC-6.3, LC-7.1, LC-7.2, LC-7.3, LC-7.4) and edits none. A claim graded under 1.0 still means what it meant.

## Standard version

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-0.1` | fatal | The claim declares the standard revision it is graded against. | grading an artifact against rules it was never written for |

## Evidence class

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-0.2` | fatal | evidence_class is one of measured, simulated, mixed. | a modelled number being read as a measured one |
| `LC-0.3` | fatal | A simulated or mixed claim states what was modelled and what measured. | an unexplained simulation passing as evidence |

## Loss column

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-1.1` | fatal | A loss column is present. | selective reporting |
| `LC-1.2` | fatal | If any cell lost, at least one region describes where. | acknowledging losses without locating them |
| `LC-1.3` | fatal | The column does not declare itself incomplete. | publishing a column known to be partial |
| `LC-1.4` | fatal | The column is in the main body, not an appendix. | burying adverse regimes |
| `LC-1.5` | warning | Every loss region carries an attributed cause. | a location without a mechanism |
| `LC-1.6` | warning | Cells that could not be measured are accounted for. | an out-of-memory regime vanishing from the record |
| `LC-1.7` | warning | The inconclusive fraction is stated and tolerable. | an underdetermined envelope reading as a settled one |
| `LC-1.8` | fatal | The design is capable of declaring a loss if one exists. | an empty loss column produced by an underpowered sweep |
| `LC-1.9` | fatal | Every region's statistical support meets the claim's own thresholds. | a region asserted as a loss on evidence weaker than the claim declares |

## Tuning-budget parity

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-2.1` | fatal | A tuned comparison ships a parity certificate. | the untuned baseline |
| `LC-2.2` | fatal | The parity certificate reports no violation. | a comparison known to be unfair |
| `LC-2.3` | fatal | At least two systems were tuned under the budget. | calling a single-system measurement a comparison |
| `LC-2.4` | warning | Trial counts are equal across systems. | unequal effort presented as equal |
| `LC-2.5` | warning | No system was still improving when its budget ended. | a lower bound reported as an optimum |
| `LC-2.6` | info | The operators of record are named. | an undocumented skill confound |
| `LC-2.7` | fatal | The baseline was tuned, or derives from systems that were. | the untuned baseline, in its most direct form |
| `LC-2.8` | fatal | The declared tuning state of each arm matches its ledger. | a tuned configuration presented as library defaults |

## Envelope, not point

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-3.1` | fatal | At least two factors were swept. | a point estimate supporting a general claim |
| `LC-3.2` | fatal | The envelope meets the minimum cell count. | a surface asserted from too few samples |
| `LC-3.3` | fatal | The headline reports best, worst and median. | a single number as a headline |
| `LC-3.4` | warning | The headline range is non-degenerate. | a point estimate dressed as a range |
| `LC-3.5` | warning | The worst case names the conditions that produce it. | an unlocatable worst case |
| `LC-3.6` | warning | The headline avoids unbounded superlatives. | 'up to 2x faster' |
| `LC-3.7` | fatal | The headline's extremes are consistent with the loss column. | a summary the artifact's own measurements do not support |

## Pre-registration

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-4.1` | fatal | A pre-registration document is present. | post-hoc selection of the favourable comparison |
| `LC-4.2` | fatal | The protocol was sealed. | a protocol written after the data |
| `LC-4.3` | fatal | The normative fields still hash to the seal. | editing the protocol after sealing |
| `LC-4.4` | fatal | The results were verified against the protocol. | a seal nobody checked |
| `LC-4.5` | fatal | No fatal protocol deviation was found. | analysing under different parameters than were registered |
| `LC-4.6` | warning | No deviation from the sealed protocol was declared. | undisclosed drift from the plan |
| `LC-4.7` | warning | The seal carries an external anchor. | ordering resting on the author's own clock |

## One-command reproduction

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-5.1` | fatal | A reproduction command is given. | unfalsifiability by inaccessibility |
| `LC-5.2` | warning | Reproduction is one command, not a pipeline. | a recipe only its author can follow |
| `LC-5.3` | fatal | A commit is pinned. | an unversioned result |
| `LC-5.4` | fatal | The producing source tree was clean. | a pinned commit that does not identify the code that ran |
| `LC-5.5` | warning | A container image is pinned. | an irreproducible software environment |
| `LC-5.6` | fatal | The hardware is documented. | a result that cannot be situated |
| `LC-5.7` | warning | A runtime estimate is given. | a reproducer unable to budget for the attempt |

## Measurement quality

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-Q1` | warning | Replicate count meets the floor. | an underpowered design |
| `LC-Q2` | warning | Measurements were interleaved. | drift confounded with the effect |
| `LC-Q3` | warning | Per-system cell coverage is high. | a sparse envelope described as full |
| `LC-Q4` | warning | Measurement noise is small relative to the claimed effect. | an effect indistinguishable from the noise floor |
| `LC-Q5` | warning | Limitations are stated. | a claim presented without its bounds |
| `LC-Q6` | fatal | Equivalence is established somewhere when the design has the power to. | an inverted or broken equivalence test, which makes every tie look inconclusive |

## Semantic state

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-6.1` | fatal | Feasibility is a state, never a sentinel value. | an out-of-memory configuration encoded as a throughput of zero, which comparison paths then skip |
| `LC-6.2` | fatal | A cell that did not run carries no measurement. | 'did not run' and 'ran and measured zero' collapsing into one number |
| `LC-6.3` | fatal | A completed run served from host memory is not device-feasible. | a run that thrashes for hours being recorded as fitting |

## Model validation

| Rule | Severity | Requirement | Forecloses |
|------|----------|-------------|------------|
| `LC-7.1` | fatal | Calibration and validation data are disjoint and the split is recorded. | a model graded on the data that chose its parameters |
| `LC-7.2` | fatal | No parameter is fitted after the model is frozen. | a validation set quietly becoming a training set |
| `LC-7.3` | fatal | A parameter that failed its quality gate is not used. | a rejected fit promoted into an active model because nobody re-read its provenance |
| `LC-7.4` | warning | A fit over a log-scaled domain reports per-regime error, not only a pooled figure. | one useless regime hiding behind three good ones, and R^2 dominated by the largest points |
