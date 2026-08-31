# Validator coverage matrix -- LC-1.0

Registry digest `sha256:c167b1a695a81a4cb388142f277c42cdc5667ac31661c2ff92947c6a36ab5684` &mdash; 54 rules, 34 fatal.

**54/54** rules have both a passing and a failing case. The corpus contains 53 deliberately defective artifacts.

A rule with no failing case is a rule nobody has shown can fire. Where that is structural, the reason is given.

| Rule | Requirement | Severity | Pass | Fail | Note |
|------|-------------|----------|------|------|------|
| `LC-0.1` | Standard version | fatal | yes | yes |  |
| `LC-0.2` | Evidence class | fatal | yes | yes |  |
| `LC-0.3` | Evidence class | fatal | yes | yes |  |
| `LC-1.1` | Loss column | fatal | yes | yes |  |
| `LC-1.2` | Loss column | fatal | yes | yes |  |
| `LC-1.3` | Loss column | fatal | yes | yes |  |
| `LC-1.4` | Loss column | fatal | yes | yes |  |
| `LC-1.5` | Loss column | warning | yes | yes |  |
| `LC-1.6` | Loss column | warning | yes | yes |  |
| `LC-1.7` | Loss column | warning | yes | yes |  |
| `LC-1.8` | Loss column | fatal | yes | yes |  |
| `LC-1.9` | Loss column | fatal | yes | yes |  |
| `LC-2.1` | Tuning-budget parity | fatal | yes | yes |  |
| `LC-2.2` | Tuning-budget parity | fatal | yes | yes |  |
| `LC-2.3` | Tuning-budget parity | fatal | yes | yes |  |
| `LC-2.4` | Tuning-budget parity | warning | yes | yes |  |
| `LC-2.5` | Tuning-budget parity | warning | yes | yes |  |
| `LC-2.6` | Tuning-budget parity | info | yes | yes | informational: names the operators of record and never fails |
| `LC-2.7` | Tuning-budget parity | fatal | yes | yes |  |
| `LC-2.8` | Tuning-budget parity | fatal | yes | yes |  |
| `LC-3.1` | Envelope, not point | fatal | yes | yes |  |
| `LC-3.2` | Envelope, not point | fatal | yes | yes |  |
| `LC-3.3` | Envelope, not point | fatal | yes | yes |  |
| `LC-3.4` | Envelope, not point | warning | yes | yes |  |
| `LC-3.5` | Envelope, not point | warning | yes | yes |  |
| `LC-3.6` | Envelope, not point | warning | yes | yes |  |
| `LC-3.7` | Envelope, not point | fatal | yes | yes |  |
| `LC-4.1` | Pre-registration | fatal | yes | yes |  |
| `LC-4.2` | Pre-registration | fatal | yes | yes |  |
| `LC-4.3` | Pre-registration | fatal | yes | yes |  |
| `LC-4.4` | Pre-registration | fatal | yes | yes |  |
| `LC-4.5` | Pre-registration | fatal | yes | yes |  |
| `LC-4.6` | Pre-registration | warning | yes | yes |  |
| `LC-4.7` | Pre-registration | warning | yes | yes |  |
| `LC-5.1` | One-command reproduction | fatal | yes | yes |  |
| `LC-5.2` | One-command reproduction | warning | yes | yes |  |
| `LC-5.3` | One-command reproduction | fatal | yes | yes |  |
| `LC-5.4` | One-command reproduction | fatal | yes | yes |  |
| `LC-5.5` | One-command reproduction | warning | yes | yes |  |
| `LC-5.6` | One-command reproduction | fatal | yes | yes |  |
| `LC-5.7` | One-command reproduction | warning | yes | yes |  |
| `LC-6.1` | Semantic state | fatal | yes | yes |  |
| `LC-6.2` | Semantic state | fatal | yes | yes |  |
| `LC-6.3` | Semantic state | fatal | yes | yes |  |
| `LC-7.1` | Model validation | fatal | yes | yes |  |
| `LC-7.2` | Model validation | fatal | yes | yes |  |
| `LC-7.3` | Model validation | fatal | yes | yes |  |
| `LC-7.4` | Model validation | warning | yes | yes |  |
| `LC-Q1` | Measurement quality | warning | yes | yes |  |
| `LC-Q2` | Measurement quality | warning | yes | yes |  |
| `LC-Q3` | Measurement quality | warning | yes | yes |  |
| `LC-Q4` | Measurement quality | warning | yes | yes |  |
| `LC-Q5` | Measurement quality | warning | yes | yes |  |
| `LC-Q6` | Measurement quality | fatal | yes | yes |  |

## Defect corpus

| Defect | Expected rule(s) | Description |
|--------|------------------|-------------|
| `insufficient-replicates` | `LC-1.8`, `LC-Q1` | Three replicates on a 24-cell grid: the sign-flip floor of 2^-3 = 0.125 cannot clear the Benjamini-Hochberg threshold of 0.05/24, so no cell could ever be declared a loss. |
| `bh-impossible-power` | `LC-1.8` | Eleven replicates, but 100000 simultaneous tests: q/m falls below the sign-flip floor, so the grid is too wide for the replicate count. |
| `inverted-tost-bounds` | `LC-Q6` | A powered 24-cell sweep in which equivalence is never once established and most cells are inconclusive: the signature of two-one-sided tests whose alternatives point outward instead of inward. |
| `mde-differs-from-seal` | `LC-4.5` | The analysis used a 5% minimum effect while the sealed protocol registered 10%, undeclared. |
| `simulated-unlabeled` | `LC-0.2` | A modelled result presented without saying so. |
| `simulated-unexplained` | `LC-0.3` | Labelled simulated, but never says what was modelled and what measured. |
| `tuned-baseline-called-default` | `LC-2.8` | The baseline is described as running at library defaults while its ledger records forty tuning trials. |
| `unsupported-headline` | `LC-3.7` | The headline reports a worst case of -4% while the loss column contains a region at -24%. |
| `dirty-tree` | `LC-5.4` | Produced from a modified working tree, so the pinned commit does not identify the code that ran. |
| `unsupported-loss-region` | `LC-1.9` | A region asserted as a loss whose q-value exceeds the threshold the claim itself declares, and whose worst case is milder than its own median. |
| `feasibility-sentinel` | `LC-6.1` | An out-of-memory configuration encoded as a throughput of zero rather than as a state. Comparison paths skip zero-valued cells, so the prediction disappears from the map. |
| `unrun-cell-carries-a-number` | `LC-6.2` | A cell recorded as infeasible that nonetheless carries a throughput, collapsing 'did not run' into 'ran and measured zero'. |
| `host-fallback-counted-as-fit` | `LC-6.3` | A run served from host memory counted as device-feasible. It completed; it did not fit. |
| `calibration-validation-leakage` | `LC-7.1` | A cell used to choose a parameter also offered as evidence that the parameter was right. |
| `never-frozen-model` | `LC-7.2` | Parameters fitted with no freeze recorded, so nothing distinguishes the validation data from the training data. |
| `rejected-parameter-used` | `LC-7.3` | A parameter that failed its quality gate marked usable, so a rejected fit feeds an active model. |
| `pooled-error-over-log-domain` | `LC-7.4` | A fit spanning four orders of magnitude in message size, reporting only a pooled error. One useless regime hides behind three good ones. |
| `wrong-standard-version` | `LC-0.1` | Declares a revision the validator does not implement. |
| `no-loss-column` | `LC-1.1` | Ships no loss column at all. |
| `losses-without-regions` | `LC-1.2` | Reports losing cells but describes no region. |
| `self-declared-incomplete` | `LC-1.3` | The column says it is incomplete. |
| `loss-column-in-appendix` | `LC-1.4` | The column is relegated to an appendix. |
| `unattributed-region` | `LC-1.5` | A region with no attributed cause. |
| `unaccounted-missing-cells` | `LC-1.6` | Unmeasurable cells that nothing explains. |
| `mostly-inconclusive` | `LC-1.7` | Over a third of the envelope is undetermined. |
| `no-parity-certificate` | `LC-2.1` | A tuned comparison with no parity certificate. |
| `parity-violated` | `LC-2.2` | The certificate reports a violation. |
| `one-system-only` | `LC-2.3` | A single ledger presented as a comparison. |
| `unequal-trial-counts` | `LC-2.4` | One engine received twice the budget. |
| `budget-was-binding` | `LC-2.5` | An engine was still improving when its budget ended. |
| `untuned-baseline` | `LC-2.7` | The baseline has no ledger and declares no derivation. |
| `single-factor` | `LC-3.1` | One factor swept. |
| `too-few-cells` | `LC-3.2` | Below the minimum cell count. |
| `headline-without-range` | `LC-3.3` | No best/worst/median. |
| `degenerate-range` | `LC-3.4` | Best equals worst: a point estimate dressed as a range. |
| `worst-without-conditions` | `LC-3.5` | A worst case nobody can locate. |
| `unbounded-superlative` | `LC-3.6` | 'up to 2x faster'. |
| `no-preregistration` | `LC-4.1` | No protocol document. |
| `unsealed-protocol` | `LC-4.2` | A protocol that was never sealed. |
| `protocol-edited-after-sealing` | `LC-4.3` | Normative fields no longer hash to the seal. |
| `unverified-protocol` | `LC-4.4` | Nobody checked the results against the seal. |
| `declared-deviation` | `LC-4.6` | A declared departure from the sealed protocol. |
| `unanchored-seal` | `LC-4.7` | The seal rests on the author's own clock. |
| `no-reproduction-command` | `LC-5.1` | No command. |
| `pipeline-not-command` | `LC-5.2` | A shell pipeline rather than an entry point. |
| `unpinned-commit` | `LC-5.3` | No commit pinned. |
| `unpinned-image` | `LC-5.5` | No container image. |
| `undocumented-hardware` | `LC-5.6` | Hardware not stated. |
| `no-runtime-estimate` | `LC-5.7` | A reproducer cannot budget for the attempt. |
| `blocked-measurement` | `LC-Q2` | Systems measured in blocks, not interleaved. |
| `sparse-coverage` | `LC-Q3` | A third of the cells were never measured. |
| `noise-swamps-effect` | `LC-Q4` | Within-cell noise comparable to the headline effect. |
| `no-limitations` | `LC-Q5` | A claim with no stated bounds. |