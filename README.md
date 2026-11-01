# RAGA-Q reproducible synthetic simulation

This package contains fully synthetic data and executed Python simulation analyses.
It contains no real Java patches, outputs from actual language models, or PIT,
JaCoCo, CodeQL, security-test or human-review measurements. Do not describe these
records as a real software-engineering benchmark or empirical validation.

## Reproduce

Python 3.12 was used. From this folder:

    python -m pip install -r requirements.txt
    python code/raga_simulation.py --output reproduced --repetitions 30

The command generates data, fitted parameters, decisions, statistics, validation
checks and publication figures. A primary run has 720 attempted candidates across
24 fictional projects and 120 fictional tasks. The complete experiment contains
30 seeds in each of three scenarios, totaling 64,800 simulated candidate records.
Proxy_A/B/C are fictional generator identifiers, not actual model families.

The primary seed is 20261028. Repetition seeds are 20261028 through 20261057.
No seed was selected for favorable performance. The primary result can be negative.
Source code is authoritative for the exact data-generating process; config.json
records settings for audit and is not an input configuration parser.

## Files

- data/synthetic_primary.csv: primary aligned scenario, all 720 attempts.
- data/synthetic_all_scenarios.csv: all 90 simulation runs (64,800 rows).
- results/primary_policy_results.csv: held-out policy comparison.
- results/primary_decisions.csv: paired held-out decisions for all policies.
- results/repetition_results.csv: policy summaries for every run.
- results/scenario_summary.csv: descriptive means and standard deviations.
- results/fitted_parameters.json: primary fitted coefficients and routing thresholds.
  A null threshold list denotes explicit fallback to full verification.
- results/primary_inference.json: descriptive paired project-bootstrap intervals.
- results/primary_sensitivity_curve.csv: descriptive held-out tradeoff sweep.
  The test-set sweep is not used for selecting the frozen policy.
- results/validation.json: executed integrity checks and runtime versions.
- results/checksums.json: SHA-256 hashes of the two source data CSVs.
- figures/: graphs as PNG and vector PDF files.
- code/raga_simulation.py: complete generation and analysis implementation.
- DATA_DICTIONARY.csv: column definitions and units.

## Interpretation

Simulation outcomes and checker sensitivities are assumed. Detectors share the
same random uniforms across policies, making higher tiers monotonically more
sensitive by construction. Their effectiveness has not been measured on Java.
All valid patches incur the complete assigned package cost even if a check finds
a failure early. Costs are arbitrary simulated units, not seconds or rupees.
The full comparator omits a routing probe; adaptive fallback still pays its probe.
The no-probe variant omits probe screening and its score feature.
Random allocation permutes noncritical tiers only and preserves critical overrides.
It matches tier counts, not exact total cost, because patch cost multipliers differ.

Calibration minimizes mean synthetic cost among five threshold triples (four
thresholds for the two-tier variant), allowing at most +2 percentage points of
observed accepted-adverse risk and -10 points of coverage versus full checking.
These are exploratory calibration constraints, not a safety guarantee. They
replace the earlier unexecuted protocol's stricter proposed limits for this
explicitly identified simulation. The separate 1-point test margin is descriptive;
no real-world noninferiority claim is made. With only nine test projects, bootstrap
intervals can be unstable. Identical policies yield zero risk-difference intervals
by arithmetic, not evidence that adaptive selection generalizes.

The aligned scenario deliberately makes structural signals informative; the no_signal
scenario removes that influence, and shifted reverses it only on test projects.
Thus signal usefulness is an assumption being explored, not a discovery about AI
code. Fitted coefficients, costs, label prevalence and results are synthetic.
Task categories are balanced simulation tags, not the original proposed real
corpus's 45/30/25/20 composition. Invalid candidates remain in data but are excluded
from the common paired policy cohort and reported as generation yield.

Graphs include negative findings and the primary all-full fallback. Repeated-run
ranges reflect variability under this simulator; they are not intervals for
production assurance. Real task manifests, exact generator snapshots, executed
Java/tool measurements, independent review and a larger project-level study are
still required to address the original empirical reviewer concerns.
