# Loading-vector recovery in CP tensor models

This MATLAB package compares two ways to initialize alternating least squares
for canonical polyadic (CP) tensor decomposition. Tensor principal component
analysis (TPCA) and Tucker-based approximation with simultaneous diagonalization
(TASD) each provide starting values for CP-ALS. The simulations use tensors with
known loadings to compare recovery accuracy, computation time, and stability
across datasets and starting values.

## Getting started

Clone or download the repository and open MATLAB in the folder containing
`Tensor_simulations.m`. Keep the bundled `tensor_toolbox-v3.8/` directory beside
the scripts. The simulation adds the toolbox to the MATLAB path and generates
its own data.

Run the full simulation with the following MATLAB command.

```matlab
Tensor_simulations
```

The `config` structure in `Tensor_simulations.m` controls the tensor dimensions,
number of datasets (`config.M`), starts per dataset (`config.nStarts`), experiment
conditions, ALS stopping rule, and recovery tolerance.

The default design has 22 conditions covering noise level, component strength,
coherence, and fitted rank. Each condition uses 100 independent datasets and
three starts per method, for 13,200 fit attempts in total. Both methods fit the
same simulated tensors. On a given dataset, TPCA uses the same starting directions
for each fit, while TASD uses a fresh random draw for each start. Fits run serially
for the timing comparison.

## Evaluation metrics

### Reconstruction performance

The loading error compares the estimated and true vectors at initialization and
after CP-ALS. The comparison normalizes the vectors, aligns their signs, and
matches components using the same assignment across all modes. The assignment
minimizes the largest distance among the matched vectors.

When the fitted and true ranks agree, `meanInitialError` and `meanFinalError`
summarize this error across fits. When the ranks differ, `meanFinalMatchedError`
describes the error among matched components. Read it alongside the recovery
scores, which use matches within the recovery tolerance. `meanFinalRecall`
gives the average share of true components recovered, `meanFinalPrecision`
gives the average share of estimated components that match a true component,
and `meanFinalF1` averages the combined score. `meanReconError` reports the
relative Frobenius error against the clean tensor as a secondary measure of
signal reconstruction.

### Optimization behaviour

`meanTotalSeconds` adds the time spent on initialization and CP-ALS refinement.
`meanALSItersCompleted` averages the number of ALS sweeps over fits that return
an iteration count.
`iterationLimitRate` is the fraction of all attempts that reach the sweep limit.

### Stability and robustness

At the correct rank, `recoveryProbability` is the fraction of fits that recover
every loading direction within `recoveryTol` (0.05 by default). Its standard
error uses the recovery fractions computed separately for each dataset.

`sdDataError` measures how much mean loading error varies across datasets after
averaging over starts within each dataset. `q90FinalError` is the 90th percentile
of the final errors pooled across individual fits. `meanWithinDataDistance`
measures agreement by comparing the final loadings from pairs of starts on the
same dataset and averaging the distances across datasets. This last measure
requires at least two starts per dataset.

Failed fits contribute the maximum applicable loading error (`sqrt(2)`), zero
recovery credit, and an infinite reconstruction error. `failureRate` and
`warningRate` report the fractions of attempts with failures and warnings. The
run records store failure reasons and warning identifiers. Fits with warnings
contribute their measured errors when their estimates are valid.

## Outputs

The full simulation creates `Results_v3/` alongside `Tensor_simulations.m`.

| Output | Contents |
| --- | --- |
| `simulation_results.mat` | The configuration, conditions, true loadings, run records, distances between starts, and summary table. |
| `summary.csv` | One row per method and completed condition. |
| `figures/` | PDF and PNG figures for the completed study. |

The run records in `results.runs` are indexed by dataset, condition, method, and
start. Each record stores loading estimates, errors, seeds, timings, iteration
counts, and numerical status. The MAT and CSV files are updated after each
completed condition, and `results.completedConditions` records progress. Each
invocation starts a fresh simulation and writes to the configured output folder.

## Validation

To check the installation and run a smaller simulation, enter this command in
MATLAB from the folder containing the scripts.

```matlab
verify_loading_recovery
```

The validation runs 88 fits and writes its outputs to `Validation_v3/`. It
checks invariance to scaling, sign, and component order, compares exact matching
with exhaustive search, and checks missing or extra components, invalid
estimates, and reproducible initialization.

## Supporting scripts

- `run_tensor_simulations.m` generates data, fits both methods, and saves summaries.
- `tpca_init_matlab.m` and `tasd_init_matlab.m` compute the starting loadings.
- `loading_metrics.m` computes matched loading errors and component recovery scores.
- `plot_tensor_simulations.m` exports the study figures from the summary table.
- `Tensor_main.m` plots final loading-error distributions across noise and coherence settings, using `factor_eps.m` for equal-rank comparisons.

