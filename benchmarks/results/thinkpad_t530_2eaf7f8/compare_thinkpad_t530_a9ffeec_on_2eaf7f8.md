# Comparison: thinkpad_t530_2eaf7f8 vs thinkpad_t530_a9ffeec_on_2eaf7f8

- new: thinkpad-t530 | Intel(R) Core(TM) i7-3720QM CPU @ 2.60GHz | commit 2eaf7f8
- reference: thinkpad-t530 | Intel(R) Core(TM) i7-3720QM CPU @ 2.60GHz | commit a9ffeec (benchmark code 2eaf7f8)
- software stacks match

Ratio = multifit time / scipy time (below 1: multifit faster), median over the 12 largest sizes. Verdict when |change| > max(6 %, 3 x scipy drift).

## Summary

- consistently faster: beta PDF (-68 % / -69 %), beta CDF (-50 % / -17 %), arcsine CDF (-67 % / -57 %)
- consistently slower: none
- unproven (only one parameter set, or opposite): none
(changes listed as default / variable parameters)

## PDF, default parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.50 | 0.50 | +1.2 % | +0.0 % | ~ |
| laplace | 0.49 | 0.50 | +1.9 % | -1.1 % | ~ |
| skewnorm | 0.72 | 0.74 | +2.5 % | -1.6 % | ~ |
| lognorm | 0.57 | 0.57 | -0.4 % | +0.7 % | ~ |
| beta | 1.88 | 0.60 | **-68.1 %** | -0.2 % | **faster** |
| arcsine | 0.45 | 0.45 | +1.4 % | -0.6 % | ~ |
| gamma | 0.44 | 0.43 | -0.7 % | +0.1 % | ~ |
| chi2 | 0.37 | 0.37 | -1.1 % | -0.4 % | ~ |
| foldnorm | 0.77 | 0.76 | -1.1 % | -0.0 % | ~ |
| halfnorm | 0.51 | 0.52 | +1.7 % | -0.5 % | ~ |
| exp | 0.58 | 0.58 | -0.8 % | -0.0 % | ~ |
| unif | 0.25 | 0.25 | +2.4 % | -0.2 % | ~ |

## PDF, variable parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.58 | 0.56 | -2.2 % | +0.1 % | ~ |
| laplace | 0.53 | 0.52 | -1.7 % | +0.3 % | ~ |
| skewnorm | 0.70 | 0.71 | +2.3 % | -3.0 % | ~ |
| lognorm | 0.59 | 0.60 | +1.5 % | -0.4 % | ~ |
| beta | 1.88 | 0.58 | **-69.3 %** | -0.9 % | **faster** |
| arcsine | 0.50 | 0.50 | +1.5 % | +0.2 % | ~ |
| gamma | 0.44 | 0.45 | +1.0 % | -1.2 % | ~ |
| chi2 | 0.41 | 0.41 | +0.2 % | -0.6 % | ~ |
| foldnorm | 0.88 | 0.89 | +1.1 % | -0.9 % | ~ |
| halfnorm | 0.56 | 0.56 | -1.1 % | +0.4 % | ~ |
| exp | 0.57 | 0.57 | +0.1 % | +0.1 % | ~ |
| unif | 0.41 | 0.41 | -0.2 % | -1.7 % | ~ |

## CDF, default parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.64 | 0.64 | +0.7 % | -0.3 % | ~ |
| laplace | 0.37 | 0.37 | +1.0 % | -0.8 % | ~ |
| skewnorm | 0.19 | 0.19 | +2.0 % | +0.3 % | ~ |
| lognorm | 0.75 | 0.75 | +0.5 % | +0.4 % | ~ |
| beta | 0.96 | 0.48 | **-49.5 %** | -1.0 % | **faster** |
| arcsine | 0.86 | 0.29 | **-66.8 %** | -1.2 % | **faster** |
| gamma | 0.87 | 0.87 | -0.0 % | -0.5 % | ~ |
| chi2 | 0.98 | 0.98 | -0.1 % | -0.0 % | ~ |
| foldnorm | 0.83 | 0.82 | -0.9 % | -0.4 % | ~ |
| halfnorm | 0.65 | 0.66 | +1.6 % | +0.4 % | ~ |
| exp | 0.52 | 0.53 | +1.3 % | -0.8 % | ~ |
| unif | 0.26 | 0.25 | -2.9 % | +1.0 % | ~ |

## CDF, variable parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.60 | 0.61 | +2.6 % | -0.4 % | ~ |
| laplace | 0.38 | 0.38 | +0.9 % | +0.5 % | ~ |
| skewnorm | 0.29 | 0.29 | -0.3 % | +0.0 % | ~ |
| lognorm | 0.67 | 0.66 | -1.1 % | +0.7 % | ~ |
| beta | 0.92 | 0.77 | **-16.7 %** | -0.5 % | **faster** |
| arcsine | 0.87 | 0.38 | **-56.6 %** | +0.7 % | **faster** |
| gamma | 0.95 | 0.96 | +0.5 % | -0.4 % | ~ |
| chi2 | 0.88 | 0.88 | -0.1 % | -0.4 % | ~ |
| foldnorm | 0.95 | 0.95 | -0.3 % | +0.2 % | ~ |
| halfnorm | 0.70 | 0.70 | +0.7 % | +0.4 % | ~ |
| exp | 0.53 | 0.54 | +0.9 % | +0.2 % | ~ |
| unif | 0.37 | 0.37 | +0.7 % | +0.1 % | ~ |

