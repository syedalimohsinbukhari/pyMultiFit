# Comparison: sarl_gpu_ws_1_2eaf7f8 vs sarl_gpu_ws_1_a9ffeec_on_2eaf7f8

- new: sarl-gpu-ws-1 | AMD Ryzen Threadripper PRO 5955WX 16-Cores | commit 2eaf7f8
- reference: sarl-gpu-ws-1 | AMD Ryzen Threadripper PRO 5955WX 16-Cores | commit a9ffeec (benchmark code 2eaf7f8)
- software stacks match

Ratio = multifit time / scipy time (below 1: multifit faster), median over the 12 largest sizes. Verdict when |change| > max(6 %, 3 x scipy drift).

## Summary

- consistently faster: beta PDF (-70 % / -71 %), beta CDF (-62 % / -26 %), arcsine CDF (-67 % / -60 %)
- consistently slower: none
- unproven (only one parameter set, or opposite): gamma PDF (+6 % / +2 %)
(changes listed as default / variable parameters)

## PDF, default parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.42 | 0.43 | +3.4 % | -1.4 % | ~ |
| laplace | 0.44 | 0.45 | +1.4 % | -1.3 % | ~ |
| skewnorm | 0.70 | 0.70 | -0.4 % | -0.4 % | ~ |
| lognorm | 0.49 | 0.49 | +0.4 % | -0.7 % | ~ |
| beta | 2.29 | 0.67 | **-70.5 %** | +0.1 % | **faster** |
| arcsine | 0.33 | 0.33 | +1.3 % | -2.8 % | ~ |
| gamma | 0.40 | 0.43 | **+6.4 %** | -0.8 % | **slower** |
| chi2 | 0.35 | 0.35 | -1.4 % | -0.5 % | ~ |
| foldnorm | 0.65 | 0.66 | +1.6 % | -1.1 % | ~ |
| halfnorm | 0.48 | 0.49 | +1.4 % | -1.5 % | ~ |
| exp | 0.50 | 0.51 | +0.5 % | -1.0 % | ~ |
| unif | 0.19 | 0.20 | +2.9 % | -1.3 % | ~ |

## PDF, variable parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.55 | 0.55 | +0.8 % | -1.2 % | ~ |
| laplace | 0.45 | 0.44 | -0.6 % | -2.1 % | ~ |
| skewnorm | 0.68 | 0.67 | -1.1 % | -0.9 % | ~ |
| lognorm | 0.49 | 0.49 | +0.7 % | -0.5 % | ~ |
| beta | 1.87 | 0.55 | **-70.6 %** | -0.6 % | **faster** |
| arcsine | 0.34 | 0.35 | +3.7 % | -1.3 % | ~ |
| gamma | 0.40 | 0.41 | +2.0 % | -0.1 % | ~ |
| chi2 | 0.39 | 0.38 | -2.1 % | -0.7 % | ~ |
| foldnorm | 0.78 | 0.79 | +2.0 % | -0.2 % | ~ |
| halfnorm | 0.48 | 0.50 | +3.0 % | -1.0 % | ~ |
| exp | 0.49 | 0.50 | +1.5 % | -0.6 % | ~ |
| unif | 0.39 | 0.39 | +0.2 % | +1.6 % | ~ |

## CDF, default parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.67 | 0.65 | -3.2 % | -0.4 % | ~ |
| laplace | 0.35 | 0.35 | -0.4 % | +0.1 % | ~ |
| skewnorm | 0.10 | 0.10 | +0.7 % | +0.9 % | ~ |
| lognorm | 0.72 | 0.71 | -0.5 % | -0.3 % | ~ |
| beta | 1.11 | 0.42 | **-62.0 %** | +0.3 % | **faster** |
| arcsine | 0.83 | 0.27 | **-67.2 %** | -0.1 % | **faster** |
| gamma | 0.88 | 0.88 | -0.4 % | +0.3 % | ~ |
| chi2 | 0.98 | 0.97 | -1.1 % | +0.4 % | ~ |
| foldnorm | 0.80 | 0.80 | +0.3 % | +0.5 % | ~ |
| halfnorm | 0.68 | 0.64 | -5.5 % | +0.2 % | ~ |
| exp | 0.52 | 0.53 | +1.8 % | -0.7 % | ~ |
| unif | 0.20 | 0.20 | +0.2 % | -0.1 % | ~ |

## CDF, variable parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.64 | 0.61 | -4.4 % | +0.1 % | ~ |
| laplace | 0.35 | 0.35 | -0.2 % | +0.2 % | ~ |
| skewnorm | 0.19 | 0.19 | +0.9 % | -0.5 % | ~ |
| lognorm | 0.61 | 0.61 | -0.7 % | -0.5 % | ~ |
| beta | 0.98 | 0.72 | **-26.0 %** | +0.5 % | **faster** |
| arcsine | 0.85 | 0.34 | **-60.3 %** | +0.6 % | **faster** |
| gamma | 0.95 | 0.95 | -0.1 % | +0.2 % | ~ |
| chi2 | 0.89 | 0.89 | +0.3 % | +0.0 % | ~ |
| foldnorm | 0.90 | 0.90 | +0.3 % | +0.4 % | ~ |
| halfnorm | 0.72 | 0.68 | -5.4 % | +0.7 % | ~ |
| exp | 0.54 | 0.53 | -0.4 % | -0.4 % | ~ |
| unif | 0.33 | 0.32 | -2.9 % | +1.4 % | ~ |

