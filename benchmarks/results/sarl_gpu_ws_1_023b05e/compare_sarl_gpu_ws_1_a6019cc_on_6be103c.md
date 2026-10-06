# Comparison: sarl_gpu_ws_1_023b05e vs sarl_gpu_ws_1_a6019cc_on_6be103c

- new: sarl-gpu-ws-1 | AMD Ryzen Threadripper PRO 5955WX 16-Cores | commit 023b05e
- reference: sarl-gpu-ws-1 | AMD Ryzen Threadripper PRO 5955WX 16-Cores | commit a6019cc (benchmark code 6be103c)
- software stacks match

Ratio = multifit time / scipy time (below 1: multifit faster), median over the 12 largest sizes. Verdict when |change| > max(6 %, 3 x scipy drift).

## Summary

- consistently faster: arcsine PDF (-42 % / -38 %), laplace CDF (-41 % / -42 %)
- consistently slower: none
- unproven (only one parameter set, or opposite): gamma PDF (-7 % / -2 %), arcsine CDF (-14 % / -1 %)
(changes listed as default / variable parameters)

## PDF, default parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.44 | 0.43 | -1.5 % | +0.2 % | ~ |
| laplace | 0.45 | 0.44 | -2.3 % | +0.0 % | ~ |
| skewnorm | 0.70 | 0.70 | -0.7 % | +0.2 % | ~ |
| lognorm | 0.47 | 0.49 | +4.2 % | -0.1 % | ~ |
| beta | 2.23 | 2.30 | +3.4 % | +0.2 % | ~ |
| arcsine | 0.58 | 0.34 | **-42.2 %** | +0.8 % | **faster** |
| gamma | 0.43 | 0.40 | **-6.8 %** | +0.3 % | **faster** |
| chi2 | 0.37 | 0.35 | -4.6 % | +0.1 % | ~ |
| foldnorm | 0.67 | 0.68 | +2.0 % | +0.2 % | ~ |
| halfnorm | 0.49 | 0.49 | -0.8 % | -0.1 % | ~ |
| exp | 0.51 | 0.51 | -1.1 % | +0.8 % | ~ |
| unif | 0.20 | 0.20 | -2.7 % | +0.6 % | ~ |

## PDF, variable parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.56 | 0.55 | -1.3 % | +0.0 % | ~ |
| laplace | 0.45 | 0.43 | -4.8 % | +1.0 % | ~ |
| skewnorm | 0.68 | 0.67 | -0.7 % | +0.1 % | ~ |
| lognorm | 0.48 | 0.49 | +2.9 % | +0.3 % | ~ |
| beta | 1.92 | 1.89 | -1.4 % | -0.1 % | ~ |
| arcsine | 0.55 | 0.34 | **-38.4 %** | +2.1 % | **faster** |
| gamma | 0.41 | 0.40 | -2.0 % | +0.4 % | ~ |
| chi2 | 0.39 | 0.39 | -2.1 % | -0.2 % | ~ |
| foldnorm | 0.79 | 0.77 | -2.7 % | -0.0 % | ~ |
| halfnorm | 0.51 | 0.48 | -5.1 % | +0.9 % | ~ |
| exp | 0.50 | 0.48 | -5.6 % | +2.7 % | ~ |
| unif | 0.39 | 0.38 | -2.1 % | +1.2 % | ~ |

## CDF, default parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.64 | 0.63 | -2.3 % | +1.5 % | ~ |
| laplace | 0.58 | 0.34 | **-40.6 %** | -0.3 % | **faster** |
| skewnorm | 0.10 | 0.10 | -0.2 % | +0.0 % | ~ |
| lognorm | 0.70 | 0.70 | +0.1 % | +1.8 % | ~ |
| beta | 1.11 | 1.08 | -2.8 % | +0.2 % | ~ |
| arcsine | 0.97 | 0.84 | **-14.2 %** | -0.1 % | **faster** |
| gamma | 0.88 | 0.87 | -0.7 % | +0.7 % | ~ |
| chi2 | 0.98 | 0.97 | -0.3 % | +0.1 % | ~ |
| foldnorm | 0.81 | 0.80 | -1.3 % | +0.3 % | ~ |
| halfnorm | 0.66 | 0.63 | -4.3 % | +3.9 % | ~ |
| exp | 0.52 | 0.50 | -3.0 % | +3.3 % | ~ |
| unif | 0.21 | 0.20 | -4.2 % | +0.4 % | ~ |

## CDF, variable parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.61 | 0.60 | -0.1 % | +2.9 % | ~ |
| laplace | 0.59 | 0.34 | **-42.2 %** | +0.2 % | **faster** |
| skewnorm | 0.19 | 0.19 | -0.1 % | +0.0 % | ~ |
| lognorm | 0.59 | 0.60 | +1.1 % | +0.3 % | ~ |
| beta | 0.96 | 0.95 | -1.0 % | -0.4 % | ~ |
| arcsine | 0.84 | 0.83 | -1.0 % | -0.7 % | ~ |
| gamma | 0.95 | 0.95 | -0.1 % | +0.2 % | ~ |
| chi2 | 0.89 | 0.89 | -0.0 % | -0.3 % | ~ |
| foldnorm | 0.90 | 0.89 | -0.6 % | +0.2 % | ~ |
| halfnorm | 0.70 | 0.69 | -1.5 % | +0.8 % | ~ |
| exp | 0.54 | 0.53 | -1.3 % | +4.7 % | ~ |
| unif | 0.31 | 0.31 | -1.9 % | +1.5 % | ~ |

