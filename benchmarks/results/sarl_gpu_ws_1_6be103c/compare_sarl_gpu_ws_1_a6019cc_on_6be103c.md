# Comparison: sarl_gpu_ws_1_6be103c vs sarl_gpu_ws_1_a6019cc_on_6be103c

- new: sarl-gpu-ws-1 | AMD Ryzen Threadripper PRO 5955WX 16-Cores | commit 6be103c
- reference: sarl-gpu-ws-1 | AMD Ryzen Threadripper PRO 5955WX 16-Cores | commit a6019cc (benchmark code 6be103c)
- software stacks match

Ratio = multifit time / scipy time (below 1: multifit faster), median over the 12 largest sizes. Verdict when |change| > max(6 %, 3 x scipy drift).

## Summary

- consistently faster: arcsine PDF (-43 % / -40 %), laplace CDF (-40 % / -41 %)
- consistently slower: lognorm CDF (+6 % / +10 %)
- unproven (only one parameter set, or opposite): gamma PDF (-7 % / -1 %), chi2 PDF (+5 % / +8 %), arcsine CDF (-14 % / -0 %)
(changes listed as default / variable parameters)

## PDF, default parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.44 | 0.42 | -4.5 % | +1.0 % | ~ |
| laplace | 0.45 | 0.44 | -2.0 % | +0.8 % | ~ |
| skewnorm | 0.70 | 0.69 | -1.4 % | +0.8 % | ~ |
| lognorm | 0.47 | 0.50 | +4.5 % | +0.2 % | ~ |
| beta | 2.23 | 2.27 | +1.7 % | +1.6 % | ~ |
| arcsine | 0.58 | 0.33 | **-43.0 %** | +1.8 % | **faster** |
| gamma | 0.43 | 0.40 | **-6.9 %** | +1.2 % | **faster** |
| chi2 | 0.37 | 0.39 | +5.4 % | +0.7 % | ~ |
| foldnorm | 0.67 | 0.68 | +1.8 % | +0.7 % | ~ |
| halfnorm | 0.49 | 0.49 | -1.3 % | +0.6 % | ~ |
| exp | 0.51 | 0.51 | -1.3 % | +1.0 % | ~ |
| unif | 0.20 | 0.19 | -6.0 % | +1.6 % | ~ |

## PDF, variable parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.56 | 0.55 | -1.7 % | +0.6 % | ~ |
| laplace | 0.45 | 0.44 | -2.8 % | +2.8 % | ~ |
| skewnorm | 0.68 | 0.68 | -0.2 % | +1.0 % | ~ |
| lognorm | 0.48 | 0.49 | +2.6 % | +0.5 % | ~ |
| beta | 1.92 | 1.89 | -1.5 % | +0.3 % | ~ |
| arcsine | 0.55 | 0.33 | **-39.9 %** | +2.7 % | **faster** |
| gamma | 0.41 | 0.40 | -1.3 % | +0.9 % | ~ |
| chi2 | 0.39 | 0.43 | **+8.5 %** | +0.2 % | **slower** |
| foldnorm | 0.79 | 0.77 | -1.7 % | +0.6 % | ~ |
| halfnorm | 0.51 | 0.50 | -1.0 % | +0.4 % | ~ |
| exp | 0.50 | 0.50 | -1.1 % | +0.8 % | ~ |
| unif | 0.39 | 0.38 | -1.6 % | +1.1 % | ~ |

## CDF, default parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.64 | 0.65 | +0.9 % | +0.4 % | ~ |
| laplace | 0.58 | 0.35 | **-39.7 %** | -0.5 % | **faster** |
| skewnorm | 0.10 | 0.10 | -0.2 % | +0.2 % | ~ |
| lognorm | 0.70 | 0.74 | **+6.5 %** | -1.0 % | **slower** |
| beta | 1.11 | 1.09 | -1.8 % | -0.3 % | ~ |
| arcsine | 0.97 | 0.84 | **-13.5 %** | -0.3 % | **faster** |
| gamma | 0.88 | 0.88 | +0.0 % | -0.1 % | ~ |
| chi2 | 0.98 | 0.97 | -0.2 % | +0.1 % | ~ |
| foldnorm | 0.81 | 0.80 | -0.6 % | -0.1 % | ~ |
| halfnorm | 0.66 | 0.66 | +0.3 % | -0.7 % | ~ |
| exp | 0.52 | 0.55 | +5.9 % | -0.4 % | ~ |
| unif | 0.21 | 0.20 | -2.8 % | -0.8 % | ~ |

## CDF, variable parameters

| distribution | ref ratio | new ratio | change | scipy drift | verdict |
|---|---:|---:|---:|---:|---|
| norm | 0.61 | 0.62 | +3.0 % | +0.0 % | ~ |
| laplace | 0.59 | 0.35 | **-40.6 %** | +0.7 % | **faster** |
| skewnorm | 0.19 | 0.19 | +0.0 % | +0.0 % | ~ |
| lognorm | 0.59 | 0.65 | **+10.0 %** | -0.7 % | **slower** |
| beta | 0.96 | 0.96 | -0.5 % | -0.9 % | ~ |
| arcsine | 0.84 | 0.84 | -0.1 % | -1.5 % | ~ |
| gamma | 0.95 | 0.95 | +0.1 % | +0.1 % | ~ |
| chi2 | 0.89 | 0.89 | +0.0 % | -0.2 % | ~ |
| foldnorm | 0.90 | 0.90 | +0.4 % | -0.9 % | ~ |
| halfnorm | 0.70 | 0.71 | +1.7 % | -0.8 % | ~ |
| exp | 0.54 | 0.53 | -0.3 % | +2.9 % | ~ |
| unif | 0.31 | 0.32 | +2.0 % | +0.3 % | ~ |

