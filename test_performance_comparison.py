"""Performance comparison: Sequential vs Parallel bootstrap CI."""

import time
import numpy as np
from pymultifit.fitters import GaussianFitter
from pymultifit.generators import multi_gaussian

print("="*70)
print("Bootstrap CI Performance Test - Full Example from rough_ci.py")
print("="*70)

# Same parameters as rough_ci.py
params = [(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]
x = np.linspace(-35, 35, 1500)
noise_level = 0.2
y = multi_gaussian(x, params=params, noise_level=noise_level)

fitter = GaussianFitter(x_values=x, y_values=y)
guess = [(10, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

print("\nFitting model...")
fitter.fit(guess)
print("Fit complete!")

print("\n" + "-"*70)
print("Test 1: Sequential Mode (n_jobs=1) - 200 bootstrap samples")
print("-"*70)
start = time.time()
ci_seq = fitter.ci_bounds(
    ci_level=95,
    n_bootstrap=200,  # Smaller for testing
    overall_ci=True,
    n_jobs=1,
    verbose=True,
    random_state=42
)
time_seq = time.time() - start
print(f"Sequential time: {time_seq:.2f} seconds")

print("\n" + "-"*70)
print("Test 2: Parallel Mode (n_jobs=-1) - 200 bootstrap samples")
print("-"*70)
start = time.time()
ci_par = fitter.ci_bounds(
    ci_level=95,
    n_bootstrap=200,  # Same number
    overall_ci=True,
    n_jobs=-1,
    verbose=True,
    random_state=42
)
time_par = time.time() - start
print(f"Parallel time: {time_par:.2f} seconds")

# Calculate speedup
speedup = time_seq / time_par

print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)
print(f"Dataset: {len(x)} points, {len(params)} Gaussian components")
print(f"Bootstrap samples: 200")
print(f"")
print(f"Sequential time:  {time_seq:.2f} seconds")
print(f"Parallel time:    {time_par:.2f} seconds")
print(f"Speedup:          {speedup:.2f}x")
print(f"Time saved:       {time_seq - time_par:.2f} seconds")
print("="*70)

# Extrapolate to 1000 samples
print(f"\nExtrapolated time for 1000 bootstrap samples:")
print(f"  Sequential: ~{time_seq * 5:.1f} seconds ({time_seq * 5 / 60:.1f} minutes)")
print(f"  Parallel:   ~{time_par * 5:.1f} seconds ({time_par * 5 / 60:.1f} minutes)")
print(f"  Time saved: ~{(time_seq - time_par) * 5:.1f} seconds ({(time_seq - time_par) * 5 / 60:.1f} minutes)")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("✓ Always use n_jobs=-1 (parallel) for production")
print("✓ Use n_jobs=1 (sequential) only for debugging")
print(f"✓ Your system achieves ~{speedup:.1f}x speedup with parallel processing")
print("="*70)
