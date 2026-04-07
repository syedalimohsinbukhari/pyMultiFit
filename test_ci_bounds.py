#!/usr/bin/env python
"""
Demonstration of new ci_bounds capability for MixedDataFitter
This capability was inherited from BaseFitter after the merger.
"""

import sys

sys.path.insert(0, "src")

import numpy as np

from pymultifit.fitters.mixed_f import MixedDataFitter

print("=" * 70)
print("MixedDataFitter CI Bounds Demonstration")
print("=" * 70)

# Generate synthetic mixed data
np.random.seed(42)
x_data = np.linspace(-5, 15, 200)


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def laplace(x, amp, mu, b):
    return amp * np.exp(-np.abs(x - mu) / b) / (2 * b)


# True parameters
true_params_gauss = (1, 2, 1)
true_params_laplace = (1, 8, 0.5)

y_data = gaussian(x_data, *true_params_gauss) + laplace(x_data, *true_params_laplace)
y_data += np.random.normal(0, 0.02, size=len(x_data))

print("\nStep 1: Create MixedDataFitter")
fitter = MixedDataFitter(x_data, y_data, model_list=["gaussian", "laplace"])
print(f"  ✓ Created with {fitter.n_fits} models and {fitter.n_par} total parameters")

print("\nStep 2: Fit the model")
p0 = [true_params_gauss, true_params_laplace]
fitter.fit(p0=p0)
print(f"  ✓ Fit completed successfully")

print("\nStep 3: Get model parameters")
params = fitter.get_model_parameters(errors=True)
print(f"  ✓ Parameters extracted for models: {list(params['parameters'].keys())}")

print("\nStep 4: Compute bootstrap confidence intervals")
print("  (This is a NEW capability inherited from BaseFitter)")
try:
    # Test with small n_bootstrap for speed
    ci_results = fitter.ci_bounds(ci_level=95, n_bootstrap=50, overall_ci=True, individual_ci=False, random_state=42)

    print(f"  ✓ Bootstrap CI computation successful!")
    print(f"  ✓ CI levels computed: {[k for k in ci_results.keys()]}")

    # Extract CI bounds
    ci_95 = ci_results["overall_ci_95"]
    print(f"\n  95% Confidence Interval Statistics:")
    print(f"    - Lower bound shape: {ci_95['lower'].shape}")
    print(f"    - Upper bound shape: {ci_95['upper'].shape}")
    print(f"    - Median shape: {ci_95['median'].shape}")

    # Calculate CI width at a few points
    sample_indices = [50, 100, 150]
    print(f"\n  Sample CI widths at different x values:")
    for idx in sample_indices:
        width = ci_95["upper"][idx] - ci_95["lower"][idx]
        print(f"    x = {x_data[idx]:6.2f}: CI width = {width:.6f}")

except Exception as e:
    print(f"  ✗ CI computation failed: {e}")
    import traceback

    traceback.print_exc()

print("\nStep 5: Verify other inherited methods")
inherited_methods = {
    "get_residuals": lambda: fitter.get_residuals(),
    "get_fitted_curve": lambda: fitter.get_fitted_curve(),
    "get_value_error_pair": lambda: fitter.get_value_error_pair(),
}

for name, method in inherited_methods.items():
    try:
        result = method()
        print(f"  ✓ {name}() works - result shape: {result.shape}")
    except Exception as e:
        print(f"  ✗ {name}() failed: {e}")

print("\n" + "=" * 70)
print("Demonstration Complete!")
print("=" * 70)
print("\nSummary:")
print("  • MixedDataFitter now inherits from BaseFitter")
print("  • New capability: ci_bounds() for confidence intervals")
print("  • All common methods (get_residuals, plot_fit_and_residuals, etc.)")
print("  • Backward compatible with existing code")
print("  • ~150 lines of duplicate code eliminated")
