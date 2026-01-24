#!/usr/bin/env python
"""Test script to verify MixedDataFitter inheritance from BaseFitter"""

import sys
sys.path.insert(0, 'src')

from pymultifit.fitters.mixed_f import MixedDataFitter
from pymultifit.fitters.backend.baseFitter import BaseFitter
import numpy as np

# Test 1: Basic instantiation
print("=" * 60)
print("Test 1: Basic Instantiation")
print("=" * 60)
x = np.linspace(0, 10, 100)
y = np.random.rand(100)
fitter = MixedDataFitter(x, y, model_list=['gaussian', 'laplace'])

print(f"✓ MixedDataFitter created successfully")
print(f"  n_par = {fitter.n_par} (expected: 6, gaussian=3 + laplace=3)")
print(f"  n_fits = {fitter.n_fits} (expected: 2)")
print(f"  isinstance(BaseFitter) = {isinstance(fitter, BaseFitter)}")

# Test 2: Inherited methods
print("\n" + "=" * 60)
print("Test 2: Inherited Methods")
print("=" * 60)
inherited_methods = [
    'ci_bounds',
    'plot_fit_and_residuals',
    'get_residuals',
    'get_fitted_curve',
    'get_value_error_pair',
    '_params',
    '_standard_errors',
    'dry_run'
]

for method in inherited_methods:
    has_it = hasattr(fitter, method)
    print(f"  {method}: {'✓' if has_it else '✗'}")

# Test 3: Overridden methods
print("\n" + "=" * 60)
print("Test 3: Overridden Methods")
print("=" * 60)
overridden_methods = [
    '_n_fitter',
    '_plot_individual_fitter',
    'fit',
    'ci_bounds',
    'plot_fit'
]

for method in overridden_methods:
    has_it = hasattr(fitter, method)
    print(f"  {method}: {'✓' if has_it else '✗'}")

# Test 4: Fit with frozen parameters
print("\n" + "=" * 60)
print("Test 4: Fit with Frozen Parameters")
print("=" * 60)

# Generate some test data
np.random.seed(42)
x_data = np.linspace(-5, 15, 200)

# Generate mixed data using simple functions
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def laplace(x, amp, mu, b):
    return amp * np.exp(-np.abs(x - mu) / b) / (2 * b)

y_data = gaussian(x_data, 1, 2, 1) + laplace(x_data, 1, 8, 0.5)
y_data += np.random.normal(0, 0.02, size=len(x_data))

fitter2 = MixedDataFitter(x_data, y_data, model_list=['gaussian', 'laplace'])

# Initial guess
p0 = [(1, 2, 1), (1, 8, 0.5)]

try:
    # Fit with frozen parameter (e.g., freeze the 3rd parameter)
    fitter2.fit(p0=p0, frozen=[3])
    print(f"✓ Fit with frozen parameter successful")
    print(f"  Fitted parameters shape: {fitter2.params.shape}")
    print(f"  Has covariance: {fitter2.covariance is not None}")
except Exception as e:
    print(f"✗ Fit failed: {e}")

# Test 5: Get model parameters
print("\n" + "=" * 60)
print("Test 5: Get Model Parameters")
print("=" * 60)

try:
    params = fitter2.get_model_parameters()
    print(f"✓ get_model_parameters() successful")
    print(f"  Keys: {list(params.keys())}")

    params_with_errors = fitter2.get_model_parameters(errors=True)
    print(f"✓ get_model_parameters(errors=True) successful")
    print(f"  Keys: {list(params_with_errors.keys())}")
except Exception as e:
    print(f"✗ get_model_parameters failed: {e}")

# Test 6: Get residuals
print("\n" + "=" * 60)
print("Test 6: Get Residuals")
print("=" * 60)

try:
    residuals = fitter2.get_residuals()
    print(f"✓ get_residuals() successful")
    print(f"  Residuals shape: {residuals.shape}")
    print(f"  Mean residual: {np.mean(residuals):.6f}")
except Exception as e:
    print(f"✗ get_residuals failed: {e}")

# Test 7: Get fitted curve
print("\n" + "=" * 60)
print("Test 7: Get Fitted Curve")
print("=" * 60)

try:
    fitted = fitter2.get_fitted_curve()
    print(f"✓ get_fitted_curve() successful")
    print(f"  Fitted curve shape: {fitted.shape}")
except Exception as e:
    print(f"✗ get_fitted_curve failed: {e}")

print("\n" + "=" * 60)
print("All Tests Completed!")
print("=" * 60)
