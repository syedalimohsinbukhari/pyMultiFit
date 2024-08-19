# `fitter` Module

The `fitter` module provides a framework for fitting data using various statistical distributions. The core of this module is the `multiFitter` base class, which is inherited by specific fitters tailored to different distributions, which include,

- [`fitter` Module](#fitter-module)
  - [Class Structure](#class-structure)
    - [`multiFitter`](#multifitter)
    - [Inherited Fitters](#inherited-fitters)
  - [Functionalities](#functionalities)




## Class Structure

### `multiFitter`
The `multiFitter` class serves as the base class for all the fitters. It provides the foundational methods and structure required for fitting data with multiple models.

### Inherited Fitters
The following classes inherit from `multiFitter`, each designed to fit data using a specific distribution:

1. **`GaussianFitter`**: Fits data using Gaussian (normal) distributions.
2. **`LogNormalFitter`**: Fits data using Log-normal distributions.
3. **`SkewNormalFitter`**: Fits data using Skew-normal distributions.
4. **`LaplaceFitter`**: Fits data using Laplace distributions.

## Functionalities

All the fitters provide the following core functionalities:

1. **`fit`**: 
   - Fits the data to the specified distribution(s).
   - Requires an initial guess `p0` for the parameters.
   - `p0` should have a length equal to the number of models multiplied by the number of parameters of the distribution.
   - For example, if fitting two Gaussian distributions (which have three parameters each: mean, standard deviation, and amplitude), `p0` should have a length of 6.

2. **`parameter_extraction`**:
   - Extracts the parameters from the fitted data.
   - Allows for easy access to the parameters that define the best-fit model(s).

3. **`get_fit_values`**:
   - Returns the fitted values.
   - Useful for comparing the fit to the actual data.

4. **`get_value_error_pair`**:
   - Returns the value/error pair for the fitted values.
   - Provides an estimate of the uncertainties associated with the fit.

5. **`plot_fit`**:
   - Plots the fitted distribution(s) over the data.
   - Helps in visualizing how well the model(s) fit the data.