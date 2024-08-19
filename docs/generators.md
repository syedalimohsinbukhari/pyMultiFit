# `generator`

- [`generator`](#generator)
  - [Available Functions](#available-functions)
    - [`generate_multi_gaussian_data`](#generate_multi_gaussian_data)
    - [`generate_multi_skewed_normal_data`](#generate_multi_skewed_normal_data)
    - [`generate_multi_log_normal_data`](#generate_multi_log_normal_data)
    - [`generate_multi_laplace_data`](#generate_multi_laplace_data)


The `generator` module is designed to generate synthetic data using various statistical distributions. It includes functions to create data from multiple Gaussian, Skewed Normal, Log-normal, and Laplace distributions, with options to add noise and normalize the data.

All the generator functions required the following parameters,

- `x (np.ndarray)`: An array of X values where the data will be evaluated.
- `params`: A list of tuples, each containing the parameters for the required distribution in the form. For example, for Gaussian distribution, the parameters can be written as,
```
params = [(5, -4, 2), (10, 2, 2), (3, 4, 1)]
```

This will generate the data with three gaussians with amplitudes 5, 10, and 3 centered at -4, 2, and 4 with standard deviation 2, 2, and 1 respectively.

- `noise_level (float, optional)`: The standard deviation of the Gaussian noise to be added to the generated data. Default is `0.0`.
- `normalize`: If `True`, the generated data is normalized so that the integral of the PDF is less than 1. Default is `False`.

## Available Functions

### `generate_multi_gaussian_data`

Generates synthetic data by combining multiple Gaussian distributions.

### `generate_multi_skewed_normal_data`

Generates synthetic data by combining multiple Skewed Normal distributions.

### `generate_multi_log_normal_data`

Generates synthetic data by combining multiple Log-normal distributions.

### `generate_multi_laplace_data`

Generates synthetic data by combining multiple Laplace distributions.
