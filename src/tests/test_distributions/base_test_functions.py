"""Created on Feb 02 22:24:06 2025"""

import numpy as np

from ...pymultifit import EPSILON

loc1_parameter = np.random.uniform(low=-100, high=100, size=500)
loc2_parameter = np.random.uniform(low=-100, high=100, size=500)
scale_parameter = np.random.uniform(low=EPSILON, high=100, size=500)
shape1_parameter = np.random.uniform(low=EPSILON, high=100, size=500)
shape2_parameter = np.random.uniform(low=EPSILON, high=100, size=500)


def edge_cases(distribution, log_check=False):
    x = np.array([])

    result = distribution.pdf(x)
    assert result.size == 0
    result = distribution.cdf(x)
    assert result.size == 0

    if log_check:
        result = distribution.logpdf(x)
        assert result.size == 0
        result = distribution.logcdf(x)
        assert result.size == 0


def scaled_distributions(
    custom_distribution,
    scipy_distribution,
    x,
    parameters,
    log_check: bool = False,
    is_expon=False,
    is_scaled_inv_chi=False,
):
    dist = custom_distribution(*parameters)

    # make an exception for exponential distribution which gets scale = 1/scale
    if is_expon:
        parameters[1] = 1 / parameters[1]

    if is_scaled_inv_chi:
        parameters[0] = parameters[0] / 2
        parameters[-1] = parameters[-1] / 2

    np.testing.assert_allclose(actual=dist.pdf(x), desired=scipy_distribution(*parameters).pdf(x), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(actual=dist.cdf(x), desired=scipy_distribution(*parameters).cdf(x), rtol=1e-5, atol=1e-8)
    if log_check:
        np.testing.assert_allclose(
            actual=dist.logpdf(x), desired=scipy_distribution(*parameters).logpdf(x), rtol=1e-5, atol=1e-8
        )
        np.testing.assert_allclose(
            actual=dist.logcdf(x), desired=scipy_distribution(*parameters).logcdf(x), rtol=1e-5, atol=1e-8
        )


def statistics(
    custom_distribution,
    scipy_distribution,
    parameters,
    mean_variance=False,
    median=False,
    is_expon=False,
    is_scaled_inv_chi=False,
):
    d_stats = custom_distribution(*parameters).stats()

    # make an exception for exponential distribution which gets scale = 1/scale
    if is_expon:
        parameters[1] = 1 / parameters[1]

    if is_scaled_inv_chi:
        parameters[0] = parameters[0] / 2
        parameters[-1] = parameters[-1] / 2

    if mean_variance:
        scipy_mean, scipy_variance = scipy_distribution.stats(*parameters, moments="mv")
        scipy_stddev = np.sqrt(scipy_variance)
        np.testing.assert_allclose(actual=scipy_mean, desired=d_stats["mean"], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(actual=scipy_variance, desired=d_stats["variance"], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(actual=scipy_stddev, desired=d_stats["std"], rtol=1e-5, atol=1e-8)

    if median:
        scipy_median = scipy_distribution.median(*parameters)
        np.testing.assert_allclose(actual=scipy_median, desired=d_stats["median"], rtol=1e-5, atol=1e-8)


def stats(
    custom_distribution,
    scipy_distribution,
    parameters,
    equal_case=False,
    equal_params=None,
    mean_variance=True,
    median=True,
    is_expon=False,
    is_scaled_inv_chi=False,
):
    stack_ = np.column_stack(parameters)

    for stack in stack_:
        statistics(
            custom_distribution=custom_distribution,
            scipy_distribution=scipy_distribution,
            parameters=stack,
            mean_variance=mean_variance,
            median=median,
            is_expon=is_expon,
            is_scaled_inv_chi=is_scaled_inv_chi,
        )

    if equal_case:
        statistics(
            custom_distribution=custom_distribution,
            scipy_distribution=scipy_distribution,
            parameters=equal_params,
            mean_variance=mean_variance,
            median=median,
            is_expon=is_expon,
            is_scaled_inv_chi=is_scaled_inv_chi,
        )


def value_functions(
    custom_distribution,
    scipy_distribution,
    parameters,
    n_values=10,
    log_check=False,
    is_expon=False,
    is_scaled_inv_chi=False,
):
    x = np.linspace(start=-100, stop=100, num=n_values)
    x = np.concatenate([x, np.array([0, 1])])
    x.sort()

    stack_ = np.column_stack(parameters)

    for params in stack_:
        scaled_distributions(
            custom_distribution=custom_distribution,
            scipy_distribution=scipy_distribution,
            x=x,
            parameters=params,
            log_check=log_check,
            is_expon=is_expon,
            is_scaled_inv_chi=is_scaled_inv_chi,
        )


def degeneracy(custom_distribution, scipy_distribution, log_check=False):
    x_ = np.linspace(start=-0.5, stop=1.5, num=50)

    extreme_cases = [(1e-5, 1e-5), (1e5, 1e5), (1e-5, 5), (5, 1e-5), (1e-5, 1e5), (1e5, 1e-5)]

    for a_, b_ in extreme_cases:
        dist_ = custom_distribution(a_, b_)

        scipy_pdf = scipy_distribution(a_, b_).pdf(x_)
        scipy_cdf = scipy_distribution(a_, b_).cdf(x_)

        np.testing.assert_allclose(actual=dist_.pdf(x_), desired=scipy_pdf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(actual=dist_.cdf(x_), desired=scipy_cdf, rtol=1e-5, atol=1e-8)

        if log_check:
            scipy_logpdf = scipy_distribution(a_, b_).logpdf(x_)
            scipy_logcdf = scipy_distribution(a_, b_).logcdf(x_)

            np.testing.assert_allclose(actual=dist_.logpdf(x_), desired=scipy_logpdf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=dist_.logcdf(x_), desired=scipy_logcdf, rtol=1e-5, atol=1e-8)


def single_input_n_variables(
    custom_distribution,
    scipy_distribution,
    parameters,
    n_size=50,
    log_check=False,
    is_expon=False,
    is_scaled_inv_chi=False,
):
    rand_ = np.random.uniform(low=0, high=1, size=n_size)

    parameters = np.array(object=parameters, dtype=object)
    param_array = np.column_stack(parameters)

    for value, pars in zip(rand_, param_array):
        p2 = custom_distribution(*pars)

        if is_expon:
            pars[1] = 1 / pars[1]

        if is_scaled_inv_chi:
            pars[0] = pars[0] / 2
            pars[-1] = pars[-1] / 2

        p1 = scipy_distribution(*pars)

        np.testing.assert_allclose(actual=p1.pdf(value), desired=p2.pdf(value), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(actual=p1.cdf(value), desired=p2.cdf(value), rtol=1e-5, atol=1e-8)
        if log_check:
            np.testing.assert_allclose(actual=p1.logpdf(value), desired=p2.logpdf(value), rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(actual=p1.logcdf(value), desired=p2.logcdf(value), rtol=1e-5, atol=1e-8)
