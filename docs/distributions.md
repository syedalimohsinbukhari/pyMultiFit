# `distributions`

The `distributions` module provides functionalities to generate data from various statistical distributions. This
includes:

- [`distributions`](#distributions)
  - [Gaussian distribution](#gaussian-distribution)
  - [Log-normal distribution](#log-normal-distribution)
  - [Skew-normal Distribution](#skew-normal-distribution)
  - [Laplace Distribution](#laplace-distribution)
  - [Gamma Distribution](#gamma-distribution)
  - [Beta Distribution](#beta-distribution)

Each distribution is implemented in two variants:

1. **Normalized distribution**: The distribution includes a normalization constant.
2. **Un-normalized distribution**: The normalization constant is removed, and an `amplitude` parameter is introduced to
   scale the distribution.

## Gaussian distribution

The Gaussian distribution, also known as the normal distribution, is expressed as:

$$
f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

In the un-normalized form, the Gaussian distribution is given by:

$$
f(x; \mu, \sigma) = A\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Here, $A$ represents the amplitude allowing for scaling of the distribution.

## Log-normal distribution

The log-normal distribution is expressed as:

$$
f(x; \mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right)
$$

In the un-normalized form, the log-normal distribution is given by:

$$
f(x; \mu, \sigma) = A\exp\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right)
$$

Here, $A$ represents the amplitude allowing for scaling of the distribution.

## Skew-normal Distribution

The Skew-normal distribution is expressed as:

$$
f(x; \xi, \omega, \alpha) = \frac{2}{\omega} \phi\left(\frac{x - \xi}{\omega}\right) \Phi\left(\alpha \frac{x - \xi}{\omega}\right)
$$

where:
- $\xi$ is the location parameter,
- $\omega$ is the scale parameter,
- $\alpha$ is the shape parameter,
- $\phi(\cdot)$ is the probability density function of the standard normal distribution, and
- $\Phi(\cdot)$ is the cumulative distribution function of the standard normal distribution.

The Skew-normal distribution does not have an unnormalized form in this implementation. It is always normalized by design.

## Laplace Distribution

The Laplace distribution is expressed as:

$$
f(x; \mu, b) = \frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right)
$$

In the un-normalized form, the Laplace distribution is given by:

$$
f(x; \mu, b) = A\exp\left(-\frac{|x-\mu|}{b}\right)
$$

Here, $A$ represents the amplitude, allowing for scaling of the distribution.

## Gamma Distribution

The Gamma distribution is expressed as:

$$
f(x; \alpha, \beta) = \frac{\beta^\alpha x^{\alpha-1} \exp(-\beta x)}{\Gamma(\alpha)}
$$

In the un-normalized form, the Gamma distribution is given by:

$$
f(x; \alpha, \beta) = A x^{\alpha-1} \exp(-\beta x)
$$

Here, $A$ represents the amplitude, allowing for scaling of the distribution.

## Beta Distribution

The Beta distribution is expressed as:

$$
f(x; \alpha, \beta) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)}
$$

In the un-normalized form, the Beta distribution is given by:

$$
f(x; \alpha, \beta) = A x^{\alpha-1} (1-x)^{\beta-1}
$$

Here, $A$ represents the amplitude, allowing for scaling of the distribution.
