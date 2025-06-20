"""Created on Jun 20 14:39:03 2025"""

from typing import overload, Union

from numpy.typing import NDArray

@overload
def arc_sine_pdf_(
    x,
    amplitude: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def arc_sine_pdf_(
    x,
    amplitude: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def arc_sine_log_pdf_(
    x,
    amplitude: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def arc_sine_log_pdf_(
    x,
    amplitude: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def arc_sine_cdf_(
    x,
    amplitude: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def arc_sine_cdf_(
    x,
    amplitude: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def arc_sine_log_cdf_(
    x,
    amplitude: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def arc_sine_log_cdf_(
    x,
    amplitude: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def beta_pdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def beta_pdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def beta_log_pdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def beta_log_pdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def beta_cdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def beta_cdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def beta_log_cdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def beta_log_cdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def chi_square_pdf_(
    x,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def chi_square_pdf_(
    x,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def chi_square_log_pdf_(
    x,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def chi_square_log_pdf_(
    x,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def chi_square_cdf_(
    x,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def chi_square_cdf_(
    x,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def chi_square_log_cdf_(
    x,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def chi_square_log_cdf_(
    x,
    amplitude: float = 1.0,
    degree_of_freedom: Union[int, float] = 1,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def exponential_pdf_(
    x,
    amplitude: float = 1.0,
    lambda_: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def exponential_pdf_(
    x,
    amplitude: float = 1.0,
    lambda_: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def exponential_log_pdf_(
    x,
    amplitude: float = 1.0,
    lambda_: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def exponential_log_pdf_(
    x,
    amplitude: float = 1.0,
    lambda_: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def exponential_cdf_(
    x,
    amplitude: float = 1.0,
    lambda_: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def exponential_cdf_(
    x,
    amplitude: float = 1.0,
    lambda_: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def exponential_log_cdf_(
    x,
    amplitude: float = 1.0,
    lambda_: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def exponential_log_cdf_(
    x,
    amplitude: float = 1.0,
    lambda_: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def folded_normal_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def folded_normal_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def folded_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def folded_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def folded_normal_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def folded_normal_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def folded_normal_log_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def folded_normal_log_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def gamma_pdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def gamma_pdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def gamma_log_pdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def gamma_log_pdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def gamma_cdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
@overload
def gamma_cdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def gamma_log_cdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def gamma_log_cdf_(
    x,
    amplitude: float = 1.0,
    alpha: float = 1.0,
    theta: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def gaussian_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def gaussian_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
) -> NDArray: ...
@overload
def gaussian_log_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def gaussian_log_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
) -> NDArray: ...
@overload
def gaussian_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def gaussian_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
) -> NDArray: ...
@overload
def gaussian_log_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def gaussian_log_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
) -> NDArray: ...
@overload
def half_normal_pdf_(
    x,
    amplitude: float = 1.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def half_normal_pdf_(
    x,
    amplitude: float = 1.0,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def half_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def half_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def half_normal_cdf_(
    x,
    amplitude: float = 1.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def half_normal_cdf_(
    x,
    amplitude: float = 1.0,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def half_normal_log_cdf_(
    x,
    amplitude: float = 1.0,
    sigma: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def half_normal_log_cdf_(
    x,
    amplitude: float = 1.0,
    sigma: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def laplace_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    diversity: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def laplace_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    diversity: float = 1.0,
) -> NDArray: ...
@overload
def laplace_log_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    diversity: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def laplace_log_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    diversity: float = 1.0,
) -> NDArray: ...
@overload
def laplace_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    diversity: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def laplace_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    diversity: float = 1.0,
) -> NDArray: ...
@overload
def laplace_log_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    diversity: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def laplace_log_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 0.0,
    diversity: float = 1.0,
) -> NDArray: ...
@overload
def log_normal_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 1.0,
    std: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def log_normal_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 1.0,
    std: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def log_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 1.0,
    std: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def log_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 1.0,
    std: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def log_normal_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 1.0,
    std=1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def log_normal_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 1.0,
    std=1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def log_normal_log_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 1.0,
    std: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def log_normal_log_cdf_(
    x,
    amplitude: float = 1.0,
    mean: float = 1.0,
    std: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def uniform_pdf_(
    x,
    amplitude: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def uniform_pdf_(
    x,
    amplitude: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
) -> NDArray: ...
@overload
def uniform_log_pdf_(
    x,
    amplitude: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def uniform_log_pdf_(
    x,
    amplitude: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
) -> NDArray: ...
@overload
def uniform_cdf_(
    x,
    amplitude: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def uniform_cdf_(
    x,
    amplitude: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
) -> NDArray: ...
@overload
def uniform_log_cdf_(
    x,
    amplitude: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def uniform_log_cdf_(
    x,
    amplitude: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
) -> NDArray: ...
@overload
def scaled_inv_chi_square_pdf_(
    x,
    amplitude: float = 1.0,
    df: float = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def scaled_inv_chi_square_pdf_(
    x,
    amplitude: float = 1.0,
    df: float = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def scaled_inv_chi_square_log_pdf_(
    x,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def scaled_inv_chi_square_log_pdf_(
    x,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def scaled_inv_chi_square_cdf_(
    x,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def scaled_inv_chi_square_cdf_(
    x,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def scaled_inv_chi_square_log_cdf_(
    x,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def scaled_inv_chi_square_log_cdf_(
    x,
    amplitude: float = 1.0,
    df: Union[int, float] = 1.0,
    scale: float = 1.0,
    loc: float = 0.0,
) -> NDArray: ...
@overload
def skew_normal_pdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def skew_normal_pdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def skew_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def skew_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def skew_normal_cdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def skew_normal_cdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def sym_gen_normal_pdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def sym_gen_normal_pdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def sym_gen_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def sym_gen_normal_log_pdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def sym_gen_normal_cdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def sym_gen_normal_cdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
@overload
def sym_gen_normal_log_cdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    normalize: bool = False,
) -> NDArray: ...
@overload
def sym_gen_normal_log_cdf_(
    x,
    amplitude: float = 1.0,
    shape: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray: ...
