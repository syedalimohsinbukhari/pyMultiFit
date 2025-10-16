"""Created on Aug 21 14:29:30 2025"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit.distributions.utilities_d import gaussian_pdf_, skew_normal_pdf_, line
from pymultifit.fitters import MixedDataFitter, GaussianFitter, SkewNormalFitter, LineFitter
from pymultifit.generators import multiple_models

x = np.linspace(-10, 10, 10_000)

gauss = (10, -1, 0.2)
skewNorm = (3, 5, 0.2, 3)
lineParams = (-0.2, -0.3)

params = [gauss, skewNorm, lineParams]

mg_data = multiple_models(x, params, model_list=['gaussian', 'skew_normal', 'line'],
                          mapping_dict={'gaussian': gaussian_pdf_,
                                        'skew_normal': skew_normal_pdf_,
                                        'line': line}, noise_level=0.2)

guess = [(8, 0, 1), (8, 3, 0, 2), (1, 0)]

mg_fitter = MixedDataFitter(x, mg_data, model_list=['gaussian', 'skew_normal', 'line'],
                            model_dictionary={'gaussian': GaussianFitter,
                                              'skew_normal': SkewNormalFitter,
                                              'line': LineFitter})
mg_fitter.fit(guess)
mg_fitter.plot_fit(show_individuals=True)
plt.savefig('./mixed_fit_paper.png', dpi=300)
plt.show()
