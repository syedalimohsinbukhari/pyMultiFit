"""Created on Jan 01 11:33:22 2025"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

from pymultifit.fitters import GaussianFitter

# Open the FITS file
file_ = fits.open('./spec-0293-51994-0078.fits')

# Extract data from different extensions
head1 = file_[1]
head2 = file_[2]
head3 = file_[3]

flux = head1.data['flux']
log_lam = head1.data['loglam']
z = head2.data['Z']

# Compute wavelength and rest wavelength
wavelength = 10**log_lam
rest_wavelength = wavelength / (1 + z)

# Plot the spectrum
peak, _ = find_peaks(flux, height=12)

gf = GaussianFitter(rest_wavelength, flux)
guesses = [(15, i, 0.2) for i in rest_wavelength[peak][:-1]] + [(15, rest_wavelength[peak][-1], 1)]
f, ax = plt.subplots(1, 1, figsize=(20, 6))
gf.fit(guesses)
gf.plot_fit(show_individual=True, axis=ax)
# Annotate emission/absorption lines
for i, j, name in zip(head3.data['LINEWAVE'], head3.data['LINEZ'], head3.data['LINENAME']):
    if j > 0:
        rest_line = i
        ax.axvline(rest_line, color='k', alpha=0.25)
        ax.annotate(name.strip(),
                    (rest_line, -5 + np.random.normal(0, 1)),
                    color='red',
                    verticalalignment='bottom')
plt.show()
