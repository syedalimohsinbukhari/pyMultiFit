"""Created on Nov 14 18:25:10 2024"""
import pandas as pd
from matplotlib import pyplot as plt

from src.pymultifit.physical_models.sersic_d import sph_denhen

# cols = ['radius', 'mass', 'v_density', 'radial_velocity', 'tangential_velocity', 'velocity_z',
#         'radial_v_dispersion', 'tangential_v_dispersion', 'velocity_z_dispersion',
#         'cumulative mass', 'n_particles', 'm/N', 'Lx', 'Ly', 'Lz', 'Ltot', 'mcyl']

cols = ['DISTANCE', 'mass', 'DENSITY', 'v_rad', 'v_tan', 'v_z', 's_rad', 's_tan', 's_z', 'MASS_CUM', 'NUM', 'mass/num',
        'L[0]', 'L[1]', 'L[2]', 'LTOT', 'mass_cyl']

M = 5e8
L = 1

# cols = ['radius', 'N', 'v_density', 'cumMass']

imp = ['DISTANCE', 'DENSITY', 'MASS_CUM']

file_ = pd.read_csv('./fig_file/2000.fig', sep='\s+')
file_.columns = cols

file2_ = file_[imp]

# ax = plot_xy(file2_['radius'], file2_['v_density'], auto_label=True)

# sf = SersicFitter(1, file2_.DISTANCE, file2_.DENSITY, max_iterations=1000)
# p0 = [(1, 1, 1)]
# sf.fit(p0)
# plotter = sf.plot_fit(show_individual=True, auto_label=True, fig_size=(8, 6))
# plotter.xscale('log')
# plotter.yscale('log')
# plotter.tight_layout()
plt.plot(file_.DISTANCE, sph_denhen(file_.DISTANCE, 0.5, 0.2, 0.6), 'g-')
plt.show()
