import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import savefig, plot_data, set_axes
from model import ProtonModel

def plot_spectrum():
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    ylabel = r'E$^{2.7}$ I [GeV$^{1.7}$ m$^{-2}$ s$^{-1}$ sr$^{-1}$]'
    set_axes(ax, xlabel='E [GeV]', ylabel=ylabel, xscale='log', xlim=[50, 2e5], ylim=[6e3, 17e3])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    params = [6.577e+07, 2.253, 6.235, 0.239, 9.196, 0.257, 1.020, 0.969]

    plot_data(ax, 'AMS-02_H_rigidity.txt', 2.7, 1., '^', 'tab:olive', 'AMS-02', 2)
    plot_data(ax, 'DAMPE_H_totalEnergy.txt', 2.7, params[6], 's', 'tab:pink', 'DAMPE (+2\%)', 3)
    plot_data(ax, 'CALET_H_kineticEnergy.txt', 2.7, params[7], 'D', 'tab:orange', 'CALET (-3\%)', 4)

    #ax.text(2e2, 18e3, 'stat/sys uncertainties', color='tab:gray', fontsize=22)

    E = np.logspace(1, 6, 1000)
    
    # instantiate the ProtonModel with Q0, alpha, and boostfactor overrides
    model = ProtonModel(Q0=params[0], 
                        alpha=params[1],
                        E_break_le=np.exp(params[2]),
                        ddelta_le=params[3],
                        E_break_he=np.exp(params[4]),
                        ddelta_he=params[5],
                        halo_size=5 * ProtonModel.KPC, 
                        boostfactor=0)
    N = model.GCR_protons(E)
    N27 = np.power(E, 2.7) * N
    ax.plot(E, N27, color='tab:blue', label='No DM', linewidth=4, zorder=5)

    # instantiate the ProtonModel with Q0, alpha, and boostfactor overrides
    model = ProtonModel(Q0=params[0], 
                        alpha=params[1],
                        E_break_le=np.exp(params[2]),
                        ddelta_le=params[3],
                        E_break_he=np.exp(params[4]),
                        ddelta_he=params[5],
                        halo_size=5 * ProtonModel.KPC, 
                        boostfactor=1e-7)
    N = model.GCR_protons(E)
    N27 = np.power(E, 2.7) * N
    #ax.plot(E, N27, color='tab:red', label='DM', linewidth=3.0, ls=':', zorder=5)

    #ax.text(1e2, 16e3, r'$M_\text{DM} = 10^{-3}$', fontsize=25, color='k')

    ax.legend(fontsize=15, loc='best')
    savefig(plt, 'proton_spectrum.pdf')

if __name__== "__main__":
    plot_spectrum()
