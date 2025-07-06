import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import savefig, plot_data, set_axes

def plot_dm_fit():
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    xlabel = 'boost factor'
    ylabel = r'$\chi^2$'
    set_axes(ax, xlabel=xlabel, ylabel=ylabel, xscale='log', yscale='log', xlim=[1e-9, 1e-5], ylim=[5, 5e3])
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    b, chi2 = np.loadtxt('dm_fit_results.txt', unpack=True)
    ax.plot(b, chi2, color='tab:blue', label=r'$M_\text{DM} = 10^{-3}$', linewidth=4, zorder=5)
    ax.hlines(66, 1e-10, 1e-5, color='tab:gray', linestyle='--', linewidth=2.0, label='dof = 66')

    ax.legend(fontsize=25, loc='best')
    savefig(plt, 'dm_fit.pdf')

if __name__== "__main__":
    plot_dm_fit()
