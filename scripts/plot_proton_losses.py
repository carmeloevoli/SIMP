import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('gryphon.mplstyle')
import numpy as np

from utils import savefig, plot_data, set_axes

KPC = 3.086e21  # cm in kpc

def tau_escape(E: float) -> float:
    H = 5 * KPC
    D_0 = 3e28 # cm^2/s
    alpha = 0.5
    return H**2 / (D_0 * E**alpha)

def plot_losses():
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    ylabel = r'E / (dE/dt) [s]'
    set_axes(ax, xlabel='E [GeV]', ylabel=ylabel, xscale='log', xlim=[0.5, 2e5], yscale='log', ylim=[1e8, 1e16])
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    filename = '../tables/dEdt_protons.txt'
    try:
        x, y0, y1, y2 = np.loadtxt(filename, usecols=(0, 1, 2, 3), skiprows=8, unpack=True)
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return

    ax.plot(x, x / y0, color='tab:blue', label=r'$M_\chi = 10^{-3}$', linewidth=3.0, zorder=1)
    ax.plot(x, x / y1, color='tab:orange', label=r'$M_\chi = 10^{-2}$', linewidth=3.0, zorder=2)
    ax.plot(x, x / y2, color='tab:red', label=r'$M_\chi = 10^{-1}$', linewidth=3.0, zorder=3)

    # Calculate and plot escape time
    E = np.logspace(np.log10(0.5), np.log10(2e5), 100)
    tau = tau_escape(E)
    ax.plot(E, tau, color='tab:gray', label='Escape time', linewidth=3.0, zorder=4)

    ax.legend(fontsize=18, loc='best')
    savefig(plt, 'proton_losses.pdf')

if __name__== "__main__":
    plot_losses()
