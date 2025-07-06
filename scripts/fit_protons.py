import numpy as np
from iminuit import Minuit

from utils import get_data, normalize_data
from model import ProtonModel

def experiment_chi2(filename : str, params : tuple, min_energy : float = 1e1, max_energy : float =1e10):
    """Calculate the chi-squared for a given dataset."""
    xi, alpha, lnEb1, dalpha1, lnEb2, dalpha2, boostfactor, f = params
    E, y, err_tot_lo, err_tot_up = get_data(filename, min_energy, max_energy, addSystematics=True)
    E, y, err_tot_lo, err_tot_up = normalize_data(E, y, err_tot_lo, err_tot_up, norm = f)
    model = ProtonModel(Q0=xi, alpha=alpha,
                        E_break_le=np.exp(lnEb1), ddelta_le=dalpha1,
                        E_break_he=np.exp(lnEb2), ddelta_he=dalpha2,
                        boostfactor=boostfactor)
    y_model = model.GCR_protons(E)
    chi2 = 0.
    for y_i, err_lo_i, err_up_i, y_model_i in zip(y, err_tot_lo, err_tot_up, y_model):
        if y_model_i > y_i:
            chi2 += np.power((y_model_i - y_i) / err_up_i, 2.)
        else:
            chi2 += np.power((y_model_i - y_i) / err_lo_i, 2.)
    return chi2

def fit_protons(initial_params, min_energy = 50.):
    def chi2_function(xi, alpha, lnEb1, dalpha1, lnEb2, dalpha2, boostfactor, fdampe, fcalet):
        chi2 = 0.
        filenames = ['../data/AMS-02_H_rigidity.txt', '../data/CALET_H_kineticenergy.txt', '../data/DAMPE_H_totalenergy.txt']
        scalefactors = [1.0, fcalet, fdampe]
        for filename, f in zip(filenames, scalefactors):
            chi2 += experiment_chi2(filename, [xi, alpha, lnEb1, dalpha1, lnEb2, dalpha2, boostfactor, f], min_energy=min_energy)
        return chi2
    
    def len_data():
        N = 0
        filenames = ['../data/AMS-02_H_rigidity.txt', '../data/CALET_H_kineticenergy.txt', '../data/DAMPE_H_totalenergy.txt']
        for filename in filenames:
            E, _, _, _ = get_data(filename, min_energy)
            N += len(E)
        return N

    """Perform the chi-squared fit for the proton model."""
    # Perform minimization
    m = Minuit(chi2_function, xi=initial_params[0], alpha=initial_params[1], 
               lnEb1=initial_params[2], dalpha1=initial_params[3],
               lnEb2=initial_params[4], dalpha2=initial_params[5],
               boostfactor=initial_params[6], 
               fdampe=initial_params[7], fcalet=initial_params[8])
    m.errordef = Minuit.LEAST_SQUARES

    m.fixed['boostfactor'] = True

    m.limits['xi'] = (1e6, 1e9)  # xi in GeV^-1 cm^-3 s^-1
    m.limits['alpha'] = (0.1, 3.0)
    m.limits['lnEb1'] = (np.log(100.), np.log(1e4))  
    m.limits['dalpha1'] = (0.01, 0.5)  
    m.limits['lnEb2'] = (np.log(1e3), np.log(1e5)) 
    m.limits['dalpha2'] = (0.01, 0.5) 
    m.limits['fdampe'] = (0.1, 2.0)  # DAMPE scaling factor
    m.limits['fcalet'] = (0.1, 2.0)  # CALET scaling factor

    m.tol = 1e-5  # Set tolerance for convergence

    # Optimize using simplex and migrad algorithms
    m.simplex()
    m.migrad()
    m.hesse()
    #m.minos()

    # Print fit summary
    print("Fit Summary:")
    print(f"Parameters: {m.values}")
    print(f"Errors: {m.errors}")
    print(f"Chi-squared: {m.fval}")
    #print(f"Covariance matrix:\n{m.covariance}")
    print(f"Valid: {m.valid}")
    
    print(m.params)

    # Draw the matrix of correlations
    #m.draw_mnprofile("boostfactor")

    # Print fit results
    print(f'xi        : {m.values[0]:.1f}')
    print(f'alpha     : {m.values[1]:.2f}')
    print(f'LE E_b    : {np.exp(m.values[2]):.0f}')
    print(f'LE dalpha : {m.values[3]:.2f}')
    print(f'HE E_b    : {np.exp(m.values[4]):.0f}')
    print(f'HE dalpha : {m.values[5]:.2f}')
    print(f'boostfactor: {np.exp(m.values[6]):.2e}')
    print(f'DAMPE f   : {m.values[7]:.2f}')
    print(f'CALET f   : {m.values[8]:.2f}')
    print(f'chi2      : {m.fval:.2f}')
    dof = len_data() - m.nfit - 1
    print(f'dof       : {dof}')

    print(f'[{m.values[0]:.3e}, {m.values[1]:.3f}, {m.values[2]:.3f}, {m.values[3]:.3f}, {m.values[4]:.3f}, {m.values[5]:.3f}, {m.values[6]:.3f}, {m.values[7]:.3f}]')
    return m.values, m.errors, m.fval, dof

if __name__ == "__main__":
    # Initial guess for the parameters
    # xi, alpha, lnEb1, dalpha1, lnEb2, dalpha2, fdampe, fcalet
    f = open('dm_fit_results.txt', 'w')
    for boostfactor in np.logspace(-9, -5, 100):
        initial_guess = [7e7, 2.8, np.log(500.), 0.20, np.log(1e4), 0.20, boostfactor, 1.0, 1.0]  
        values, errors, fval, dof = fit_protons(initial_guess)
        f.write(f'{boostfactor:.3e} {fval:.3f}\n')
    f.close()
