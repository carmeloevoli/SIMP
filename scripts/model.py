import numpy as np

class ProtonModel:
    ENORM = 1e1        # GeV, normalization energy for the proton flux
    KPC   = 3.086e21   # cm in kpc
    D0_OVER_H = 0.35e28 / KPC,
    TAU_FILENAME = '../tables/dEdt_protons.txt'

    def __init__(self,
                 Q0: float = 1.,
                 alpha: float = 2.2,
                 delta: float = 0.56,
                 E_break_le: float = 3e2,
                 ddelta_le: float = 0.25,
                 E_break_he: float = 1e4,
                 ddelta_he: float = 0.25,
                 halo_size: float = 5 * KPC,
                 boostfactor: float = 1e-10):
        """Initialize the proton model with parameters."""
        self.Q0          = Q0
        self.alpha       = alpha
        self.delta       = delta
        self.E_break_le  = E_break_le
        self.ddelta_le   = ddelta_le
        self.E_break_he  = E_break_he
        self.ddelta_he   = ddelta_he
        self.H           = halo_size
        self.boostfactor = boostfactor

    def Q(self, E: np.ndarray) -> np.ndarray:
        """Calculate the proton injection spectrum Q(E)."""
        S = 0.01 # smoothness parameter
        val = self.Q0 * np.power(E / self.ENORM, -self.alpha)
        val /= np.power(1 + np.power(E / self.E_break_he, self.ddelta_he / S), S)
        return val

    def D_over_H(self, E: np.ndarray) -> np.ndarray:
        """Calculate the diffusion coefficient D(E)/H."""
        S = 0.1 # smoothness parameter
        val = self.D0_OVER_H * np.power(E / self.ENORM, self.delta)
        val /= np.power(1 + np.power(E / self.E_break_le, self.ddelta_le / S), S)
        return val

    def tau_losses(self, E: np.ndarray) -> np.ndarray:
        """Calculate the energy loss time τ(E)."""
        E_, dEdt_ = np.loadtxt(self.TAU_FILENAME,
                               usecols=(0, 1),
                               skiprows=8,
                               unpack=True)
        dEdt_ = self.boostfactor * dEdt_  # Apply boost factor if needed

        eps = 1e-30
        tau_ = E_ / np.where(dEdt_ == 0, eps, dEdt_)
        return np.exp(np.interp(np.log(E), np.log(E_), np.log(tau_)))

    def GCR_protons(self, E: np.ndarray) -> np.ndarray:
        """Compute the propagated GCR proton spectrum N(E)."""
        if self.boostfactor < 0.:
            raise ValueError("Boost factor must be non-negative.")
        N = self.Q(E)
        N *= 0.5 / self.D_over_H(E)
        tau_loss = self.tau_losses(E)
        tau_escape = self.H / self.D_over_H(E) 
        alpha_H = np.sqrt(tau_escape / tau_loss)
        coth = np.cosh(alpha_H) / np.sinh(alpha_H)
        N /= alpha_H * coth
        return N