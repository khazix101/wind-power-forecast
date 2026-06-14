"""Variational Mode Decomposition — pure NumPy/SciPy implementation.

Implements VMD following Dragomiretskiy & Zosso (2014) "Variational Mode
Decomposition", IEEE Trans. Signal Process.

Provides:
- vmd(): low-level VMD solver
- VMDDecomposer: sklearn-style wrapper with fit/transform
"""

import numpy as np
from numpy.fft import fft, ifft


def vmd(signal, alpha=2000, tau=0.0, K=4, DC=False, init=1, tol=1e-7, max_iter=500):
    """Variational Mode Decomposition.

    Parameters
    ----------
    signal : (N,) array
        Real-valued 1-D input signal.
    alpha  : float
        Bandwidth constraint (penalty factor).  Default 2000.
    tau    : float
        Lagrange multiplier update step (noise tolerance).  0 = clean signal.
    K      : int
        Number of extracted modes (IMFs).
    DC     : bool
        If True, the first mode is captured as the DC (zero-frequency) component.
    init   : int
        0 = all omegas start at 0;  1 = omegas uniformly spaced.
    tol    : float
        Convergence tolerance (relative change in modes).
    max_iter : int
        Maximum ADMM iterations.

    Returns
    -------
    u : (K, N) array
        Decomposed modes (real-valued) in original time domain.
    u_hat : (K, N) complex array
        Modes in frequency domain (full spectrum).
    omega : (K,) array
        Final centre frequencies (radians / sample, in [0, pi]).
    """
    N = signal.shape[0]
    N_half = N // 2 + 1

    # ── frequency grid (non-negative half) ──
    omega_axis = 2 * np.pi * np.arange(N_half) / N  # [0 … π]

    # ── signal in frequency domain ──
    f_hat = fft(signal)              # full spectrum (N,)
    f_half = f_hat[:N_half]          # non-negative half

    # ── initialise modes (frequency domain, full spectrum) ──
    u_hat = np.zeros((K, N), dtype=complex)
    omega_k = np.zeros(K)

    if init == 1:
        # centre frequencies spread uniformly across [0, π]
        for k in range(K):
            omega_k[k] = (0.5 + k) * np.pi / K
    # else: init=0 → all zeros (already done)

    lambda_hat = np.zeros(N, dtype=complex)

    # ── dual ascent step for DC mode ──
    u_diff = tol + 1.0  # convergence measure

    for n_iter in range(max_iter):
        u_hat_old = u_hat.copy()

        # --- update each mode ---
        for k in range(K):
            # residual: f - sum_{i≠k} u_i + lambda/2
            sum_others = u_hat.sum(axis=0) - u_hat[k]
            residual = f_hat - sum_others + lambda_hat / 2.0

            # Wiener filter denominator (non-negative half)
            denom = 1.0 + 2.0 * alpha * (omega_axis - omega_k[k]) ** 2
            u_hat[k, :N_half] = residual[:N_half] / denom

            # symmetric conjugate for negative frequencies
            u_hat[k, N_half:] = np.conj(u_hat[k, 1 : N - N_half + 1][::-1])

            # update centre frequency (from non-negative half)
            u_power = np.abs(u_hat[k, :N_half]) ** 2
            omega_new = np.sum(omega_axis * u_power) / (np.sum(u_power) + 1e-20)
            if not (DC and k == 0):
                omega_k[k] = omega_new

        # --- update Lagrange multiplier ---
        if tau > 0:
            lambda_hat = lambda_hat + tau * (f_hat - u_hat.sum(axis=0))

        # --- convergence check ---
        u_diff = np.sum(np.abs(u_hat - u_hat_old) ** 2, axis=1)
        u_norm = np.sum(np.abs(u_hat_old) ** 2, axis=1) + 1e-20
        if np.all(u_diff / u_norm < tol):
            break

    # ── back to time domain ──
    u = np.real(ifft(u_hat, axis=1))

    return u, u_hat, omega_k


class VMDDecomposer:
    """sklearn-style VMD decomposer.

    Parameters
    ----------
    K     : int, number of modes (default 4)
    alpha : float, bandwidth penalty (default 2000)
    tau   : float, noise tolerance (default 0)
    DC    : bool, capture DC as separate mode (default 0)
    init  : int, frequency init method (0=zeros, 1=uniform; default 1)
    tol   : float, convergence tolerance (default 1e-7)
    max_iter : int, maximum ADMM iterations (default 500)
    seed  : int, random seed for reproducibility (default 42)
    """

    def __init__(self, K=4, alpha=2000, tau=0.0, DC=False, init=1,
                 tol=1e-7, max_iter=500, seed=42):
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed

        self.imfs_ = None
        self.omega_ = None
        self.n_samples_ = None

    def fit_transform(self, signal):
        """Decompose *signal* and store IMFs.

        Parameters
        ----------
        signal : (N,) array
            Real-valued 1-D time series.

        Returns
        -------
        imfs : (N, K) array
            Columns = IMF1, IMF2, …, IMF_K.
        """
        np.random.seed(self.seed)
        signal = np.asarray(signal, dtype=float)
        u, u_hat, omega = vmd(
            signal,
            alpha=self.alpha,
            tau=self.tau,
            K=self.K,
            DC=self.DC,
            init=self.init,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.imfs_ = u.T         # (N, K)
        self.omega_ = omega      # (K,)
        self.n_samples_ = len(signal)
        return self.imfs_

    def transform(self, signal):
        """Decompose a new signal using the same parameters.

        Note: VMD is data-driven — each call is independent.  For strict
        reproducibility, the caller should save returned IMFs.
        """
        np.random.seed(self.seed)
        signal = np.asarray(signal, dtype=float)
        u, _, _ = vmd(
            signal,
            alpha=self.alpha,
            tau=self.tau,
            K=self.K,
            DC=self.DC,
            init=self.init,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        return u.T  # (N, K)
