"""Variational Mode Decomposition — pure NumPy/SciPy implementation.

Implements VMD following Dragomiretskiy & Zosso (2014) "Variational Mode
Decomposition", IEEE Trans. Signal Process.

Provides:
- vmd(): low-level VMD solver (NumPy CPU)
- vmd_torch(): GPU-accelerated VMD solver (PyTorch)
- VMDDecomposer: sklearn-style wrapper with fit/transform, GPU support, disk cache
"""

import os

import numpy as np
from numpy.fft import fft, ifft


def vmd(signal, alpha=2000, tau=0.0, K=4, DC=False, init=1, tol=1e-7, max_iter=500):
    """Variational Mode Decomposition (CPU, NumPy).

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


def vmd_torch(signal, alpha=2000, tau=0.0, K=4, DC=False, init=1, tol=1e-7,
              max_iter=500, device=None):
    """GPU-accelerated VMD using PyTorch FFT.

    Identical algorithm to vmd() but runs on GPU via torch.fft.
    Falls back to CPU if no GPU available.

    Parameters
    ----------
    signal  : (N,) array
        Real-valued 1-D input signal.
    alpha, tau, K, DC, init, tol, max_iter : same as vmd()
    device  : torch.device or None
        Target device.  None → auto-detect GPU, fallback CPU.

    Returns
    -------
    u      : (K, N) ndarray  — modes in time domain (CPU)
    u_hat  : (K, N) ndarray  — modes in frequency domain (CPU)
    omega  : (K,) ndarray    — centre frequencies (CPU)
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signal = np.asarray(signal, dtype=float)
    N = signal.shape[0]
    N_half = N // 2 + 1

    # ── move to device ──
    signal_t = torch.as_tensor(signal, dtype=torch.float64, device=device)

    omega_axis = 2 * torch.pi * torch.arange(N_half, device=device, dtype=torch.float64) / N

    f_hat = torch.fft.fft(signal_t)
    f_half = f_hat[:N_half]

    u_hat = torch.zeros((K, N), dtype=torch.complex128, device=device)
    omega_k = torch.zeros(K, dtype=torch.float64, device=device)

    if init == 1:
        for k in range(K):
            omega_k[k] = torch.as_tensor((0.5 + k) * torch.pi / K,
                                         dtype=torch.float64, device=device)

    lambda_hat = torch.zeros(N, dtype=torch.complex128, device=device)

    for n_iter in range(max_iter):
        u_hat_old = u_hat.clone()

        for k in range(K):
            sum_others = u_hat.sum(dim=0) - u_hat[k]
            residual = f_hat - sum_others + lambda_hat / 2.0

            denom = 1.0 + 2.0 * alpha * (omega_axis - omega_k[k]) ** 2
            u_hat[k, :N_half] = residual[:N_half] / denom

            u_hat[k, N_half:] = torch.conj(u_hat[k, 1:N - N_half + 1].flip(0))

            u_power = torch.abs(u_hat[k, :N_half]) ** 2
            omega_new = torch.sum(omega_axis * u_power) / (torch.sum(u_power) + 1e-20)
            if not (DC and k == 0):
                omega_k[k] = omega_new

        if tau > 0:
            lambda_hat = lambda_hat + tau * (f_hat - u_hat.sum(dim=0))

        u_diff = torch.sum(torch.abs(u_hat - u_hat_old) ** 2, dim=1)
        u_norm = torch.sum(torch.abs(u_hat_old) ** 2, dim=1) + 1e-20
        if torch.all(u_diff / u_norm < tol):
            break

    u = torch.fft.ifft(u_hat, dim=1).real
    return u.cpu().numpy(), u_hat.cpu().numpy(), omega_k.cpu().numpy()


class VMDDecomposer:
    """sklearn-style VMD decomposer with optional GPU acceleration and disk cache.

    Parameters
    ----------
    K      : int, number of modes (default 4)
    alpha  : float, bandwidth penalty (default 2000)
    tau    : float, noise tolerance (default 0)
    DC     : bool, capture DC as separate mode (default 0)
    init   : int, frequency init method (0=zeros, 1=uniform; default 1)
    tol    : float, convergence tolerance (default 1e-7)
    max_iter : int, maximum ADMM iterations (default 500)
    seed   : int, random seed for reproducibility (default 42)
    use_gpu : bool, use GPU via torch if available (default False — CPU faster for N < 100k)
    cache_dir : str or None, directory for .npz cache files (default None)
    cache_tag : str or None, suffix appended to cache filename for domain separation
    """

    def __init__(self, K=4, alpha=2000, tau=0.0, DC=False, init=1,
                 tol=1e-7, max_iter=500, seed=42,
                 use_gpu=False, cache_dir=None, cache_tag=None):
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.cache_tag = cache_tag

        self.imfs_ = None
        self.omega_ = None
        self.n_samples_ = None

    def _make_cache_path(self):
        if self.cache_dir is None:
            return None
        tag = f"_{self.cache_tag}" if self.cache_tag else ""
        return os.path.join(self.cache_dir,
                            f"vmd_K{self.K}_a{self.alpha}_s{self.seed}{tag}.npz")

    def _load_cache(self):
        path = self._make_cache_path()
        if path is None:
            return False
        if os.path.exists(path):
            data = np.load(path)
            self.imfs_ = data["imfs"]
            self.omega_ = data["omega"]
            self.n_samples_ = data["n_samples"]
            return True
        return False

    def _save_cache(self):
        path = self._make_cache_path()
        if path is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez(path, imfs=self.imfs_, omega=self.omega_,
                     n_samples=self.n_samples_)

    def fit_transform(self, signal):
        """Decompose *signal* and store IMFs.

        Uses GPU if use_gpu=True and torch.cuda.is_available();
        loads from disk cache if available.

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

        # Try disk cache first
        if self._load_cache():
            return self.imfs_

        # Run VMD (GPU if available, else CPU)
        gpu_ok = False
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_ok = True
            except ImportError:
                pass

        if gpu_ok:
            u, u_hat, omega = vmd_torch(
                signal,
                alpha=self.alpha, tau=self.tau, K=self.K,
                DC=self.DC, init=self.init, tol=self.tol,
                max_iter=self.max_iter,
            )
        else:
            u, u_hat, omega = vmd(
                signal,
                alpha=self.alpha, tau=self.tau, K=self.K,
                DC=self.DC, init=self.init, tol=self.tol,
                max_iter=self.max_iter,
            )

        self.imfs_ = u.T         # (N, K)
        self.omega_ = omega      # (K,)
        self.n_samples_ = len(signal)

        self._save_cache()
        return self.imfs_

    def transform(self, signal):
        """Decompose a new signal using the same parameters.

        Note: VMD is data-driven — each call is independent.  For strict
        reproducibility, the caller should save returned IMFs.
        This method does NOT use cache (different signal = different result).
        """
        np.random.seed(self.seed)
        signal = np.asarray(signal, dtype=float)

        gpu_ok = False
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_ok = True
            except ImportError:
                pass

        if gpu_ok:
            u, _, _ = vmd_torch(
                signal,
                alpha=self.alpha, tau=self.tau, K=self.K,
                DC=self.DC, init=self.init, tol=self.tol,
                max_iter=self.max_iter,
            )
        else:
            u, _, _ = vmd(
                signal,
                alpha=self.alpha, tau=self.tau, K=self.K,
                DC=self.DC, init=self.init, tol=self.tol,
                max_iter=self.max_iter,
            )
        return u.T  # (N, K)


def decompose_by_domain(signal, masks, K=4, alpha=2000, tol=1e-7, max_iter=500,
                        seed=42, cache_dir=None, use_gpu=False, verbose=True):
    """VMD decomposition with strict temporal separation to prevent data leakage.

    Runs independent VMD on each domain (train/val/test), ensuring that
    a domain's IMFs are computed without access to future data.

    Parameters
    ----------
    signal : (N,) array
        Full 1-D time series.
    masks  : list of (name, mask) tuples
        Each mask is a boolean array of length N selecting the domain.
        e.g. [("train", train_mask), ("val", val_mask), ("test", test_mask)]
    K, alpha, tol, max_iter, seed : VMD parameters (see VMDDecomposer)
    cache_dir : str or None
        Directory for per-domain .npz cache files.
    use_gpu : bool
        Use GPU via torch if available.
    verbose : bool
        Print progress.

    Returns
    -------
    imfs_full : (N, K) ndarray
        Combined IMFs, same indexing as *signal*.
    omegas : dict
        {name: omega_array} for each domain.
    """
    import time

    N = len(signal)
    imfs_full = np.zeros((N, K), dtype=float)
    omegas = {}

    for name, mask in masks:
        seg = signal[mask]
        if verbose:
            print(f"  [VMD] {name}: N={len(seg):,}  alpha={alpha} ...", end=" ", flush=True)
        t0 = time.time()

        vmd = VMDDecomposer(
            K=K, alpha=alpha, tol=tol, max_iter=max_iter, seed=seed,
            use_gpu=use_gpu, cache_dir=cache_dir, cache_tag=name,
        )
        imfs_seg = vmd.fit_transform(seg)  # (n, K)
        imfs_full[mask] = imfs_seg
        omegas[name] = vmd.omega_

        if verbose:
            print(f"{time.time() - t0:.1f}s  omega={np.round(vmd.omega_, 3)}")

    return imfs_full, omegas
