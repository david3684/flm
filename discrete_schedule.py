from math import isfinite
from typing import Union

import numpy as np
from numpy.polynomial.hermite import hermgauss
import torch
from scipy.stats import norm
from scipy.special import log_ndtr  # stable log
from scipy.interpolate import CubicSpline  # Added for LUT/Spline

# ----------------------------
# Utilities: standardized means
# ----------------------------
def standardized_means(alpha: float, tau: float, b: float, diffusion=False):
    """
    Returns (m_c, m_u, m_a, sigma), where
      sigma = b * (1 - alpha),
      m_c = (alpha - tau) / sigma   (label / 'correct'),
      m_u = -tau / sigma            (other data),
      m_a = 0.0                     (absorbing)
    """
    sigma = b * (1.0 - alpha)
    if diffusion:
      sigma = sigma ** 0.5
    if sigma <= 0.0:
        sigma = 1e-12
    m_c = (alpha - tau) / sigma
    m_u = (-tau) / sigma
    m_a = 0.0
    return m_c, m_u, m_a, sigma

# ----------------------------
# Core: GH with precomputed log Φ-shifts (≤ 6 calls)
# ----------------------------
def compute_qs_fast(alpha: float, tau: float, b: float, K: int, M: int, *,
                    n_gh: int = 100, sigma_floor: float = 1e-12,
                    diffusion=False) -> tuple[float, float, float]:
    """
    Returns (q_c, q_u, q_a) using log-stabilized Gauss–Hermite and
    only a constant number of log_ndtr calls per evaluation.

    q_c : probability the label ('correct') class wins (per label)
    q_u : probability a particular non-label data class wins (per class)
    q_a : probability a particular absorbing class wins (per class)
    """
    # standardized means
    m_c, m_u, m_a, sigma = standardized_means(alpha, tau, b, diffusion)
    if sigma < sigma_floor:
        sigma = sigma_floor  # keep GH numerically sane; values remain consistent

    # GH nodes/weights for exp(-x^2); normalize to N(0,1)
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x  # Z ~ N(0,1) evaluated at √2 x_ℓ

    # --- Precompute the LOG-CDFs for the few unique shifts we need ---
    # 0-shift (same-class competitors)
    L0     = log_ndtr(z_nodes)                # log Φ(z)
    # label vs absorbing / absorbing vs label
    L_ca   = log_ndtr(z_nodes + m_c)          # log Φ(z + (m_c - 0))
    L_ac   = log_ndtr(z_nodes - m_c)          # log Φ(z + (0   - m_c))
    # data vs absorbing / absorbing vs data
    L_ua   = log_ndtr(z_nodes + m_u)          # log Φ(z + (m_u - 0))
    L_au   = log_ndtr(z_nodes - m_u)          # log Φ(z + (0   - m_u))
    # label vs data / data vs label
    d_cu   = m_c - m_u
    L_cu   = log_ndtr(z_nodes + d_cu)         # log Φ(z + (m_c - m_u))
    L_uc   = log_ndtr(z_nodes - d_cu)         # log Φ(z + (m_u - m_c))

    # --- Build node-wise log-products for each grouped case ---
    # Label winner: (K-1) non-label data + M absorbing competitors
    #   log_prod_c(u) = (K-1)*log Φ(z + (m_c - m_u)) + M*log Φ(z + (m_c - 0))
    log_prod_c = (K - 1) * L_cu + M * L_ca

    # Wrong-data winner (per class): 1 label + (K-2) other data + M absorbing
    #   log_prod_u(u) = log Φ(z + (m_u - m_c)) + (K-2)*log Φ(z) + M*log Φ(z + (m_u - 0))
    if K > 1:
        log_prod_u = L_uc + max(K - 2, 0) * L0 + M * L_ua
    else:
        log_prod_u = None  # no wrong-data class exists

    # Absorbing winner (per class): 1 label + (K-1) data + (M-1) absorbing
    #   log_prod_a(u) = log Φ(z + (0 - m_c)) + (K-1)*log Φ(z + (0 - m_u)) + (M-1)*log Φ(z)
    if M > 0:
        log_prod_a = L_ac + (K - 1) * L_au + max(M - 1, 0) * L0
    else:
        log_prod_a = None  # no absorbing class exists

    # --- Weighted sum over nodes; clip exponents for safety ---
    def weighted_exp_sum(logv):
        # return float(np.sum(w * np.exp(np.clip(logv, -745.0, 745.0))))  # -745 ~ float64 underflow
        return float(np.sum(w * np.exp(logv)))  # -745 ~ float64 underflow

    q_c = weighted_exp_sum(log_prod_c)
    q_u = weighted_exp_sum(log_prod_u) if (log_prod_u is not None and K > 1) else 0.0
    q_a = weighted_exp_sum(log_prod_a) if (log_prod_a is not None and M > 0) else 0.0

    return q_c, q_u, q_a


def gamma_to_alpha(alpha: np.ndarray, K: int, n_gh: int = 100, sigma_floor: float = 1e-12) -> np.ndarray:
    """
    Computes q_c (Gamma) from Alpha using Gauss-Hermite integration.
    
    Settings hardcoded from previous implementation:
      - tau = 0
      - M = 0 (no absorbing classes)
      - b = 1.0
      - diffusion = False
    
    Optimized Logic:
      1. sigma = 1 - alpha
      2. m_c = alpha / sigma
      3. m_u = 0 (since tau=0)
      4. dist = m_c - m_u = m_c
      5. q_c = E[ Phi(z + m_c)^(K-1) ]
    
    Args:
        alpha: Shape (B,) array of alpha values.
        K: Number of classes.
        n_gh: Number of Gauss-Hermite nodes.
        sigma_floor: Minimum value for sigma to avoid division by zero.
        
    Returns:
        q_c: Shape (B,) array of probabilities.
    """
    # Ensure alpha is an array
    alpha = np.asarray(alpha)

    # 1. Standardized means (Optimized for tau=0, b=1.0)
    # sigma = b * (1 - alpha) = 1 - alpha
    sigma = 1.0 - alpha
    sigma = np.maximum(sigma, sigma_floor)
    
    # m_c = (alpha - tau) / sigma = alpha / sigma
    m_c = alpha / sigma
    
    # 2. GH nodes/weights
    # x, w shape: (n_gh,)
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x  # Z ~ N(0,1)

    # 3. Broadcasting Setup
    # We want final log calculations to be shape (B, n_gh).
    m_c_expanded = m_c[:, None]   # (B, 1)
    z_expanded = z_nodes[None, :] # (1, n_gh)

    # 4. Compute Log-CDFs
    # For tau=0, m_u = 0.
    # The distance between correct and incorrect means is d_cu = m_c - m_u = m_c.
    # We need log(Phi(z + d_cu)) = log(Phi(z + m_c))
    
    # L_cu shape: (B, n_gh)
    L_cu = log_ndtr(z_expanded + m_c_expanded)

    # 5. Build node-wise log-products
    # Probability correct class wins vs (K-1) incorrect classes.
    # log_prod_c = (K - 1) * log(Phi(z + m_c))
    # Note: M=0, so the absorbing term is gone.
    log_prod_c = (K - 1) * L_cu

    # 6. Weighted sum over nodes (axis=-1)
    # q_c = sum(w * exp(log_prod_c))
    q_c = np.sum(w * np.exp(log_prod_c), axis=-1)

    return q_c


# ----------------------------
# Core Exact Computation (Alpha -> Gamma)
# ----------------------------
def compute_gamma_exact(alpha: np.ndarray, K: int, n_gh: int = 100, sigma_floor: float = 1e-12) -> np.ndarray:
    """
    Computes q_c (Gamma) from Alpha using Gauss-Hermite integration.
    This is the ground-truth function mapping Alpha -> Gamma.
    """
    alpha = np.asarray(alpha)

    # 1. Standardized means (assuming tau=0, b=1.0 for this conversion)
    sigma = 1.0 - alpha
    sigma = np.maximum(sigma, sigma_floor)
    
    # m_c = alpha / sigma
    m_c = alpha / sigma
    
    # 2. GH nodes/weights
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x

    # 3. Broadcasting
    m_c_expanded = m_c[:, None]   # (B, 1)
    z_expanded = z_nodes[None, :] # (1, n_gh)

    # 4. Compute Log-CDFs
    # L_cu = log(Phi(z + m_c))
    L_cu = log_ndtr(z_expanded + m_c_expanded)

    # 5. Weighted sum
    # log_prod_c = (K - 1) * L_cu
    log_prod_c = (K - 1) * L_cu
    gamma = np.sum(w * np.exp(log_prod_c), axis=-1)

    gamma += (alpha-1) * 1e-10 # minor trick to ensure monotonicity

    gamma = np.clip(gamma, 0.0, 1.0)
    gamma = np.where(alpha==0., 0., gamma)
    gamma = np.where(alpha==1., 1., gamma)

    return gamma

# ----------------------------
# LUT / Spline Implementation
# ----------------------------

def build_luts(K: int, n_points: int = 10000) -> tuple[CubicSpline, CubicSpline]:
    """
    Builds two lookup tables (Splines):
    1. Alpha -> Gamma (Forward)
    2. Gamma -> Alpha (Inverse)
    
    Reverted to Linear (Uniform) spacing.
    Chebyshev nodes concentrate points at 0 and 1, but for large K, the curve 
    is often sigmoid-like (flat at ends, steep in middle). 
    Uniform spacing captures the transition region better.
    """
    # 1. Create Alpha grid using Uniform Spacing
    # Simple linspace covers the whole range evenly.
    alpha_vals = np.linspace(0.0, 1.0, n_points)
    
    # 2. Compute corresponding Gamma grid (Exact)
    gamma_vals = compute_gamma_exact(alpha_vals, K=K)
    
    # 3. Build Forward Spline (Alpha -> Gamma)
    # Alpha is strictly increasing. Safe.
    lut_a2g = CubicSpline(alpha_vals, gamma_vals)
    
    # 4. Build Inverse Spline (Gamma -> Alpha)
    # Gamma values must be strictly increasing to be 'x' in CubicSpline.
    
    # Sort just in case (though usually monotonic)
    sorted_indices = np.argsort(gamma_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]
    
    # Remove duplicates in Gamma
    # Duplicates often happen at very low alpha (gamma ~ 1/K) or very high alpha (gamma ~ 1.0)
    unique_gamma, unique_indices = np.unique(gamma_sorted, return_index=True)
    print(len(unique_indices))
    unique_alpha = alpha_sorted[unique_indices]

    print(unique_gamma.min(), unique_gamma.max())
    print(unique_alpha.min(), unique_alpha.max())
    
    # Create Spline
    lut_g2a = CubicSpline(unique_gamma, unique_alpha)
    
    return lut_a2g, lut_g2a

# Initialize LUTs globally (lazy loading or explicit init recommended in real apps, 
# but running here for immediate use)
# Using a default K=50000 as per previous context.

# LUT_A2G, LUT_G2A = build_luts(K=50000)

def alpha_to_gamma(alpha: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Alpha -> Gamma using the LUT.
    """
    if isinstance(alpha, torch.Tensor):
        gamma = np.clip(lut(alpha.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(gamma).to(alpha.device)
    else:
        return np.clip(lut(alpha), 0.0, 1.0)

def gamma_to_alpha(gamma: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Gamma -> Alpha using the LUT.
    """
    # Clip result to [0, 1] to avoid spline overshoot
    if isinstance(gamma, torch.Tensor):
        alpha = np.clip(lut(gamma.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(alpha).to(gamma.device)
    else:
        return np.clip(lut(gamma), 0.0, 1.0)