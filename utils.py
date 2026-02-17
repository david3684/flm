"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import argparse
import logging
import os
import sys
import pickle
import time

import fsspec
import lightning
import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import norm
from timm.scheduler import CosineLRScheduler
from math import isfinite
from typing import Union

from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm
from scipy.special import log_ndtr  # stable log
from scipy.interpolate import CubicSpline

def count_parameters(model):
    return sum(p.numel()
               for p in model.parameters()
               if p.requires_grad)


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Mkdirs in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(name, tensor)


class LRHalveScheduler:
    def __init__(self, warmup_steps, n_halve_steps):
        self.warmup_steps = warmup_steps
        self.n_halve_steps = n_halve_steps

    def __call__(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        return 0.5 ** ((current_step - self.warmup_steps)
                       // self.n_halve_steps)


class CosineDecayWarmupLRScheduler(
        CosineLRScheduler,
        torch.optim.lr_scheduler._LRScheduler):
    """Wrap timm.scheduler.CosineLRScheduler
    Enables calling scheduler.step() without passing in epoch.
    Supports resuming as well.
    Adapted from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        # We call either step or step_update, depending on
        # whether we're using the scheduler every epoch or every
        # step.
        # Otherwise, lightning will always call step (i.e.,
        # meant for each epoch), and if we set scheduler
        # interval to "step", then the learning rate update will
        # be wrong.
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)


class LoggingContext:
    """Context manager for selective logging."""

    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


class GradientInspectionCallback(lightning.Callback):
    def __init__(self, num_grads_log):
        self.num_grads_log = 10

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        gradients = []
        for name, param in pl_module.backbone.blocks.named_parameters():
            gradients.append(param.grad.view(-1))

        if gradients:
            grads = torch.cat((gradients))
            if not hasattr(pl_module, 'grad_accum_buffer'):
                pl_module.grad_step = torch.tensor(
                    0, device=pl_module.device)
                pl_module.grad_accum_buffer = torch.zeros(
                    self.num_grads_log,
                    grads.shape[0],
                    device=pl_module.device)
            pl_module.grad_accum_buffer[pl_module.grad_step] = grads
            pl_module.grad_step += 1

        if (hasattr(pl_module, 'grad_accum_buffer')
                and pl_module.grad_step == self.num_grads_log):
            grads = pl_module.grad_accum_buffer
            grad_var = grads.std(0).mean()
            pl_module.log(name='trainer/grad_var',
                          value=grad_var.item(),
                          on_step=True,
                          on_epoch=False,
                          sync_dist=True)
            # TODO: save the grads tensor as a numpy array
            # and visualize mean, median, top-k
            pl_module.grad_accum_buffer.zero_()
            pl_module.grad_step = 0


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ('debug', 'info', 'warning', 'error',
                  'exception', 'fatal', 'critical'):
        setattr(logger,
                level,
                lightning.pytorch.utilities.rank_zero_only(
                    getattr(logger, level)))

    return logger


# Copied from https://github.com/jdeschena/sdtt/blob/bbc54d5b3c5fcffd79602cff17ed34dde1f3eff6/src/sdtt/core/sampling/utils.py#L10
def top_k_top_p_filtering(
        logits,
        top_k=0,
        top_p=0.0,
        filter_value=-float("Inf"),
        dim=-1):
    """Filter a distribution of logits using top-k/top-p (nucleus) filtering.
    Adapted from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
      logits (Tensor): Tensor of logits
      top_k (int, optional): Number of top values to keep.
          Deactivated if k is 0. Defaults to 0.
      top_p (float, optional): Cumulative mass to retain.
          Deactivated if p = 0. Defaults to 0.0.
      filter_value (float, optional): Fill value to replace
          the entries removed by top-k/top-p filtering.
          Defaults to -float('Inf').
      dim (int, optional): Dimension of the filtering. Defaults to -1.

    Returns:
        logits: Tensor whose axis `dim` was filtered.
    """
    if dim != -1:
        logits = torch.transpose(logits, dim, -1)

    assert top_k < logits.size(dim)
    if top_k > 0:
        # Remove all tokens with a probability less than
        # the last token of the top-k
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        to_remove_mask = (
            logits < torch.min(values, dim=-1, keepdim=True)[0]
        )  # min returns a tuple (values, indices)
        logits[to_remove_mask] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cum_probs > top_p
        # Ensures at least one token is kept
        sorted_indices_to_remove[..., 1:] = \
            sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask_to_remove = torch.empty_like(sorted_indices_to_remove)
        mask_to_remove.scatter_(dim=-1,
                                index=sorted_indices,
                                src=sorted_indices_to_remove)
        logits[mask_to_remove] = filter_value

    if dim != -1:
        logits = torch.transpose(logits, dim, -1)

    return logits


def _discrete_prob_map(gamma_t, N=10):
    snr_sqrt = np.exp(-gamma_t / 2)

    def value(x):
        cdf = norm.cdf(x, scale=1) ** (N - 1)
        pdf = norm.pdf(x, loc=snr_sqrt, scale=1)
        return pdf * cdf
    return value


def _discrete_prob_grad(gamma_t, N=10):
    snr_sqrt = np.exp(-gamma_t / 2)

    def value(x):
        coef = -0.5 * snr_sqrt * (x - snr_sqrt)
        cdf = norm.cdf(x, scale=1) ** (N - 1)
        pdf = norm.pdf(x, loc=snr_sqrt, scale=1)
        return coef * pdf * cdf
    return value


def _cache_prob_usdm_in_partition(
        vocab_size=30522, partition_index=0, num_partitions=1,
        log10_num_points=5):
    print(f'Caching partition:{partition_index} / {num_partitions}')
    path = 'integral'
    gamma_min = -5
    gamma_max = -1
    num_points = 10 ** log10_num_points
    p_cache = []
    grad_p_cache = []
    start_time = time.time()
    gammas = np.linspace(gamma_min, gamma_max, num_points)
    n = num_points // num_partitions
    for gamma in gammas[partition_index * n:
                        (partition_index + 1) * n]:
        pt, _ = quad(_discrete_prob_map(gamma, vocab_size),
                     -np.inf, np.inf)
        p_cache.append(pt)
        grad_pt, _ = quad(_discrete_prob_grad(gamma, vocab_size),
                          -np.inf, np.inf)
        grad_p_cache.append(grad_pt)
        if len(p_cache) % 100 == 0:
            print('{}% completed. Time elapsed:{:.2f} mins'.format(
                int(100 * len(p_cache) / num_points),
                (time.time() - start_time) / 60))

    filename = os.path.join(
        path, '{}_{}_{}-{}.pkl'.format(
            vocab_size, log10_num_points, partition_index,
            num_partitions))
    with open(filename, 'wb') as f:
        pickle.dump({
            'vocab_size': vocab_size,
            'gamma_min': gamma_min,
            'gamma_max': gamma_max,
            'num_points': num_points,
            'pt': np.asarray(p_cache),
            'grad_pt': np.asarray(grad_p_cache)}, f)


def test_cache_prob_usdm_in_partition(
        partition_index=0, num_partitions=1, vocab_size=30522,
        log10_num_points=5):
    path = 'integral/{}_{}_{}-{}.pkl'.format(
        vocab_size, log10_num_points, partition_index,
        num_partitions)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    num_points = data['num_points']

    def _get_index(x):
        return round((num_points - 1) * (x - data['gamma_min']) / (
            data['gamma_max'] - data['gamma_min']))

    pt_errors = []
    grad_pt_errors = []
    gammas = np.linspace(data['gamma_min'],
                         data['gamma_max'],
                         num_points)
    n = num_points // num_partitions
    for gamma in gammas[partition_index * n:
                        (partition_index + 1) * n]:
        pt, _ = quad(
            _discrete_prob_map(gamma, data['vocab_size']),
            -np.inf, np.inf)
        grad_pt, _ = quad(
            _discrete_prob_grad(gamma, data['vocab_size']),
            -np.inf, np.inf)
        idx = _get_index(gamma)
        print(idx)
        pt_errors.append((pt - data['pt'][idx]) ** 2)
        grad_pt_errors.append((grad_pt - data['grad_pt'][idx]) ** 2)
    print('Integral MSE:{} Integral Squared:{:.4f}'.format(
        np.mean(pt_errors), np.mean(data['pt'] ** 2)))
    print('Integral Grad MSE:{} Integral Grad Squared:{:.4f}'.format(
        np.mean(grad_pt_errors), np.mean(data['grad_pt'] ** 2)))


if __name__ == "__main__":
    # Usage: python utils.py --vocab_size=N
    parser = argparse.ArgumentParser(
        description='Caches the integral appearing in the '
        'Diffusion Transformation operator.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=50257,  # For the gpt2 tokenizer
        help='Vocabulary size (default: 50257)')
    parser.add_argument(
        '--partition_index',
        type=int,
        default=0,
        help='Helps parallelize caching')
    parser.add_argument(
        '--num_partitions',
        type=int,
        default=1,
        help='Helps parallelize caching')
    parser.add_argument(
        '--log10_num_points',
        type=int,
        default=5,
        help=('The integral is function that needs to be '
              'evaluated for inputs with a range [-5, 1]. '
              'This argument represents the logarithm base 10 '
              'of number of bins of discretization.'))
    args = parser.parse_args()

    # Computing the integral over [-5, 1] can be slow,
    # so one might prefer splitting it into `num_partitions`
    # bins and compute each separately and merge them later.
    _cache_prob_usdm_in_partition(
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
        vocab_size=args.vocab_size,
        log10_num_points=args.log10_num_points)

    test_cache_prob_usdm_in_partition(
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
        vocab_size=args.vocab_size,
        log10_num_points=args.log10_num_points)

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


# ----------------------------
# Core Exact Computation (Gamma -> Alpha)
# ----------------------------
def compute_alpha_exact(gamma: np.ndarray, K: int, n_gh: int = 100, sigma_floor: float = 1e-12, is_diffusion=False) -> np.ndarray:
    """
    Computes q_c (Alpha) from Gamma using Gauss-Hermite integration.
    This is the ground-truth function mapping Gamma -> Alpha.
    """
    gamma = np.asarray(gamma)

    # 1. Standardized means (assuming tau=0, b=1.0 for this conversion)
    sigma = 1.0 - gamma
    if is_diffusion:
        sigma = np.sqrt(sigma)
    sigma = np.maximum(sigma, sigma_floor)
    
    m_c = gamma / sigma
    
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
    q_c = np.sum(w * np.exp(log_prod_c), axis=-1)
    
    # Debugged. should consider prob. from uniform noise.
    alpha = K/(K-1.) * (q_c - 1./K)

    alpha += (gamma-1) * 1e-10 # minor trick to ensure monotonicity

    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha

def compute_alpha_exact_torch(gamma, K: int, x_np, w_np, sigma_floor: float = 1e-12, is_diffusion=False, device=None) -> torch.Tensor:
    """
    Computes q_c (Alpha) from Gamma using Gauss-Hermite integration (PyTorch version).
    """
    
    dtype = gamma.dtype
    device = gamma.device 

    # 1. Standardized means
    sigma = 1.0 - gamma
    if is_diffusion:
        sigma = torch.sqrt(sigma)
    
    sigma = torch.maximum(sigma, torch.tensor(sigma_floor, device=device, dtype=dtype))
    
    m_c = gamma / sigma
    
    # Tensor 변환
    x = torch.tensor(x_np, dtype=dtype, device=device)
    w = torch.tensor(w_np, dtype=dtype, device=device)
    
    w = w / np.sqrt(np.pi)
    z_nodes = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device)) * x

    m_c_expanded = m_c.unsqueeze(-1)
    z_expanded = z_nodes.unsqueeze(0)

    L_cu = torch.special.log_ndtr(z_expanded + m_c_expanded)

    log_prod_c = (K - 1) * L_cu
    
    q_c = torch.sum(w.unsqueeze(0) * torch.exp(log_prod_c), dim=-1)
    
    alpha = (K / (K - 1.0)) * (q_c - (1.0 / K))

    alpha = alpha + (gamma - 1) * 1e-10

    alpha = torch.clamp(alpha, 0.0, 1.0)

    return alpha

# ----------------------------
# LUT / Spline Implementation
# ----------------------------

def build_luts(K: int, n_points: int = 10000, is_diffusion=False) -> tuple[CubicSpline, CubicSpline]:
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
    gamma_vals = np.linspace(0.0, 1.0, n_points) # cont.
    
    # 2. Compute corresponding Gamma grid (Exact)
    alpha_vals = compute_alpha_exact(gamma_vals, K=K, is_diffusion=is_diffusion) # disc.
    
    # 3. Build Forward Spline (Alpha -> Gamma)
    # Alpha is strictly increasing. Safe.
    lut_g2a = CubicSpline(gamma_vals, alpha_vals)
    
    # 4. Build Inverse Spline (Gamma -> Alpha)
    # Gamma values must be strictly increasing to be 'x' in CubicSpline.
    
    # Sort just in case (though usually monotonic)
    sorted_indices = np.argsort(alpha_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]
    
    # Remove duplicates in Gamma
    # Duplicates often happen at very low alpha (gamma ~ 1/K) or very high alpha (gamma ~ 1.0)
    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]

    # Create Spline
    lut_a2g = CubicSpline(unique_alpha, unique_gamma)
    
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
        dtype = alpha.dtype
        gamma = np.clip(lut(alpha.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(gamma).to(alpha.device, dtype=dtype)
    else:
        return np.clip(lut(alpha), 0.0, 1.0)

def gamma_to_alpha(gamma: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Gamma -> Alpha using the LUT.
    """
    # Clip result to [0, 1] to avoid spline overshoot
    if isinstance(gamma, torch.Tensor):
        dtype = gamma.dtype
        alpha = np.clip(lut(gamma.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(alpha).to(gamma.device, dtype=dtype)
    else:
        return np.clip(lut(gamma), 0.0, 1.0)
    
    
    