import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import trainer_base
import utils
import math
import models
from torch.func import functional_call, jvp
from models.dit import modulate_fused
from torch.func import vmap, functional_call, stack_module_state
import functools

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean(dim=-1)


def stopgrad(x):
    """Stop gradient for x."""
    return x.detach()


class AR(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if (not hasattr(tokenizer, 'mask_token')
                or tokenizer.mask_token is None):
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer,
                         vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert not self.config.algo.time_conditioning
        assert self.config.prior.type == 'none'

    def _process_model_input(self, x0, valid_tokens):
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        valid_tokens = valid_tokens[:, 1:]
        return input_tokens, output_tokens, valid_tokens

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step):
        del current_accumulation_step
        output = self.backbone(input_tokens, None)
        output[:, :, self.mask_index] = self.neg_infinity
        output = output.log_softmax(-1)
        return - output.gather(
            -1, output_tokens[:, :, None])[:, :, 0]

    def generate_samples(self, num_samples, **kwargs):
        # precompute token buffer
        num_pred_tokens = self.num_tokens - 1
        x = torch.zeros(
            (num_samples, num_pred_tokens + 1),
            dtype=torch.long,
            device=self.device)
        x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (torch.distributions.Gumbel(0, 1)
                 .sample((num_samples, num_pred_tokens, self.vocab_size))
                 .to(self.device))
        if self.config.sampling.use_float64:
            noise = noise.to(torch.float64)
        for i in range(num_pred_tokens):
            output = self.backbone(x[:, :i + 1], None)
            output[:, :, self.mask_index] = self.neg_infinity
            output = output.log_softmax(-1)
            y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _process_sigma(self, sigma):
        del sigma
        return None


class MDLM(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == 'ancestral_cache'

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        model_output[:, :, self.mask_index] += self.neg_infinity

        # Normalize the model_output such that x.exp() is
        # a probability distribution over vocab_size.
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True)
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(
            model_output)
        unmasked_score = torch.scatter(
            unmasked_score,
            -1,
            x[..., None],
            torch.zeros_like(unmasked_score[..., :1]))
        unmasked_score[:, :, self.mask_index] = - (
            log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(
            model_output.dtype)[:, :, None]
        model_output = (
            masked_score * masked_indices
            + unmasked_score * (1 - masked_indices))
        return model_output.exp()


class D3PMAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.noise.type == 'log-linear'
        assert self.parameterization == 'mean'

    def _process_model_output(self, model_output, xt, sigma):
        del xt
        del sigma
        if self.subs_masking:
            model_output[:, :, self.mask_index] += self.neg_infinity
        return model_output.log_softmax(dim=-1)

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        del dalpha_t
        assert not low_var
        dt = 1 / self.T
        t = 1 - alpha_t  # Only valid for log-linear schedule.
        t = t.clamp(0., 1.0 - 1e-4)
        alpha_t = alpha_t + torch.zeros_like(xt)
        alpha_s = t - dt + torch.zeros_like(xt)
        assert alpha_s.shape == xt.shape
        assert alpha_t.shape == xt.shape
        log_x_theta_at_x0 = torch.gather(
            log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(
            alpha_s * x_theta_at_m / (t - dt) + 1)
        L_vb_masked = (
            term_1_coef * (term_1_log_nr - term_1_log_dr)
            + term_2_coef * (term_2_log_nr - term_2_log_dr))

        diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
        return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.predictor == 'analytic'

    def _get_score(self, x, sigma):
        return self.forward(x, sigma).exp()

    def _process_model_output(self, model_output, xt, sigma):
        esigm1_log = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            sigma.exp() - 1).log().to(model_output.dtype)
        # logits shape
        # (batch_size, context_length, vocab_size)
        model_output = (model_output
                        - esigm1_log[:, None, None]
                        - np.log(model_output.shape[-1] - 1))
        # The below scatter operation sets the log score
        # for the input word to 0.
        model_output = torch.scatter(
            model_output, -1, xt[..., None],
            torch.zeros_like(model_output[..., :1]))
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        """Computes the SEDD loss for the Absorbing State Diffusion.

        Args:
          log_x_theta: float torch.Tensor with shape (batch_size,
              context_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              context_length), input.
          x0: int torch.Tensor with shape (batch_size,
              context_length), input.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
              time derivative of signal level.
          low_var: bool, low variance loss during training.

        Returns:
          loss with shape (batch_size, context_length).
        """
        assert not low_var
        masked_indices = xt == self.mask_index
        sigma = self._sigma_from_alphat(alpha_t)
        dsigma = - dalpha_t / alpha_t

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_x_theta[masked_indices],
            -1,
            words_that_were_masked[..., None]).squeeze(-1)
        score = log_x_theta[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(
                dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(* xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return dsigma * entropy


def stopgrad(x):
    """Stop gradient for x."""
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)  # (B,)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


def mse_loss(error):
    per_sample = (error ** 2).mean(dim=(1, 2))  # [B]
    return per_sample.mean()


class DUO_BASE(trainer_base.UniformState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        # Handle _orig_mod prefix from torch.compile and filter teacher keys
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            # Filter out teacher keys
            if k.startswith('teacher'):
                continue
            # Strip _orig_mod prefix from torch.compile
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v
        checkpoint['state_dict'] = new_state_dict
        super().on_load_checkpoint(checkpoint)

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return model_output.log_softmax(dim=-1)

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
          x: Either clean input `x0` (one-hot),
            or model's predicted `x_theta` of shape (B, L, V).
          xt: The noisy latent (as indices) of shape (B, L).
          alpha_s: Noise level at s of shape (B, [L | 1], 1).
          alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
          Posterior / approximate posterior of shape (B, L, V).
        """
        if self.config.sampling.use_float64:
            x = x.to(torch.float64)
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(
            self.dtype).to(self.device)
        return (
            (alpha_t * self.vocab_size * x * xt_one_hot + (
                alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
                1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
                alpha_t * self.vocab_size * torch.gather(
                    x, -1, xt[..., None]) + (1 - alpha_t)))

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):  # Computes Eq 5.
        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        assert xt.ndim == 2
        if torch.is_tensor(dalpha_t) and dalpha_t.ndim == 1:
            dalpha_t = dalpha_t.unsqueeze(-1)
        assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
        x_reconst = log_x_theta.exp()  # convert logits to probabilities
        x_bar_theta = self.vocab_size * alpha_t[
            :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
        coeff = dalpha_t / (self.vocab_size * alpha_t)
        x_eq_xt = (x0 == xt).float()
        x_neq_xt = 1 - x_eq_xt
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
        xbar_theta_xt = torch.gather(
            x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x = torch.gather(
            x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
        term1 = self.vocab_size * (1 / xbar_xt
                                   - 1 / xbar_theta_xt)  # Eq 5. term 1

        const = (1 - alpha_t) / (self.vocab_size * alpha_t
                                 + 1 - alpha_t)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                        - (1 / const) * x_neq_xt) * const.log()
        term2_theta = - term2_coefs * (
            x_bar_theta.log().sum(-1)
            - self.vocab_size * xbar_theta_xt.log())
        term2_theta = (
            term2_theta
            - self.vocab_size * alpha_t / (1 - alpha_t) * (
                xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
        term2 = term2_theta + term2_offset
        diffusion_loss = coeff * (term1 - term2)
        assert diffusion_loss.ndim == 2
        return diffusion_loss

    def _ancestral_update(self, x, t, dt, p_x0=None,
                          noise_removal_step=False, step_index=None):
        del p_x0
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        sigma_t = self._sigma_from_alphat(alpha_t)

        assert alpha_t.ndim == 2

        q_xs = self._compute_posterior(
            x=self.forward(x, sigma_t).exp(),
            xt=x,
            alpha_s=alpha_s,
            alpha_t=alpha_t)
        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(
                q_xs.log(), top_p=self.p_nucleus)
        return None, trainer_base.sample_categorical(q_xs, self.config.sampling.temperature)


class Integral(torch.autograd.Function):
    """
    torch module calculating UDLM's p_t 
    """

    @staticmethod
    def forward(ctx, gamma_t, data):
        gamma_max = data['gamma_max']
        gamma_min = data['gamma_min']
        if (gamma_t.max() > gamma_max) or (
                gamma_t.min() < gamma_min):
            # print('max:{} {}'.format(gamma_t.max(), gamma_max))
            # print('min:{} {}'.format(gamma_t.min(), gamma_min))
            gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
        indices = torch.round(
            (data['num_points'] - 1) * (gamma_t - gamma_min) / (
                gamma_max - gamma_min)).long()
        grad_pt = data['grad_pt']
        ctx.grad_pt = grad_pt[indices]

        pt = data['pt'][indices]
        assert pt.shape == gamma_t.shape
        return pt

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        with fsspec.open(self.config.algo.integral_cache_path,
                                         'rb') as f:
            self.integral_cache = pickle.load(f)
        self.integral_cache['pt'] = torch.from_numpy(
            self.integral_cache['pt'])
        self.integral_cache['grad_pt'] = torch.from_numpy(
            self.integral_cache['grad_pt'])
        self.gamma_min = self.config.algo.gamma_min
        self.gamma_max = self.config.algo.gamma_max
        self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
        self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
        self.curriculum_start = self.config.algo.curriculum_start
        self.curriculum_end = self.config.algo.curriculum_end
        self.loss_type = self.config.algo.loss_type
        self._validate_configuration()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.integral_cache['pt'] = self.integral_cache[
            'pt'].to(*args, **kwargs)
        self.integral_cache['grad_pt'] = self.integral_cache[
            'grad_pt'].to(*args, **kwargs)
        return self

    def _compute_gumbel_tau_inverse(self):
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start
        if self.global_step < self.curriculum_start:
            tau = start
        elif self.global_step < self.curriculum_end:
            frac = (self.global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start)
            tau = start + frac * delta
        else:
            tau = -10
        return 10 ** (-tau)

    def training_step(self, batch, batch_idx):
        self.log(name='gumbel_tau_log10',
                         value=1 / self._compute_gumbel_tau_inverse(),
                         on_step=True,
                         on_epoch=False,
                         sync_dist=True)
        return super().training_step(batch, batch_idx)

    def _gamma_to_alphat(self, gamma_t): # eq 10.
        integral = Integral.apply(gamma_t, self.integral_cache)
        return (self.vocab_size * integral - 1) / (
            self.vocab_size - 1)

    def _prior_loss(self):
        alpha_1 = self._gamma_to_alphat(
            torch.tensor(self.gamma_max))
        loss = ((alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
            (self.vocab_size - 1) * alpha_1 + 1) + (
                1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1))
        return loss.item()

    def _q_xt_gaussian(self, x, gamma_t):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert x.ndim == 3
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()
        epsilon = torch.randn(x.shape, dtype=torch.float32,
                                                    device=self.device)
        return alpha_t * x + sigma_t * epsilon

    def nll(self, x0, output_tokens,
                    current_accumulation_step=None, train_mode=False, xT=None, **kwargs):
        # TODO: use xT
        use_true_nll = (self.global_step > self.curriculum_end
                                        or not train_mode)
        if use_true_nll:
            return super().nll(x0, output_tokens,
                                                 current_accumulation_step)
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)    
        gamma_t_prime = self.gamma_max - self.gamma_min
        alpha_t = self._gamma_to_alphat(gamma_t)
        T = 1000
        dalpha_t = gamma_t_prime * T * (
            self._gamma_to_alphat(gamma_t + 1 / T) - alpha_t)
        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        x0_one_hot = F.one_hot(x0, self.vocab_size)
        xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
        xt = xt * self._compute_gumbel_tau_inverse()
        xt_usdm = xt.argmax(-1)
        log_x_theta = self.forward(xt, sigma=sigma)

        return self.nll_per_token(log_x_theta=log_x_theta,
                                                            xt=xt_usdm,
                                                            x0=x0,
                                                            alpha_t=alpha_t,
                                                            dalpha_t=dalpha_t,
                                                            low_var=False)




class Distillation(DUO):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.update_teacher_every = config.algo.update_teacher_every
        self.save_hyperparameters()
        self.teacher = None
        self.teacher_ema = config.algo.teacher_ema
        self.linear_growth_dt = config.algo.linear_growth_dt
        self.linear_growth_min = config.algo.linear_growth_min
        self.linear_growth_max = config.algo.linear_growth_max

    def _validate_configuration(self):
        assert os.path.exists(
            self.config.algo.integral_cache_path), (
            'The integral cache (Eq. 10 in the paper) for '
            f'the {self.config.data.tokenizer_name_or_path} '
            ' tokenizer doesnt exist at '
            f'{self.config.algo.integral_cache_path}. '
            'Please generate it by running the utils.py script, '
            'and ensure the correct path is specified using the '
            'algo.integral_cache_path flag.')
        assert self.loss_type in {
            'kl-fwd', 'kl-bwd', 'posterior', 'kl-posterior'}

    def _maybe_update_teacher_weights(self):
        if self.global_step % self.update_teacher_every != 0:
            return
        if self.teacher_ema:
            self.ema.copy_to(self.teacher.parameters())
        else:
            for better_param, current_param in zip(
                    self.backbone.parameters(), self.teacher.parameters()):
                if current_param.requires_grad:
                    current_param.data.copy_(better_param.data)

    @torch.no_grad()
    def _teacher_logits(self, xt, sigma):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self.backbone)
        self._maybe_update_teacher_weights()

        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.teacher(xt, sigma)
        logits = self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)
        return logits.detach()

    def _sample_trajectory(self, x0, gamma_t, gamma_s):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert gamma_s.ndim == 1
        assert x0.ndim == 2
        x0 = F.one_hot(x0, self.vocab_size).to(
            self.dtype).to(self.device)
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()

        gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
        alpha_s = torch.sigmoid(-gamma_s).sqrt()
        sigma_s = torch.sigmoid(gamma_s).sqrt()

        epsilon = torch.randn(x0.shape, dtype=torch.float32,
                              device=self.device)
        xt = alpha_t * x0 + sigma_t * epsilon
        xs = alpha_s * x0 + sigma_s * epsilon
        return xt, xs

    def _compute_dt(self):
        if self.linear_growth_dt:
            scale = self.global_step / self.trainer.max_steps
            return self.linear_growth_min + scale * (
                self.linear_growth_max - self.linear_growth_min)
        n = self.global_step // self.update_teacher_every
        return 2 ** n / self.T

    def nll(self, x0, output_tokens,
            current_accumulation_step=None, train_mode=None, xT=None):
        # TODO: use xT
        del output_tokens, train_mode
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        dt = self._compute_dt()
        t = torch.clip(t + dt, 0, 1)

        gamma_t = self.gamma_min + t * (self.gamma_max
                                        - self.gamma_min)
        gamma_s = self.gamma_min + (
            t - dt) * (self.gamma_max - self.gamma_min)

        alpha_t = self._gamma_to_alphat(gamma_t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        usdm_alpha_s = self._gamma_to_alphat(gamma_s)
        usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
        assert usdm_alpha_s.ndim == 2

        xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
        xt_discrete = xt.argmax(-1)
        xs_discrete = xs.argmax(-1)
        log_x_theta_student = self.forward(
            xt_discrete, sigma=self._sigma_from_alphat(alpha_t))
        log_x_theta_teacher = self._teacher_logits(
            xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s))
        if self.config.training.loss_precision == 'float64':
            log_x_theta_student = log_x_theta_student.to(torch.float64)
            log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
        if self.loss_type == 'kl-fwd':
            return (log_x_theta_teacher.exp() * (
                log_x_theta_teacher - log_x_theta_student)).sum(-1)
        elif self.loss_type == 'kl-bwd':
            return (log_x_theta_student.exp() * (
                log_x_theta_student - log_x_theta_teacher)).sum(-1)

    def training_step(self, batch, batch_idx):
        self.log(name='dt',
                 value=self._compute_dt(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return super().training_step(batch, batch_idx)


class Rectification(DUO):  # Training as duo, without curriculum
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.save_hyperparameters()
        self.use_linear_schedule = config.algo.use_linear_schedule
        self.use_simple_loss = config.algo.use_simple_loss
        self.onestep_mode = config.algo.onestep_mode
        self.debug = getattr(config.algo, 'debug', False)

    def _compute_gumbel_tau_inverse(self):
        return 1e-10

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False, simple_loss=False):
        if simple_loss:
            loss = F.cross_entropy(
                log_x_theta.view(-1, self.vocab_size),
                x0.view(-1),
                reduction='none')
            loss = loss.view(xt.shape)
            return loss
        else:
            return super().nll_per_token(
                log_x_theta=log_x_theta,
                xt=xt,
                x0=x0,
                alpha_t=alpha_t,
                dalpha_t=dalpha_t,
                low_var=low_var
            )

    def nll(self, x0, output_tokens,
            current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del output_tokens
        if given_t is not None:
            if not_sampling_t:
                assert torch.is_tensor(given_t)
                t = 1-given_t
            else:
                t = self._sample_t(
                    x0.shape[0], current_accumulation_step, given_t=1-given_t)
        else:
            t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            assert 0

        dalpha_t, alpha_t = self.noise(t)

        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        if given_t is not None and xT is not None:
            # x0 with alpha_t, xT with (1-alpha_t)
            random = torch.rand_like(x0, dtype=torch.float32)
            given_t = given_t.unsqueeze(1)
            random = given_t + random * (1 - given_t)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)
        elif xT is None or self.debug:
            if not self.debug:
                assert not self.training, 'xT should be provided during training'
            xT = self.prior_sample(x0.shape[0], x0.shape[1])
            random = torch.rand_like(x0, dtype=torch.float32)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)
        else:
            # x0 with alpha_t, xT with (1-alpha_t)
            random = torch.rand_like(x0, dtype=torch.float32)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)

        log_x_theta = self.forward(xt, sigma=sigma)

        return self.nll_per_token(log_x_theta=log_x_theta,
                                  xt=xt,
                                  x0=x0,
                                  alpha_t=alpha_t,
                                  dalpha_t=dalpha_t,
                                  low_var=train_mode and self.loss_type == 'low_var',
                                  simple_loss=self.use_simple_loss,
                                  )


class FLM(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)
            
    def on_load_checkpoint(self, checkpoint):
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('teacher'):
                continue
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v

        if self.config.mode == 'sample_eval':
            # In eval mode, allow loading checkpoints without learnable_loss_weighting
            if getattr(self.backbone, 'learnable_loss_weighting', None) is not None:
                if not any(k.startswith('backbone.learnable_loss_weighting') for k in new_state_dict.keys()):
                    print("Learnable_loss_weighting not found in checkpoint. Initializing from scratch for eval mode.")
                    for name, param in self.backbone.learnable_loss_weighting.named_parameters():
                        param_key = f'backbone.learnable_loss_weighting.{name}'
                        new_state_dict[param_key] = param.data.clone()
        
        checkpoint['state_dict'] = new_state_dict
        super().on_load_checkpoint(checkpoint)

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def _validate_configuration(self):
        pass

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value = 30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)
    
    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        
        if t_max is None:
            t_max = self.t_max
        
        if accum_step is not None:
            # During training
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]

        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[
                accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        return t

    # convert discrete time schedule alpha_t to continuous time schedule gamma_t
    def _alpha_t_to_gamma(self, alpha_t):
        return utils.alpha_to_gamma(alpha_t, self.lut_a2g)

    def _gamma_to_alphat(self, gamma_t):
        return utils.gamma_to_alpha(gamma_t, self.lut_g2a)

    def corrupt_continuous(self, x0, t):
        t = t.unsqueeze(-1).unsqueeze(-1)

        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data, noise
    
    def loss(self, x0, output_tokens,
                  current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens
        B = x0.shape[0]

        t = self._sample_t_interval(B, current_accumulation_step, t_min = self.t_min, t_max = self.t_max)
        c_t = self._alpha_t_to_gamma(t)

        x_t, target_data, noise = self.corrupt_continuous(x0, c_t)
        f = self.forward(x_t, t)
        tfm_loss = -(target_data * f).sum(dim=-1)
        
        if self.config.algo.learnable_loss_weighting is True:
            loss_weight = self.backbone.learnable_loss_weighting(t)
            loss_weight = loss_weight.unsqueeze(-1)
            tfm_loss = torch.exp(-loss_weight) * tfm_loss + loss_weight
        return tfm_loss

    def nll(self, input_tokens, output_tokens, current_accumulation_step=None, train_mode=False):
        raise NotImplementedError

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens
    
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                         eps=1e-5):
        """Generate samples from the model."""
        if num_steps is None:
            num_steps = self.config.sampling.steps

        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  
        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
        
        for i in range(num_steps):                 
            t_curr = t_vals[i]
            t_next = t_vals[i + 1]
            t_in = t_curr.expand(B)
            c_t_in = self._alpha_t_to_gamma(t_in)
            c_d_in = self._alpha_t_to_gamma(t_next.expand(B)) - c_t_in
            x_1_pred = self.forward(z, t_in)             
            x_1_pred_probs = x_1_pred.exp()  
                           
            if i == num_steps - 1:
                z = x_1_pred_probs
                break
            
            v = (x_1_pred_probs - z) / (1.0 - c_t_in.view(-1, 1, 1) + 1e-5)
            z = z + c_d_in.view(-1, 1, 1) * v

        return z.argmax(dim=-1)
    
class FLM_distill(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.flow_ratio = config.algo.flow_ratio
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self._base_num_steps = float(config.algo.base_num_steps)
        self.d_max_step_size = 1.0 / self._base_num_steps
        self.progressive_distill = config.algo.progressive_distill
        if self.progressive_distill:
            self.distill_round = 0
            self.step_per_round = 10000
        
        self.log_flag = False
        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)
        self.teacher_model = None
        self._is_resuming = (
            config.checkpointing.resume_from_ckpt 
            and config.checkpointing.resume_ckpt_path  is not None
        )
    def setup(self, stage: str):
        if self.teacher_model is None:
            self._initialize_teacher()
        if stage == 'fit' and not self._is_resuming:
            print(">>> Initializing student from teacher...")
            self._initialize_student_from_teacher()
        elif self._is_resuming:
            print(">>> Skipping student initialization (resuming from checkpoint).")

    
    def _initialize_teacher(self):
        path = self.config.algo.teacher_path
        print(f"Loading teacher model from: {path}")

        original_double_temb = self.config.algo.double_temb
        original_learnable_loss_weighting = self.config.algo.learnable_loss_weighting
        # ensure no double temb and no learnable loss weighting for teacher to match ema parameter
        self.config.algo.double_temb = False 
        self.config.algo.learnable_loss_weighting = False
        
        assert self.config.algo.backbone == 'dit', "Only DIT model supported for teacher in FLM_distill"
        model = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        
        self.config.algo.double_temb = original_double_temb
        self.config.algo.learnable_loss_weighting = original_learnable_loss_weighting
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(model, checkpoint)
        model.load_state_dict(state_dict, strict=False)
        self.teacher_model = model.to(self.device).eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False 
    
    def _extract_ema_state_dict(self, model, checkpoint):
        ema_state = checkpoint.get('ema', None)
        if not ema_state:
            print("Warning: No EMA found, using regular state_dict")
            return {k.replace('backbone.', '').replace('._orig_mod.', ''): v 
                    for k, v in checkpoint['state_dict'].items() if k.startswith('backbone.')}

        new_sd = collections.OrderedDict()
        shadow_params = ema_state['shadow_params']
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        
        print(f"EMA shadow_params: {len(shadow_params)}, Model param_names: {len(param_names)}")
        
        # Load EMA params that match
        min_len = min(len(shadow_params), len(param_names))
        for name, val in zip(param_names[:min_len], shadow_params[:min_len]):
            new_sd[name] = val
        
        # Fill in any missing parameters from checkpoint state_dict
        for k, v in checkpoint['state_dict'].items():
            clean_k = k.replace('backbone.', '').replace('._orig_mod.', '')
            if clean_k not in new_sd and clean_k in [n for n, _ in model.named_parameters()]:
                new_sd[clean_k] = v
                print(f"Loaded missing param from state_dict: {clean_k}")
        
        if len(shadow_params) != len(param_names):
            print(f"Warning: EMA param count mismatch. Loaded {min_len}/{len(param_names)} from EMA, rest from state_dict")
        
        return new_sd
    
    def _initialize_student_from_teacher(self):
        with torch.no_grad():
            teacher_dict = self.teacher_model.state_dict()
            student_dict = self.backbone.state_dict()

            for name, param in teacher_dict.items():
                print(f"Copying parameter: {name}")
                if name in student_dict:
                    student_dict[name].copy_(param)

            if hasattr(self.backbone, 'sigma_map_prime') and self.backbone.sigma_map_prime is not None:
                for name, param in self.backbone.sigma_map_prime.named_parameters():
                    if 'mlp.2' in name:
                        param.zero_()
                        print(f"Zero initialized student sigma_map_prime: {name}")

            if hasattr(self.backbone, 'output_layer'):
                self._zero_init_module(self.backbone.output_layer)
                # self._random_init_module(self.backbone.output_layer)
                print("Random initialized student output_layer")
    
    def on_load_checkpoint(self, checkpoint):
        print("Resuming training from checkpoint...")
        self._is_resuming = True
        if self.config.mode == 'sample_eval':
            self._initialize_teacher()
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() 
                                      if not k.startswith('teacher_model.')}
        super().on_load_checkpoint(checkpoint)
        
    def _zero_init_module(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _random_init_module(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def on_train_start(self):
        super().on_train_start()
        if self.teacher_model is None:
            print("Initializing from checkpoint...")
            self._initialize_teacher_from_checkpoint(self.config.algo.teacher_path)
            
            print("Initializing student from teacher with zero-initialized output layer")
            self._initialize_student_from_teacher()

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def _validate_configuration(self):
        pass

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value = 30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)
    
    def forward_no_softmax(self, xt, sigma, sigma_prime=None):

        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(xt, sigma, sigma_prime)

        return model_output

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        
        if t_max is None:
            t_max = self.t_max
        
        if accum_step is not None:
            # During training
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]

        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[
                accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        return t
    # convert discrete time schedule alpha_t to continuous time schedule gamma_t
    def _alpha_t_to_gamma(self, alpha_t):
        return utils.alpha_to_gamma(alpha_t, self.lut_a2g)

    def _gamma_to_alphat(self, gamma_t):
        return utils.gamma_to_alpha(gamma_t, self.lut_g2a)

    def corrupt_continuous(self, x0, t):
        t = t.unsqueeze(-1).unsqueeze(-1)

        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data
    
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)
    
    def teacher_forward(self, xt, t=None, d=None):
        sigma = t.unsqueeze(-1) if t.ndim == 1 else t
        sigma = self._process_sigma(sigma)
        with torch.no_grad(): 
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                model_output = self.teacher_model(xt, sigma)
        
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)
    
    def zero_center_output(self, model_output):
        centered_output = model_output - model_output.mean(dim=-1, keepdim=True)
        return centered_output

    def _maybe_update_progressive_d_max_step(self):
        if not self.progressive_distill:
            return
        if self.step_per_round <= 0:
            return
        current_round = int(self.global_step // self.step_per_round)
        if current_round != self.distill_round:
            self.distill_round = current_round
            self.d_max_step_size = (2 ** self.distill_round) / self._base_num_steps
            if self.d_max_step_size >= 1:
                self.d_max_step_size = 1.0
    
    def loss(self, x0, output_tokens,
                      current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens
        self._maybe_update_progressive_d_max_step()
        B = x0.shape[0]
        L = x0.shape[1]
        V = self.vocab_size
        K_max = self.config.algo.base_num_steps
        d_min = 1.0 / K_max
        max_power = int(math.log2(K_max))


        if self.config.algo.sample_d_on_grid:
            current_max_k = min(self.distill_round , max_power)
            k_sc = torch.randint(0, current_max_k + 1, (B,), device=self.device)
            d_sc = (2.0 ** k_sc) * d_min

            num_steps = (1.0 / d_sc).round()
            i = (torch.rand(B, device=self.device) * num_steps).floor()
            t_sc = i * d_sc
            # print(t_sc, d_sc)
        elif self.config.algo.use_continuous_shortcut:
            d_sc = self._sample_t_interval(B, current_accumulation_step, t_min = 0.0, t_max = self.d_max_step_size).clamp(min=1e-6)
            # d_sc = self.d_max_step * torch.ones(B, device=self.device)
            t_sc = torch.rand(B, device=self.device) * (1.0 - d_sc)
            if self.config.algo.add_boundary:
                p_boundary = 1.0 / self.config.algo.boundary_prob
                is_boundary = torch.rand(B, device=self.device) < p_boundary
                
                t_sc = torch.where(is_boundary, torch.tensor(0.0, device=self.device), t_sc)
                # d_sc = torch.where(is_boundary, torch.tensor(1.0, device=self.device), d_sc)
        

        if self.config.algo.shortcut_on_alpha_t: # half on alpha axis
            c_t_sc = self._alpha_t_to_gamma(t_sc)
            x_t_sc, _ = self.corrupt_continuous(x0, c_t_sc)
            
            f_final_sc = self.forward_no_softmax(x_t_sc, t_sc, t_sc+d_sc)
            
            with torch.no_grad():
                d_half = d_sc / 2.0
                d_half_view = d_half.view(-1, 1, 1)
                c_t_sc_view = c_t_sc.view(-1, 1, 1)

                c_t_mid = self._alpha_t_to_gamma(t_sc + d_half).view(-1, 1, 1)
                c_t_end = self._alpha_t_to_gamma(t_sc + d_sc).view(-1, 1, 1)
                c_d_half_1 = c_t_mid - c_t_sc_view
                c_d_half_2 = c_t_end - c_t_mid
                c_d_sc = c_t_end - c_t_sc_view         
                c_t_sc = c_t_sc_view

                f_theta_s = self.teacher_forward(
                    x_t_sc,
                    t_sc,
                ).exp()
                    
                v_s_hat = (f_theta_s - x_t_sc) / (1.0 - c_t_sc + 1e-5)
                g_theta_s_u = self.forward_no_softmax(
                    x_t_sc, 
                    t_sc, 
                    t_sc+d_half,
                ) 
                    
                # v_su_hat = v_s(x_s) + 1/2(u-s)g_theta(x_s,u)
                v_s_u_hat = (f_theta_s - x_t_sc) / (1.0 - c_t_sc + 1e-5) + (c_d_half_1)/2.0 * g_theta_s_u
                
                # F_s,u(x_s) = x_s + (u-s)v_s(x_s) + 1/2(u-s)^2 g_theta(x_s,u)
                large_f_s_u = x_t_sc + (c_d_half_1) * v_s_hat + 0.5 * (c_d_half_1) ** 2 * g_theta_s_u
                
                f_theta_u = self.teacher_forward(
                    large_f_s_u, 
                    t_sc + d_half,
                ).exp()
                    
                v_u_hat = (f_theta_u - large_f_s_u) / (1.0 - c_t_mid + 1e-5)
                g_theta_u_t = self.forward_no_softmax(
                    large_f_s_u, 
                    t_sc + d_half, 
                    # d_half,
                    t_sc+d_sc,
                ) 

                # v_u,t_hat(x_u') = v_u(x_u') + 1/2*(t-u)*g_theta(x_u',u, t)
                v_u_t_hat = v_u_hat + (c_d_half_2)/2.0 * g_theta_u_t
                
                t_minus_s = c_d_sc
                v_hat = (c_d_half_1 * v_s_u_hat + c_d_half_2 * v_u_t_hat) / t_minus_s
                
                # x_1_hat = stopgrad(x_s + (1-s)*v_hat)
                x_boot = x_t_sc + (1.0 - c_t_sc) * v_hat
                x_boot = x_boot.detach()
        
        f_final_fm = f_theta_s
            
        weight = 0.5 * c_d_sc * (1.0 - c_t_sc)
        f_final = f_final_fm + weight * f_final_sc

        if self.config.algo.shortcut_loss_type == 'mse':
            error = x_boot - f_final  # (B, L, V)
            loss = (error ** 2).mean(dim=-1) * self.vocab_size  # (B, L)
        
                            
        if self.config.algo.learnable_loss_weighting is True:
            loss_weight = self.backbone.learnable_loss_weighting(t_sc, t_sc + d_sc)
            loss_weight = loss_weight.unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight
                
        return loss
        
    def nll(self, input_tokens, output_tokens, current_accumulation_step=None, train_mode=False):
        raise NotImplementedError

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                         eps=1e-5):
        """Generate samples from the model."""
        if num_steps is None:
            num_steps = self.config.sampling.steps
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device
    
        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)        
        t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_curr = t_vals[i]
            t_next = t_vals[i + 1]
            t_in = t_curr.expand(B)
            gamma_t_in = self._alpha_t_to_gamma(t_in)
            c_d_in = self._alpha_t_to_gamma(t_next.expand(B)) - gamma_t_in

            x_1_pred_fm = self.teacher_forward(z, t_in).exp()
            x_1_pred_sc = self.forward_no_softmax(z, t_in, t_next.expand(B))
            v_pred = (x_1_pred_fm - z) / (1.0 - gamma_t_in.view(-1, 1, 1) + eps)
            z = z + v_pred * c_d_in.view(-1, 1, 1) + 0.5 * (c_d_in.view(-1, 1, 1) **2) * x_1_pred_sc
        
        return z.argmax(dim=-1)
    
class FLM_distill_double(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max

        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)
        
        # Teacher will be initialized in on_load_checkpoint after student is loaded
        self.teacher_model_f = None
        self.teacher_model_g = None
        # Check if we're resuming from checkpoint by looking at config
        self._is_resuming = (
            config.checkpointing.resume_from_ckpt 
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        )

    def setup(self, stage: str):
        if self.teacher_model_f is None or self.teacher_model_g is None:
            self._initialize_teacher_f()
            self._initialize_teacher_g()

        if stage == 'fit' and not self._is_resuming:
            print(">>> Initializing student from teacher...")
            self._initialize_student_from_teacher()
        elif self._is_resuming:
            print(">>> Skipping student initialization (resuming from checkpoint).")
    
    def _initialize_teacher_f(self):
        path = self.config.algo.teacher_f_path
        print(f"Loading teacher f model from: {path}")

        original_double_temb = self.config.algo.double_temb
        original_learnable_loss_weighting = self.config.algo.learnable_loss_weighting
        original_separated_blocks = self.config.algo.n_separated_blocks
        # ensure no double temb and no learnable loss weighting for teacher to match ema parameter
        self.config.algo.double_temb = False 
        self.config.algo.learnable_loss_weighting = False
        if self.config.algo.backbone == 'dit':
            model = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'dimamba':
            model = models.dimamba.DiMamba(self.config, vocab_size=self.vocab_size, pad_token_id=self.tokenizer.pad_token_id)
        
        self.config.algo.double_temb = original_double_temb
        self.config.algo.learnable_loss_weighting = original_learnable_loss_weighting
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(model, checkpoint)

        model.load_state_dict(state_dict, strict=False)
        self.teacher_model_f = model.to(self.device).eval()
        for param in self.teacher_model_f.parameters():
            param.requires_grad = False 
    
    # init residual teacher model
    def _initialize_teacher_g(self):
        path = self.config.algo.teacher_g_path
        print(f"Loading teacher g model from: {path}")

        model = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(model, checkpoint)

        model.load_state_dict(state_dict, strict=False)
        self.teacher_model_g = model.to(self.device).eval()
        for param in self.teacher_model_g.parameters():
            param.requires_grad = False 
    
    def _extract_ema_state_dict(self, model, checkpoint):
        ema_state = checkpoint.get('ema', None)
        if not ema_state:
            print("Warning: No EMA found, using regular state_dict")
            return {k.replace('backbone.', '').replace('._orig_mod.', ''): v 
                    for k, v in checkpoint['state_dict'].items() if k.startswith('backbone.')}

        new_sd = collections.OrderedDict()
        shadow_params = ema_state['shadow_params']
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        
        print(f"EMA shadow_params: {len(shadow_params)}, Model param_names: {len(param_names)}")
        
        # Load EMA params that match
        min_len = min(len(shadow_params), len(param_names))
        for name, val in zip(param_names[:min_len], shadow_params[:min_len]):
            new_sd[name] = val
        
        for k, v in checkpoint['state_dict'].items():
            clean_k = k.replace('backbone.', '').replace('._orig_mod.', '')
            if clean_k not in new_sd and clean_k in [n for n, _ in model.named_parameters()]:
                new_sd[clean_k] = v
                print(f"Loaded missing param from state_dict: {clean_k}")
        
        if len(shadow_params) != len(param_names):
            print(f"Warning: EMA param count mismatch. Loaded {min_len}/{len(param_names)} from EMA, rest from state_dict")
        
        return new_sd
    
    def _initialize_student_from_teacher(self):
        with torch.no_grad():
            teacher_dict = self.teacher_model_f.state_dict()
            student_dict = self.backbone.state_dict()
            
            for name, param in teacher_dict.items():
                print(f"Copying parameter: {name}")
                if name in student_dict:
                    student_dict[name].copy_(param)

            if hasattr(self.backbone, 'sigma_map_prime') and self.backbone.sigma_map_prime is not None:
                for name, param in self.backbone.sigma_map_prime.named_parameters():
                    if 'mlp.2' in name:
                        param.zero_()
                        print(f"Zero initialized student sigma_map_prime: {name}")
    
    def on_load_checkpoint(self, checkpoint):
        print("Resuming training from checkpoint...")
        self._is_resuming = True
            
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() 
                                      if not k.startswith('teacher_model.')}
        super().on_load_checkpoint(checkpoint)
        
    def _zero_init_module(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _random_init_module(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def on_train_start(self):
        super().on_train_start()
        if self.teacher_model_f is None or self.teacher_model_g is None:
            print("Initializing from checkpoint...")
            self._initialize_teacher_from_checkpoint(self.config.algo.teacher_path)
            
            print("Initializing student from teacher with zero-initialized output layer")
            self._initialize_student_from_teacher()

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def _validate_configuration(self):
        pass

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value = 30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)
    

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        
        if t_max is None:
            t_max = self.t_max
        
        if accum_step is not None:
            # During training
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]

        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[
                accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        return t

    # convert discrete time schedule alpha_t to continuous time schedule gamma_t
    def _alpha_t_to_gamma(self, alpha_t):
        return utils.alpha_to_gamma(alpha_t, self.lut_a2g)

    def _gamma_to_alphat(self, gamma_t):
        return utils.gamma_to_alpha(gamma_t, self.lut_g2a)

    def corrupt_continuous(self, x0, t):
        t = t.unsqueeze(-1).unsqueeze(-1)

        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data
    
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)
    
    def teacher_f_forward(self, xt, t=None, d=None, use_auxiliary_head=False, use_jvp_attn=False):
        sigma = t.unsqueeze(-1) if t.ndim == 1 else t
        sigma = self._process_sigma(sigma)
        with torch.no_grad(): 
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                model_output = self.teacher_model_f(xt, sigma, use_auxiliary_head=use_auxiliary_head)
        
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

    
    def teacher_g_forward(self, xt, sigma, sigma_prime=None, use_auxiliary_head=False, c_d=None, c_t=None, use_jvp_attn=False):
        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.teacher_model_g(xt, sigma, sigma_prime, use_auxiliary_head=use_auxiliary_head, c_d=c_d, c_t=c_t, use_jvp_attn=use_jvp_attn)

        return model_output

    def forward_no_softmax(self, xt, sigma, sigma_prime=None, use_auxiliary_head=False, c_d=None, c_t=None, use_jvp_attn=False):

        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(xt, sigma, sigma_prime, use_auxiliary_head=use_auxiliary_head, c_d=c_d, c_t=c_t, use_jvp_attn=use_jvp_attn)

        return model_output
    
    def loss(self, x0, output_tokens,
                      current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens
        B = x0.shape[0]
        L = x0.shape[1]
        V = self.vocab_size
        d_sc = self._sample_t_interval(B, current_accumulation_step, t_min = 0.0, t_max = 1.0).clamp(min=1e-5, max=1.0)
        t_sc = torch.rand(B, device=self.device) * (1.0 - d_sc)

        if self.config.algo.add_boundary:
            p_boundary = 1.0 / self.config.algo.boundary_prob
            is_boundary = torch.rand(B, device=self.device) < p_boundary
            
            t_sc = torch.where(is_boundary, torch.tensor(0.0, device=self.device), t_sc)
            d_sc = torch.where(is_boundary, torch.tensor(1.0, device=self.device), d_sc)
            
        c_t_sc = self._alpha_t_to_gamma(t_sc)
        x_t_sc, _ = self.corrupt_continuous(x0, c_t_sc)
        c_d_sc = self._alpha_t_to_gamma(t_sc + d_sc) - c_t_sc
        
        f_final_f = self.teacher_f_forward(x_t_sc, t_sc).exp()
        v_f = (f_final_f - x_t_sc) / (1.0 - c_t_sc.view(-1, 1, 1) + 1e-5)
        f_final_g = self.teacher_g_forward(x_t_sc, t_sc, t_sc+d_sc)
        F_s_t = x_t_sc + v_f * c_d_sc.view(-1, 1, 1) + 0.5 * (((c_d_sc).view(-1, 1, 1))**2) * f_final_g
        student_pred = self.forward(x_t_sc, t_sc, t_sc + d_sc).exp()
        student_v = (student_pred - x_t_sc) / (1.0 - c_t_sc.view(-1, 1, 1) + 1e-5)
        F_s_t_distilled = x_t_sc + c_d_sc.view(-1,1,1) * student_v
        
        error = F_s_t - F_s_t_distilled

        loss = (error ** 2).mean(dim=-1) * self.vocab_size 
           
        
        return loss
        
    def nll(self, input_tokens, output_tokens, current_accumulation_step=None, train_mode=False):
        raise NotImplementedError("FLM only supports meanflow loss")

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                         eps=1e-5):
        """Generate samples from the model."""
        if num_steps is None:
            num_steps = self.config.sampling.steps

        print(f" sampling step {num_steps}")
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  
        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
        
        for i in range(num_steps):
            t_curr = t_vals[i]
            t_next = t_vals[i + 1]

            t_in = t_curr.expand(B)
            gamma_t_in = self._alpha_t_to_gamma(t_in)
            c_d_in = self._alpha_t_to_gamma(t_next.expand(B)) - gamma_t_in

            x_1_pred_probs = self.forward(z, t_in, t_next.expand(B)).exp()
            if i == num_steps - 1:
                z = x_1_pred_probs
                break

            v = (x_1_pred_probs - z) / (1.0 - gamma_t_in.view(-1, 1, 1) + eps)
            z = z + v * c_d_in.view(-1, 1, 1)
                
        return z.argmax(dim=-1)
