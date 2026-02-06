from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from omegaconf import OmegaConf
from scipy.special import gamma

from ..models import DiffusionModel
from ..sampling import Sampler
from ..utils import AGGRESSIVE_LOGGING_ENABLED, log_tensor_stats
from .adjoint_matching import AMTrainerFlow
from .objective import ZDT5Torch

EPS = 1e-6


def sample_lambda_first_quadrant(shape=(1,)):
    theta = (torch.pi / 2) * (
        torch.rand(*shape, dtype=torch.float32).clamp(EPS, 1 - EPS)
    )
    return torch.stack([torch.sin(theta), torch.cos(theta)], axis=-1)


SAVE = False
BREAKPOINT = False
class ChebyshevTrainer(AMTrainerFlow):
    def __init__(
        self,
        config: OmegaConf,
        model: DiffusionModel,
        base_model: DiffusionModel,
        pre_trained_model: DiffusionModel,
        device: Optional[torch.device] = None,
        sampler: Optional[Sampler] = None,
        ref: Optional[torch.Tensor] = None,
    ):
        self.alpha_div = config.get("alpha_div", 1.0)
        self.lmbda = lmbda = config.get("lmbda", 1.0)
        self.problem = ZDT5Torch(
            n=2, device=device if device is not None else torch.device("cpu")
        )
        self.pre_trained_model = pre_trained_model
        self.ref = ref
        self.k = k = 2  # the reward space
        self.ck = (np.pi ** (k / 2)) / (2**k * gamma(k / 2 + 1))

        logging.info(f"[ChebyshevTrainer] Initialized with alpha_div={self.alpha_div}, lambda={self.lmbda}, k={k}, ck={self.ck:.6f}")
        logging.info(f"[ChebyshevTrainer] Reference point: {ref}")

        def grad_reward_fn(x):
            cbsh = self.compute_chebyshev_grad(x)
            div = self.divergence(x)
            total_grad = self.lmbda * cbsh - div
            logging.debug(f"[grad_reward_fn] Chebyshev grad norm: {torch.norm(cbsh).item():.6f}, Divergence norm: {torch.norm(div).item():.6f}, Total norm: {torch.norm(total_grad).item():.6f}")
            return total_grad

        self.lmbda = lmbda

        super().__init__(
            config.adjoint_matching,
            model,
            base_model,
            grad_reward_fn,
            None,
            device=device,
            sampler=sampler,
        )

    def divergence(self, x):
        zero = torch.tensor(0, device=x.device).float().detach()
        score_base = self.base_model.score_func(x, zero)
        score_pretrained = self.pre_trained_model.score_func(x, zero)
        div = self.alpha_div * (score_base - score_pretrained)
        logging.debug(f"[divergence] Base score norm: {torch.norm(score_base).item():.6f}, Pretrained score norm: {torch.norm(score_pretrained).item():.6f}, Divergence norm: {torch.norm(div).item():.6f}")
        return div

    def compute_chebyshev_grad(self, x, MC_times=500):
        # Compute norms of the 2D points:  (Batch_size, sampling_set_n * 2)
        if SAVE:
            with open("input_debug.pt", "wb") as f:
                torch.save(x, f)
            
        if BREAKPOINT:
            breakpoint()
        batch_size, dim = x.shape
        sampling_set_n = dim // self.k
        x_norms = torch.norm(x.view(batch_size, sampling_set_n, self.k), dim=2)  # Shape: (batch_size, sampling_set_n)
        
        # Make sure x_reshaped requires grad BEFORE any operations
        x = x.requires_grad_(True)
        
        # (batch_size, sampling_set_n, 2)
        rewards = self.problem.evaluate(x.view(-1, self.k)).reshape(
            batch_size, sampling_set_n, -1
        )

        
        # Compute actual hypervolumes for verification (detached, not part of grad computation)
        with torch.no_grad():
            actual_volumes = []
            for i in range(batch_size):
                actual_volume = Hypervolume(self.ref.cpu()).compute(rewards[i].detach().cpu())
                actual_volumes.append(actual_volume)
            actual_volumes = torch.tensor(actual_volumes).to(x.device)

        if BREAKPOINT:
            breakpoint()
            
        lambda_ = sample_lambda_first_quadrant((batch_size, MC_times)).to(x.device)
        
        # Use unsqueeze and expand carefully to maintain gradient flow
        lambda_ = lambda_.unsqueeze(2).expand(-1, -1, sampling_set_n, -1)
        rewards_expanded = rewards.unsqueeze(1).expand(-1, MC_times, -1, -1)
        
        # Compute with proper gradient tracking
        diff = (rewards_expanded - self.ref) / lambda_
        diff_dot = torch.relu(diff) ** self.k
        
        s_lambda = -torch.logsumexp(-diff_dot, dim=3)
        
        hypervolume_estimates = torch.logsumexp(s_lambda, dim=2)
        
        hypervolume_values = self.ck * hypervolume_estimates.mean(dim=1)
        
        # Compare estimated vs actual
        with torch.no_grad():
            rel_error = torch.where(
                actual_volumes > 1e-6,
                torch.abs(hypervolume_values.detach() - actual_volumes)
                / actual_volumes,
                torch.abs(hypervolume_values.detach() - actual_volumes),
            )

        if BREAKPOINT:
            breakpoint()
        # Compute gradients with respect to x_reshaped using autograd.grad
        loss = hypervolume_values.sum()
        loss.backward()
        
        grads = x.grad
        
        if AGGRESSIVE_LOGGING_ENABLED:
            logging.info(f"[Chebyshev Gradient] Input x shape: {x.shape}, requires_grad: {x.requires_grad}")
            log_tensor_stats("x", x, "Chebyshev Gradient")
            log_tensor_stats("x_norms", x_norms, "Chebyshev Gradient")
            logging.info(f"[Chebyshev Gradient] Input x norms means per group {x_norms.mean(dim=1).tolist()}")
            log_tensor_stats("rewards", rewards, "Chebyshev Gradient")
            logging.info(f"[Chebyshev Gradient] Actual hypervolumes: {actual_volumes.tolist()}")
            log_tensor_stats("actual_volumes", actual_volumes, "Chebyshev Gradient")
            log_tensor_stats("lambda_", lambda_, "Chebyshev Gradient")
            log_tensor_stats("diff_dot", diff_dot, "Chebyshev Gradient")
            log_tensor_stats("s_lambda", s_lambda, "Chebyshev Gradient")
            log_tensor_stats("hypervolume_estimates", hypervolume_estimates, "Chebyshev Gradient")
            logging.info(f"[Chebyshev Gradient] Estimated hypervolumes: {hypervolume_values.tolist()}")
            log_tensor_stats("hypervolume_values", hypervolume_values, "Chebyshev Gradient")
            logging.info(f"[Chebyshev Gradient] Relative errors: {rel_error.tolist()}")
            log_tensor_stats("rel_error", rel_error, "Chebyshev Gradient")
            logging.info(f"[Chebyshev Gradient] Total loss for gradient computation: {loss.item():.6f}")
            log_tensor_stats("loss", loss, "Chebyshev Gradient")
            logging.info(f"[Chebyshev Gradient] Gradient shape: {grads.shape}")
            log_tensor_stats("grads", grads, "Chebyshev Gradient")
            if torch.isnan(grads).any():
                grad_norms = torch.norm(grads, dim=1)
                log_tensor_stats("grad_norms (per point)", grad_norms, "Chebyshev Gradient")

            logging.info("=" * 80)

        return grads.reshape(batch_size, -1)

    def update_base_model(self):
        logging.info("[ChebyshevTrainer] Updating base model with fine model weights")
        self.base_model.load_state_dict(self.fine_model.state_dict())