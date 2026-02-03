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
from .adjoint_matching import AMTrainerFlow
from .objective import ZDT5Torch


def sample_lambda_first_quadrant(shape=(1,)):
    theta = (torch.pi / 2) * torch.rand(*shape)
    return torch.stack([torch.sin(theta), torch.cos(theta)], axis=-1)


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
            print(f"[grad_reward_fn] Chebyshev grad norm: {torch.norm(cbsh).item():.6f}, Divergence norm: {torch.norm(div).item():.6f}, Total norm: {torch.norm(total_grad).item():.6f}")
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
        # x is of size (batch_size, sampling_set_n*2)
        logging.info("=" * 80)
        logging.info("[Chebyshev Gradient] Starting computation...")
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, -1, 2)
        sampling_set_n = x_reshaped.shape[1]
        logging.info(f"[Chebyshev Gradient] Batch size: {batch_size}, Sampling set size: {sampling_set_n}, MC samples: {MC_times}")
        
        # Compute norms of the 2D points
        x_norms = torch.norm(x_reshaped, dim=2)  # Shape: (batch_size, sampling_set_n)
        logging.info(f"[Chebyshev Gradient] Input x norms - min: {x_norms.min().item():.6f}, max: {x_norms.max().item():.6f}, mean: {x_norms.mean().item():.6f}")
        
        # Check for NaN in input
        if torch.isnan(x).any():
            logging.error("[Chebyshev Gradient] NaN detected in input x!")
            logging.error(f"[Chebyshev Gradient] NaN locations: {torch.isnan(x).sum().item()} out of {x.numel()} elements")
        
        # Make sure x_reshaped requires grad BEFORE any operations
        x_reshaped = x_reshaped.clone().requires_grad_(True)
        
        # (batch_size, sampling_set_n, 2)
        rewards = self.problem.evaluate(x_reshaped.reshape(-1, 2)).reshape(
            batch_size, sampling_set_n, -1
        )
        logging.info(f"[Chebyshev Gradient] Rewards shape: {rewards.shape}")
        logging.info(f"[Chebyshev Gradient] Rewards stats - min: {rewards.min().item():.6f}, max: {rewards.max().item():.6f}, mean: {rewards.mean().item():.6f}")
        
        # Check for NaN in rewards
        if torch.isnan(rewards).any():
            logging.error("[Chebyshev Gradient] NaN detected in rewards!")
            logging.error(f"[Chebyshev Gradient] NaN locations in rewards: {torch.isnan(rewards).sum().item()} out of {rewards.numel()} elements")

        # Compute actual hypervolumes for verification (detached, not part of grad computation)
        with torch.no_grad():
            actual_volumes = []
            for i in range(batch_size):
                actual_volume = Hypervolume(self.ref.cpu()).compute(rewards[i].detach().cpu())
                actual_volumes.append(actual_volume)
            actual_volumes = torch.tensor(actual_volumes).to(x.device)
            logging.info(f"[Chebyshev Gradient] Actual hypervolumes: {actual_volumes.tolist()}")
            logging.info(f"[Chebyshev Gradient] Actual HV stats - min: {actual_volumes.min().item():.6f}, max: {actual_volumes.max().item():.6f}, mean: {actual_volumes.mean().item():.6f}")

        lambda_ = sample_lambda_first_quadrant((batch_size, MC_times)).to(x.device)
        logging.debug(f"[Chebyshev Gradient] Lambda shape: {lambda_.shape}, stats - min: {lambda_.min().item():.4f}, max: {lambda_.max().item():.4f}")
        
        # Use unsqueeze and expand carefully to maintain gradient flow
        lambda_ = lambda_.unsqueeze(2).expand(-1, -1, sampling_set_n, -1)
        rewards_expanded = rewards.unsqueeze(1).expand(-1, MC_times, -1, -1)
        
        # Compute with proper gradient tracking
        diff = (rewards_expanded - self.ref) / lambda_
        diff_dot = torch.relu(diff) ** self.k
        logging.debug(f"[Chebyshev Gradient] diff_dot shape: {diff_dot.shape}, stats - min: {diff_dot.min().item():.6f}, max: {diff_dot.max().item():.6f}")
        
        # Check for NaN in diff_dot
        if torch.isnan(diff_dot).any():
            logging.error("[Chebyshev Gradient] NaN detected in diff_dot!")
            logging.error(f"[Chebyshev Gradient] NaN locations in diff_dot: {torch.isnan(diff_dot).sum().item()} out of {diff_dot.numel()} elements")
        
        s_lambda = -torch.logsumexp(-diff_dot, dim=3)
        logging.debug(f"[Chebyshev Gradient] s_lambda shape: {s_lambda.shape}, stats - min: {s_lambda.min().item():.6f}, max: {s_lambda.max().item():.6f}")
        
        # Check for NaN in s_lambda
        if torch.isnan(s_lambda).any():
            logging.error("[Chebyshev Gradient] NaN detected in s_lambda!")
            logging.error(f"[Chebyshev Gradient] NaN locations in s_lambda: {torch.isnan(s_lambda).sum().item()} out of {s_lambda.numel()} elements")
        
        hypervolume_estimates = torch.logsumexp(s_lambda, dim=2)
        logging.debug(f"[Chebyshev Gradient] hypervolume_estimates shape: {hypervolume_estimates.shape}, stats - min: {hypervolume_estimates.min().item():.6f}, max: {hypervolume_estimates.max().item():.6f}")
        
        # Check for NaN in hypervolume_estimates
        if torch.isnan(hypervolume_estimates).any():
            logging.error("[Chebyshev Gradient] NaN detected in hypervolume_estimates!")
            logging.error(f"[Chebyshev Gradient] NaN locations in hypervolume_estimates: {torch.isnan(hypervolume_estimates).sum().item()} out of {hypervolume_estimates.numel()} elements")
        
        hypervolume_values = self.ck * hypervolume_estimates.mean(dim=1)
        logging.info(f"[Chebyshev Gradient] Estimated hypervolumes: {hypervolume_values.tolist()}")
        logging.info(f"[Chebyshev Gradient] Estimated HV stats - min: {hypervolume_values.min().item():.6f}, max: {hypervolume_values.max().item():.6f}, mean: {hypervolume_values.mean().item():.6f}")
        
        # Check for NaN in hypervolume_values
        if torch.isnan(hypervolume_values).any():
            logging.error("[Chebyshev Gradient] NaN detected in final hypervolume_values!")
            logging.error(f"[Chebyshev Gradient] NaN locations in hypervolume_values: {torch.isnan(hypervolume_values).sum().item()} out of {hypervolume_values.numel()} elements")
        
        # Compare estimated vs actual
        with torch.no_grad():
            rel_error = torch.where(
                actual_volumes > 1e-6,
                torch.abs(hypervolume_values.detach() - actual_volumes) / actual_volumes,
                torch.abs(hypervolume_values.detach() - actual_volumes)
            )
            logging.info(f"[Chebyshev Gradient] Relative errors: {rel_error.tolist()}")
            logging.info(f"[Chebyshev Gradient] Mean relative error: {rel_error.mean().item():.6f}, Max relative error: {rel_error.max().item():.6f}")

            if rel_error.mean().item() > 10:
                print("yoyoyo")
        
        # Compute gradients with respect to x_reshaped using autograd.grad
        loss = hypervolume_values.sum()
        logging.info(f"[Chebyshev Gradient] Total loss for gradient computation: {loss.item():.6f}")
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            logging.error("[Chebyshev Gradient] NaN detected in loss value!")
        
        grads = torch.autograd.grad(loss, x_reshaped, create_graph=False)[0]
        logging.info(f"[Chebyshev Gradient] Gradient shape: {grads.shape}")
        logging.info(f"[Chebyshev Gradient] Gradient stats - min: {grads.min().item():.6f}, max: {grads.max().item():.6f}, mean: {grads.mean().item():.6f}, norm: {torch.norm(grads).item():.6f}")
        
        # Check for NaN in gradients
        if torch.isnan(grads).any():
            logging.error("[Chebyshev Gradient] NaN detected in gradients!")
            logging.error(f"[Chebyshev Gradient] NaN locations in grads: {torch.isnan(grads).sum().item()} out of {grads.numel()} elements")
            
            # Compute per-point gradient norms to see which points have NaN gradients
            grad_norms = torch.norm(grads, dim=2)  # Shape: (batch_size, sampling_set_n)
            logging.error(f"[Chebyshev Gradient] Gradient norms per point - min: {grad_norms.min().item():.6f}, max: {grad_norms.max().item():.6f}, mean: {grad_norms.mean().item():.6f}")
    
        logging.info("=" * 80)

        return grads.reshape(batch_size, -1)

    def update_base_model(self):
        logging.info("[ChebyshevTrainer] Updating base model with fine model weights")
        self.base_model.load_state_dict(self.fine_model.state_dict())
