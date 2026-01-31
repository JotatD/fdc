from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.special import gamma

from ..models import DiffusionModel
from ..sampling import Sampler
from .adjoint_matching import AMTrainerFlow
from .objective import ZDT5Torch

# Add this near the top, before creating the trainer
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


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

        def grad_reward_fn(x):
            cbsh = self.compute_chebyshev_grad(x)
            div = self.divergence(x)
            return self.lmbda * cbsh - div

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
        return self.alpha_div * (score_base - score_pretrained)

    def compute_chebyshev_grad(self, x, MC_times=500):
        # x is of size (batch_size, sampling_set_n*2)
        breakpoint()
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, -1, 2)
        sampling_set_n = x_reshaped.shape[1]
        x_reshaped.requires_grad_(True)
        # (batch_size, sampling_set_n, 2)
        rewards = self.problem.evaluate(x_reshaped.reshape(-1, 2)).reshape(
            batch_size, sampling_set_n, -1
        )

        hypervolume_values = []

        # actual_volumes = []
        # for i in range(batch_size):
        #     actual_volume = Hypervolume(self.ref).compute(rewards[i].detach().cpu())
        #     actual_volumes.append(actual_volume)
        # actual_volumes = torch.tensor(actual_volumes).to(x.device)

        lambda_ = sample_lambda_first_quadrant((batch_size, MC_times)).to(x.device)
        lambda_ = lambda_.unsqueeze(2)
        lambda_ = lambda_.expand(-1, -1, sampling_set_n, -1)
        rewards = rewards.unsqueeze(1)
        rewards = rewards.expand(-1, MC_times, -1, -1)
        diff_dot = torch.relu(((rewards - self.ref) * 1 / lambda_)) ** self.k
        s_lambda = -torch.logsumexp(-diff_dot, dim=3)
        hypervolume_estimates = torch.logsumexp(s_lambda, dim=2)
        hypervolume_values = self.ck * hypervolume_estimates.mean(dim=1)

        # Compute gradients with respect to x_reshaped using autograd.grad
        loss = hypervolume_values.sum()
        grads = torch.autograd.grad(loss, x_reshaped)[0]

        return grads.reshape(batch_size, -1)

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())
