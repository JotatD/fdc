from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from ..models import DiffusionModel
from ..sampling import Sampler
from .adjoint_matching import AMTrainerFlow
from .objective import ZDT5Torch


def sample_lambda_first_quadrant():
    theta = (np.pi / 2) * np.random.rand()
    return torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float32)

class ChebyshevTrainer(AMTrainerFlow):
    def __init__(self,
                 config: OmegaConf,
                 model: DiffusionModel,
                 base_model: DiffusionModel,
                 pre_trained_model: DiffusionModel,
                 device: Optional[torch.device] = None,
                 sampler: Optional[Sampler] = None,
                 ref: Optional[torch.Tensor] = None
                 ):

        rew_type = config.get('rew_type', 'score_matching')
        self.alpha_div = config.get('alpha_div', 1.)
        self.lmbda = lmbda = config.get('lmbda', 1.)
        self.problem = ZDT5Torch(n=2, device=device if device is not None else torch.device("cpu"))
        self.pre_trained_model = pre_trained_model
        self.ref = ref

        
        if rew_type == 'score-matching':
            def grad_reward_fn(x):
                return lmbda * self.compute_chebyshev_grad(x) - self.divergence(x)
            self.lmbda = lmbda
        else:
            raise NotImplementedError
        
        super().__init__(config.adjoint_matching, model, base_model, grad_reward_fn, None, device=device, sampler=sampler)

    def divergence(self, x):
        zero = torch.tensor(0, device=x.device).float().detach()
        score_base = self.base_model.score_func(x, zero)
        score_pretrained = self.pre_trained_model.score_func(x, zero)
        self.alpha_div * (score_base - score_pretrained)
        
    def compute_chebyshev_grad(self, x, MC_times=500):
        # sample batch of N data points x from current model to estimate expectation of outer product of features
        x = x.reshape(2, -1)
        rewards = self.problem.evaluate(x)
        
        hypervolume_values = []
        for i in range(MC_times):
        #sample a vector from the 2d unit ball first quadrant   
            lambda_ = sample_lambda_first_quadrant()
            
            s_lambdas = []
            for rew in rewards:
                obj = torch.relu((rew - self.ref) * (1/lambda_))**self.k
                s_lambda = obj.min()
                s_lambdas.append(s_lambda)
            hypervolume_values.append(np.max(s_lambdas))
        hypervolume_values = np.array(hypervolume_values)
        
        
        return final_gradient

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())


