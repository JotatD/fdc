from ..models import DiffusionModel
from .adjoint_matching_trajectory import AdjointMatchingTrajectoryFinetuningTrainer
import torch

class MaxEntTrajectoryTrainer(AdjointMatchingTrajectoryFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 lr, 
                 traj_samples_per_stage, 
                 data_shape, 
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 rew_type='score-matching',
                 base_model=None,
                 traj_len=100,
                 lmbda=1.0,
                 clip_grad_norm=None,
                 **kwargs):
        
        if rew_type == 'score-matching':
            print("Using score-matching reward, lambda:", lmbda)
            eps = 0.05
            grad_reward_fn = lambda x: -base_model.score_func(x, torch.tensor(eps, device=x.device).float().detach())*lmbda
            # compute gradient of f_k along trajectory
            grad_f_k_trajectory = lambda x, t: -base_model.score_func(x, torch.tensor(t, device=x.device).float().detach())*lmbda 
            self.lmbda = lmbda
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, grad_f_k_trajectory, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm,**kwargs)
    

    def update_reward(self):
        self.grad_reward_fn = lambda x: -self.base_model.score_func(x, torch.tensor(0.0, device=x.device).float().detach())*self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
