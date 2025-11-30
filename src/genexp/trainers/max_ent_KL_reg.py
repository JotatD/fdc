from ..models import DiffusionModel
from .adjoint_matching import AdjointMatchingFinetuningTrainer
import torch

class MaxEntKLRegTrainer(AdjointMatchingFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 lr, 
                 traj_samples_per_stage, 
                 data_shape, 
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 rew_type='score-matching',
                 base_model=None,
                 pre_trained_model=None,
                 alpha_div=1.0,
                 traj_len=100,
                 lmbda=1.0,
                 clip_grad_norm=None,
                 epsilon=0, # timestep offset for reward function
                 **kwargs):
        
        if rew_type == 'score-matching':
            print("Using score-matching reward, lambda:", lmbda)
            grad_reward_fn = lambda x: lmbda * (-(base_model.score_func(x, torch.tensor(epsilon, device=x.device).float().detach())) - alpha_div * (base_model.score_func(x, torch.tensor(epsilon, device=x.device).float().detach()) - pre_trained_model.score_func(x, torch.tensor(epsilon, device=x.device).float().detach())))
            self.lmbda = lmbda
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm,
                         **kwargs)
    
    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
