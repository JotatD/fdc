from ..models import DiffusionModel
from .adjoint_matching import AdjointMatchingFinetuningTrainer
import torch
import copy

class MaxEntTrainer(AdjointMatchingFinetuningTrainer):
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
                 kl_penalty=0.,
                 epsilon=1. / 100.,
                 **kwargs):
        
        if rew_type == 'score-matching':
            print("Using score-matching reward, lambda:", lmbda, ' alpha: ', kl_penalty)
            base_model = model if base_model is None else base_model
            self.base_base_model = copy.deepcopy(base_model)
            self.lmbda = lmbda
            self.alpha = kl_penalty
            self.epsilon = epsilon
            base_score = lambda x: self.base_model.score_func(x, torch.tensor(epsilon, device=x.device).float().detach())
            base_base_score = lambda x: self.base_base_model.score_func(x, torch.tensor(epsilon, device=x.device).float().detach())
            grad_reward_fn = lambda x: self.lmbda * (-base_score(x) - self.alpha * (base_score(x) - base_base_score(x)))
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm,**kwargs)
    

    def update_reward(self):
        base_score = lambda x: self.base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())
        base_base_score = lambda x: self.base_base_model.score_func(x, torch.tensor(self.epsilon, device=x.device).float().detach())
        self.grad_reward_fn = lambda x: self.lmbda * (-base_score(x) - self.alpha * (base_score(x) - base_base_score(x)))
        

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
