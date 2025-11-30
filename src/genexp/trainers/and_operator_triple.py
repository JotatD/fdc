from ..models import DiffusionModel
from .adjoint_matching_trajectory import AdjointMatchingTrajectoryFinetuningTrainer
import torch

class AndOperatorTrainerTriple(AdjointMatchingTrajectoryFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 lr,  
                 traj_samples_per_stage, 
                 data_shape, 
                 grad_reward = None,
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 rew_type='score-matching',
                 base_model=None,
                 pre_trained_model_1=None,
                 pre_trained_model_2=None,
                 pre_trained_model_3=None,
                 alpha_div=[1.0,1.0,1.0],
                 traj_len=100,
                 lmbda=1.0,
                 clip_grad_norm=None,
                 running_cost=False):
        
        if rew_type == 'score-matching':
            print("Using first variation of double KL as reward, lambda:", lmbda)
            # grad_reward =  lmbda * (-3*score_base + score_pre1 + score_pre2 + score_pre3)
            eps = 0.1
            if grad_reward is not None:
                grad_reward_fn = lambda x: lmbda * (grad_reward(x) - (alpha_div[0] + alpha_div[1] + alpha_div[2]) * self.base_model.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[0] * pre_trained_model_1.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[1] *pre_trained_model_2.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[2] *pre_trained_model_3.score_func(x, torch.tensor(eps, device=x.device).float().detach()))
            else:
                grad_reward_fn = lambda x: lmbda * (- (alpha_div[0] + alpha_div[1] + alpha_div[2]) * self.base_model.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[0] * pre_trained_model_1.score_func(x, torch.tensor(eps, device=x.device).float().detach()) + alpha_div[1] *pre_trained_model_2.score_func(x, torch.tensor(0, device=x.device).float().detach()) + alpha_div[2] *pre_trained_model_3.score_func(x, torch.tensor(eps, device=x.device).float().detach()))

            # compute gradient of f_k along trajectory
            grad_f_k_trajectory = lambda x, t: lmbda * (- (alpha_div[0] + alpha_div[1] + alpha_div[2]) * self.base_model.score_func(x, torch.tensor(t, device=x.device).float().detach()) + alpha_div[0] * pre_trained_model_1.score_func(x, torch.tensor(t, device=x.device).float().detach()) + alpha_div[1] *pre_trained_model_2.score_func(x, torch.tensor(t, device=x.device).float().detach()) + alpha_div[2] *pre_trained_model_3.score_func(x, torch.tensor(t, device=x.device).float().detach()))
            self.lmbda = lmbda
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, grad_f_k_trajectory, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm, running_cost=running_cost)
    

    def update_reward(self):
        self.grad_reward_fn = lambda x: -self.base_model.score_func(x, torch.tensor(0.0, device=x.device).float().detach())*self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
