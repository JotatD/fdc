from ..models import DiffusionModel
from .adjoint_matching import AdjointMatchingFinetuningTrainer
import torch
from genexp.trainers.adjoint_matching import AdjointMatchingFinetuningTrainer, sample_trajectories_ddpm

class OedKlTrainer(AdjointMatchingFinetuningTrainer):
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
                 num_traj_MC=15,
                 traj_len=100,
                 lambda_reg_ridge=0.1,
                 lmbda=1.0,
                 clip_grad_norm=None):
        
        if rew_type == 'score-matching':
            print("Using first variation of OED objective and KL divergence as reward, lambda:", lmbda)
            # linear kernel case \Phi(x) = x
            # oed_grad(x) - alpha_div * (score_base(x) - score_pre(x))
            grad_reward_fn = lambda x: lmbda * self.compute_oed_grad(x, base_model, pre_trained_model, alpha_div, num_traj_MC, lambda_reg_ridge) - alpha_div * (base_model.score_func(x, torch.tensor(0, device=x.device).float().detach()) - pre_trained_model.score_func(x, torch.tensor(0, device=x.device).float().detach()))
            self.lmbda = lmbda
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm)
    

    def compute_oed_grad(self, x, base_model, pre_trained_model, alpha_div, num_traj_MC, lambda_reg_ridge):
        # compute the gradient of the OED objective
        # sample batch of N data points x from current model to estimate expectation of outer product of features
        with torch.no_grad():
            x0 = torch.randn(num_traj_MC, 2, device='cpu') # sample 15 trajectories
            trajs1 = sample_trajectories_ddpm(base_model, x0, 100) 
            trajs1 = trajs1[0]
            x_sampled = trajs1[:, -1, :]
            outer_products = x_sampled.unsqueeze(2) * x_sampled.unsqueeze(1)
            MC_expectation = outer_products.mean(dim=0)

        # print('MC_expectation.shape', MC_expectation.shape)
        # print('x.shape', x.shape)
        matrix_inverse = torch.linalg.inv(MC_expectation + lambda_reg_ridge * torch.eye(2))
        matrix_inverse_sqrt = matrix_inverse @ matrix_inverse
        matrix_inverse_sqrt = matrix_inverse_sqrt.to(x.device)
        final_gradient = 2 * x @ matrix_inverse_sqrt
        # print('final_gradient', final_gradient)
        return final_gradient

    def update_reward(self):
        self.grad_reward_fn = lambda x: -self.base_model.score_func(x, torch.tensor(0.0, device=x.device).float().detach())*self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
