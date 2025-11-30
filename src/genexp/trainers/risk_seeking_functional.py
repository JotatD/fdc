from ..models import DiffusionModel
from .adjoint_matching import AdjointMatchingFinetuningTrainer
import torch
from maxentdiff.trainers.adjoint_matching import AdjointMatchingFinetuningTrainer, sample_trajectories_ddpm
# from scipy.optimize import minimize
# from autograd import grad  
import torch.nn.functional as F

class RiskSeekingKlTrainer(AdjointMatchingFinetuningTrainer):
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
                 lmbda=1.0,
                 alpha_cvar=0.95,
                 reward_function=None,
                 clip_grad_norm=None):
        
        self.lmbda = lmbda
        self.alpha_cvar = alpha_cvar
        self.num_traj_MC = num_traj_MC
        self.traj_len = traj_len

        if rew_type == 'score-matching':
            grad_reward_fn = lambda x: lmbda * (self.compute_superquantile_grad(x, base_model, reward_function) - alpha_div * (base_model.score_func(x, torch.tensor(0, device=x.device).float().detach()) - pre_trained_model.score_func(x, torch.tensor(0, device=x.device).float().detach())))
        else:
            raise NotImplementedError
        super().__init__(model, grad_reward_fn, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, clip_grad_norm=clip_grad_norm)
    

    def compute_superquantile_grad(self, x, base_model, reward_function):
        alpha_cvar = self.alpha_cvar
        num_traj_MC = self.num_traj_MC
        gamma = 10.0

        # samples based estimation of beta_star
        with torch.no_grad():
            x0 = torch.randn(num_traj_MC, 2, device='cpu')
            trajs1 = sample_trajectories_ddpm(base_model, x0, self.traj_len) 
            trajs1 = trajs1[0]
            samples = trajs1[:, -1, :]
            # beta_star = self.estimate_beta_star_torch(samples, reward_function, alpha_cvar=alpha_cvar, gamma=gamma)
            sample_loss = reward_function(samples)
            beta_star = torch.quantile(sample_loss, self.alpha_cvar)

        # activate gradient tracking for x
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        torch.set_grad_enabled(True)

        # SMOOTHED CVaR
        # x_loss_val = reward_function(x)
        # beta_star = beta_star.to(x.device)
        # factor = 1.0 / ((1.0 - alpha_cvar) * (1.0 + torch.exp(-gamma * (x_loss_val - beta_star))))
        # grad_r = torch.autograd.grad(x_loss_val.sum(), x, create_graph=False, retain_graph=True)[0]
        # factor = factor.unsqueeze(-1)
        # final_gradient = factor * grad_r

        # UN-SMOOTHED CVaR
        loss_x = reward_function(x)          
        # print("loss_x.shape:", loss_x.shape)                     
        mask = (loss_x > beta_star).float()
        grad_L = torch.autograd.grad(loss_x.sum(), x)[0]
        # print("grad_L.shape:", grad_L.shape)
        factor = mask / ((1.0 - self.alpha_cvar) * mask.numel())    
        # print("factor", factor, "factor.shape:", factor.shape)
        final_gradient = factor.unsqueeze(-1) * grad_L 
        # print("final_gradient.shape:", final_gradient.shape)

        return final_gradient


    def estimate_beta_star_torch(self, samples, reward_function, alpha_cvar, gamma, init_beta=0.0, num_iterations=100):
        beta = torch.tensor([init_beta], dtype=torch.float32, requires_grad=True, device=samples.device)

        def objective_function_torch():
            samples_loss_val = reward_function(samples)
            sp = F.softplus(samples_loss_val - beta, beta=gamma)
            val = beta + (1.0/(1.0 - alpha_cvar)) * torch.mean(sp)
            return val

        optimizer = torch.optim.LBFGS([beta], max_iter=num_iterations, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = objective_function_torch()
            loss.backward()
            return loss

        optimizer.step(closure)

        # eval for debugging 
        est_CVaR = objective_function_torch()
        print("estimated CVaR during opt:", est_CVaR.item())

        return beta.detach()

    def update_reward(self):
        self.grad_reward_fn = lambda x: -self.base_model.score_func(x, torch.tensor(0.0, device=x.device).float().detach())*self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
