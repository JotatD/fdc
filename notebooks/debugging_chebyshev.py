from __future__ import annotations

import copy
from logging import config

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from genexp.models import DiffusionModel
from genexp.sampling import VPSDE, EulerMaruyamaSampler
from genexp.trainers.chebyshev import ChebyshevTrainer


def sample_lambda_first_quadrant(shape=(1,)):
    theta = (torch.pi / 2) * torch.rand(*shape)
    return torch.stack([torch.sin(theta), torch.cos(theta)], axis=-1)


def create_network(sampling_set_n):
    """Create the neural network architecture."""
    return nn.Sequential(
        nn.Linear(2 * sampling_set_n + 1, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 2 * sampling_set_n),
    )
        
if __name__ == "__main__":
    sampling_set_n = 5
    network = create_network(sampling_set_n)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    sde = VPSDE(0.1, 12)
    model = DiffusionModel(network, sde).to(device)
    model.model.load_state_dict(
        torch.load(f"models/multi_obj_finetuned_{sampling_set_n}.pth", map_location=device)
    )

    batch_size = 256
    num_samples = 10000

    # Initialize sample storage
    samples_before = None
    samples_after = None
    samples_fdc = None
    
    sampler = EulerMaruyamaSampler(
        model.to(device), data_shape=(2 * sampling_set_n,), device=device
    )
    
    config_path = "../configs/example_fdc.yaml"
    config = OmegaConf.load(config_path)
    
    fdc_trainer = ChebyshevTrainer(
        config=config,
        model=copy.deepcopy(model),
        base_model=copy.deepcopy(model),
        pre_trained_model=copy.deepcopy(model),
        device=device,
        sampler=sampler,
        ref=torch.tensor([-11.0, -11.0]).to(device),
    )
    

    x_orig = torch.load("input_debug.pt").to(device)
    # x_orig = x_orig.requires_grad_(True)
    # x = 2 * x_orig
    # x.sum().backward()
    # print(x_orig.grad)  # This will show all 2s
    # print(x.grad)  # This will be None (or all 1s if you use retain_grad)
    
    
    fdc_trainer.compute_chebyshev_grad(
        x = x_orig,
        MC_times=500
    )
