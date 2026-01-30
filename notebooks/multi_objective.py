import argparse
import os

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genexp.lighting_diffusion import LightningDiffusion
from genexp.models import DiffusionModel
from genexp.sampling import VPSDE, EulerMaruyamaSampler

parser = argparse.ArgumentParser()
parser.add_argument("--sampling_set_n", type=int, default=5000, help="Number of points in the sampling set")
parser.add_argument("--mode", type=str, default="full", choices=["pretrain_only", "pretrain_and_after", "full"], 
                    help="Mode: pretrain_only (no sampling), pretrain_and_after (sample after), full (sample before+after)")

args = parser.parse_args()
sampling_set_n = args.sampling_set_n
mode = args.mode


mean = torch.zeros(2 * sampling_set_n)  # Centered at the origin
covariance = torch.eye(2 * sampling_set_n)  # Identity matrix as covariance
num_samples_gaussian = 5000  # Number of samples to generate

# Generate samples

dataset = torch.distributions.MultivariateNormal(mean, covariance).sample((num_samples_gaussian,))

network = nn.Sequential(
    nn.Linear(2 * sampling_set_n + 1, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2 * sampling_set_n),
)

sde = VPSDE(0.1, 12)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = DiffusionModel(network, sde).to(device)
pl_model = LightningDiffusion(model)

# Generate samples before finetuning
samples_before = None
if mode == "full":
    presampler = EulerMaruyamaSampler(model.to(device), data_shape=(2 * sampling_set_n,), device=device) 
    samples_before = []
    batch_size = 256
    num_samples = 10000
    for i in tqdm(range(num_samples // batch_size + 1)):
        trajs, ts = presampler.sample_trajectories(N=batch_size, T=1000, device=device)
        samples_before.append(trajs[-1].full.detach().cpu())
    samples_before = torch.vstack(samples_before)[:num_samples]

dl = DataLoader(TensorDataset(dataset), batch_size=128, shuffle=True)

trainer = Trainer(max_epochs=100)
trainer.fit(pl_model, dl)

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# Save model
torch.save(model.model.state_dict(), f"models/multi_obj_pretrained_{sampling_set_n}.pth")

# Load model
model.model.load_state_dict(torch.load(f"models/multi_obj_pretrained_{sampling_set_n}.pth", map_location=device))

# Visualize pre-trained density
samples_after = None
if mode in ["pretrain_and_after", "full"]:
    sampler = EulerMaruyamaSampler(model.to(device), data_shape=(2 * sampling_set_n,), device=device)
    batch_size = 256
    num_samples = 10000
    
    # Generate samples after pretraining
    samples_after = []
    for i in tqdm(range(num_samples // batch_size + 1)):
        trajs, ts = sampler.sample_trajectories(N=batch_size, T=1000, device=device)
        samples_after.append(trajs[-1].full.detach().cpu())
    samples_after = torch.vstack(samples_after)[:num_samples]

# Visualization
if mode != "pretrain_only":
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(dataset[:, 0].detach().cpu(), dataset[:, 1].detach().cpu(), alpha=0.5)
    ax[0].set_title("Data density")
    
    if samples_before is not None:
        ax[1].scatter(samples_before[:, 0].detach().cpu(), samples_before[:, 1].detach().cpu(), alpha=0.5)
    ax[1].set_title("Pre-trained model (before finetuning)")
    
    if samples_after is not None:
        ax[2].scatter(samples_after[:, 0].detach().cpu(), samples_after[:, 1].detach().cpu(), alpha=0.5)
    ax[2].set_title("Model after finetuning")
    
    plt.tight_layout()
    fig.savefig(f"figs/pretrained_density_objective_{sampling_set_n}.png")
    plt.close(fig)
