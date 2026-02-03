import argparse
import copy
import logging
import os

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genexp.lighting_diffusion import LightningDiffusion
from genexp.models import DiffusionModel
from genexp.sampling import VPSDE, EulerMaruyamaSampler
from genexp.trainers.chebyshev import ChebyshevTrainer
from genexp.utils import seed_everything


def setup_logging():
    """Configure logging to file only (no console output)."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/multi_objective.log"),
        ],
    )
    logging.info("Logging is set up.")


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


def generate_samples(model, device, sampling_set_n, batch_size, num_samples, reshape=False):
    """Generate samples from the model.
    
    Args:
        model: The diffusion model to sample from
        device: Device to run sampling on
        sampling_set_n: Number of points in the sampling set
        batch_size: Batch size for sampling
        num_samples: Total number of samples to generate
        reshape: If True, reshape output to (-1, sampling_set_n, 2)
    
    Returns:
        Generated samples tensor
    """
    sampler = EulerMaruyamaSampler(
        model.to(device), data_shape=(2 * sampling_set_n,), device=device
    )
    samples = []
    for i in tqdm(range(num_samples // batch_size + 1)):
        
        trajs, ts = sampler.sample_trajectories(N=batch_size, T=1000, device=device)
        samples.append(trajs[-1].full.detach().cpu())
    samples = torch.vstack(samples)[:num_samples]
    
    if reshape:
        samples = samples.reshape(-1, sampling_set_n, 2)
    
    return samples


def pretrain_model(model, sampling_set_n):
    """Pretrain the model on Gaussian samples."""
    mean = torch.zeros(2 * sampling_set_n)
    covariance = torch.eye(2 * sampling_set_n)
    num_samples_gaussian = 5000
    dataset = torch.distributions.MultivariateNormal(mean, covariance).sample(
        (num_samples_gaussian,)
    )
    pl_model = LightningDiffusion(model)
    trainer = Trainer(max_epochs=100)
    dl = DataLoader(TensorDataset(dataset), batch_size=128, shuffle=True)
    trainer.fit(pl_model, dl)
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.model.state_dict(), f"models/multi_obj_pretrained_{sampling_set_n}.pth")


def finetune_model(model, device, sampling_set_n, config_path="../configs/example_fdc.yaml"):
    """Fine-tune the model using flow density control."""
    config = OmegaConf.load(config_path)
    sampler = EulerMaruyamaSampler(
        model.to(device), data_shape=(2 * sampling_set_n,), device=device
    )
    seed_everything(config.seed)

    logging.info("Next is the flow density control fine-tuning...")
    fdc_trainer = ChebyshevTrainer(
        config=config,
        model=copy.deepcopy(model),
        base_model=copy.deepcopy(model),
        pre_trained_model=copy.deepcopy(model),
        device=device,
        sampler=sampler,
        ref=torch.tensor([-11.0, -11.0]).to(device),
    )

    for k in tqdm(range(config.num_md_iterations)):
        for i in range(config.adjoint_matching.num_iterations):
            am_dataset = fdc_trainer.generate_dataset()
            fdc_trainer.finetune(am_dataset, steps=config.adjoint_matching.finetune_steps)
        fdc_trainer.update_base_model()

    torch.save(fdc_trainer.fine_model.model.state_dict(), f"models/multi_obj_finetuned_{sampling_set_n}.pth")


def plot_results(args, sampling_set_n, samples_before=None, samples_after=None, samples_fdc=None):
    """Plot the results based on available samples."""
    # Determine how many plots we need
    num_plots = 0
    if args.sample_before_pretrain and samples_before is not None:
        num_plots += 1
    if args.sample_after_pretrain and samples_after is not None:
        num_plots += 1
    if args.sample_after_finetune and samples_fdc is not None:
        num_plots += sampling_set_n

    if num_plots == 0:
        return

    # Create figure with appropriate number of subplots
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    ax = ax.flatten() if num_plots > 1 else [ax]

    plot_idx = 0

    # Plot before pretrain
    if args.sample_before_pretrain and samples_before is not None:
        ax[plot_idx].scatter(samples_before[:, 0], samples_before[:, 1], alpha=0.5)
        ax[plot_idx].set_title("Before pretrain")
        plot_idx += 1

    # Plot after pretrain
    if args.sample_after_pretrain and samples_after is not None:
        ax[plot_idx].scatter(samples_after[:, 0], samples_after[:, 1], alpha=0.5)
        ax[plot_idx].set_title("After pretrain")
        plot_idx += 1

    # Plot each point after finetuning
    if args.sample_after_finetune and samples_fdc is not None:
        for i in range(sampling_set_n):
            ax[plot_idx].scatter(
                samples_fdc[:, i, 0].detach().cpu(),
                samples_fdc[:, i, 1].detach().cpu(),
                alpha=0.5,
            )
            ax[plot_idx].set_title(f"Point {i + 1} after finetune")
            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(ax)):
        ax[i].axis("off")

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig(f"figs/fdc_density_objective_{sampling_set_n}.png")
    plt.close(fig)


def main():
    """Main execution function."""
    setup_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampling_set_n", type=int, default=5, help="Number of points in the sampling set"
    )
    parser.add_argument("-sample-before-pretrain", action="store_true", help="Sample before pretraining")
    parser.add_argument("-pretrain", action="store_true", help="Pretrain the model")
    parser.add_argument("-sample-after-pretrain", action="store_true", help="Sample after pretraining")
    parser.add_argument("-finetune", action="store_true", help="Finetune the model")
    parser.add_argument("-sample-after-finetune", action="store_true", help="Sample after finetuning")

    args = parser.parse_args()
    sampling_set_n = args.sampling_set_n

    # Setup model and device
    network = create_network(sampling_set_n)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    sde = VPSDE(0.1, 12)
    model = DiffusionModel(network, sde).to(device)

    batch_size = 256
    num_samples = 10000

    # Initialize sample storage
    samples_before = None
    samples_after = None
    samples_fdc = None

    # Execute pipeline based on arguments
    if args.sample_before_pretrain:
        samples_before = generate_samples(model, device, sampling_set_n, batch_size, num_samples)

    if args.pretrain:
        pretrain_model(model, sampling_set_n)

    # Load pretrained model
    model.model.load_state_dict(
        torch.load(f"models/multi_obj_pretrained_{sampling_set_n}.pth", map_location=device)
    )

    if args.sample_after_pretrain:
        samples_after = generate_samples(model, device, sampling_set_n, batch_size, num_samples)

    if args.finetune:
        finetune_model(model, device, sampling_set_n)

    # Load finetuned model
    model.model.load_state_dict(
        torch.load(f"models/multi_obj_finetuned_{sampling_set_n}.pth", map_location=device)
    )

    if args.sample_after_finetune:
        samples_fdc = generate_samples(model, device, sampling_set_n, batch_size, 10, reshape=True)

    # Plot results
    plot_results(args, sampling_set_n, samples_before, samples_after, samples_fdc)


if __name__ == "__main__":
    main()
