import argparse
import copy
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genexp.lighting_diffusion import LightningDiffusion
from genexp.models import DiffusionModel
from genexp.sampling import VPSDE, EulerMaruyamaSampler
from genexp.trainers.chebyshev import ChebyshevTrainer
from genexp.trainers.objective import ZDT5Torch
from genexp.utils import (
    get_config,
    init_wandb,
    log_to_wandb,
    seed_everything,
    set_aggressive_logging,
)


class DotDict(dict):
    """Dictionary that supports both dictionary and attribute access."""
    
    def __getattr__(self, key):
        try:
            value = self[key]
            # Recursively convert nested dicts to DotDict
            if isinstance(value, dict) and not isinstance(value, DotDict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def setup_logging():
    """Configure logging to file only (no console output)."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"logs/multi_objective_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
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


def finetune_model(model, device, sampling_set_n, config, args):
    """Fine-tune the model using flow density control."""
    sampler = EulerMaruyamaSampler(
        model.to(device), data_shape=(2 * sampling_set_n,), device=device
    )
    seed_everything(config['seed'])

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
    
    num_samples = config.get("num_samples", 10000)
    batch_size = config.get("batch_size", 256)
    ref_point = torch.tensor([-11.0, -11.0])

    for k in tqdm(range(config.num_md_iterations), desc="Mirror Descent Iterations"):
        for i in range(config.adjoint_matching.num_iterations):
            am_dataset = fdc_trainer.generate_dataset()
            fdc_trainer.finetune(am_dataset, steps=config.adjoint_matching.finetune_steps)
        
        # Generate samples and compute hypervolumes after each MD iteration
        samples = generate_samples(model, device, sampling_set_n, batch_size, num_samples, reshape=True)
        
        if args.use_wandb:
            # Compute hypervolumes for this MD iteration
            hypervolumes = compute_hypervolumes(samples, ref_point, device)
            
            # Log hypervolume statistics
            log_to_wandb({
                "md_iteration": k,
                "hypervolume/mean": hypervolumes.mean(),
                "hypervolume/std": hypervolumes.std(),
                "hypervolume/min": hypervolumes.min(),
                "hypervolume/max": hypervolumes.max(),
                "hypervolume/median": np.median(hypervolumes),
            }, step=k)
            
            logging.info(f"MD Iter {k}: HV Mean={hypervolumes.mean():.4f}, Std={hypervolumes.std():.4f}")
        
        fdc_trainer.update_base_model()

    torch.save(fdc_trainer.fine_model.model.state_dict(), f"models/multi_obj_finetuned_{sampling_set_n}.pth")


def compute_hypervolumes(samples, ref_point, device):
    """Compute hypervolumes for each sample.
    
    Args:
        samples: Tensor of shape (num_samples, sampling_set_n, 2)
        ref_point: Reference point for hypervolume computation
        device: Device to run computation on
    
    Returns:
        Array of hypervolumes for each sample
    """
    problem = ZDT5Torch(n=2, device=device)
    num_samples = samples.shape[0]
    
    hypervolumes = []
    for i in tqdm(range(num_samples), desc="Computing hypervolumes"):
        # Evaluate objectives for this sample's points
        sample_points = samples[i].to(device)  # (sampling_set_n, 2)
        rewards = problem.evaluate(sample_points)  # (sampling_set_n, 2)
        
        # Compute hypervolume using botorch
        hv = Hypervolume(ref_point.cpu()).compute(rewards.detach().cpu())
        hypervolumes.append(hv)
    
    return np.array(hypervolumes)


def plot_hypervolumes(args, sampling_set_n, samples_after, samples_fdc, device):
    """Compute and plot hypervolume distributions.
    
    Args:
        args: Command line arguments
        sampling_set_n: Number of points in the sampling set
        samples_after: Samples after pretraining (optional)
        samples_fdc: Samples after finetuning (optional)
        device: Device to run computation on
    """
    if not (args.sample_after_pretrain or args.sample_after_finetune):
        return
    
    ref_point = torch.tensor([-11.0, -11.0])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if samples_after is not None:
        hv_after = compute_hypervolumes(samples_after, ref_point, device)
        ax.hist(hv_after, bins=50, alpha=0.6, label='After Pretrain', density=True)
    else:
        #check if file exists and load it
        pretrain_samples_path = f"models/samples_after_pretrain_{sampling_set_n}.pt"
        if os.path.exists(pretrain_samples_path):
            samples_after = torch.load(pretrain_samples_path)
            hv_after = compute_hypervolumes(samples_after, ref_point, device)
            ax.hist(hv_after, bins=50, alpha=0.6, label='After Pretrain', density=True)
        
    
    if args.sample_after_finetune and samples_fdc is not None:
        logging.info("Computing hypervolumes for finetuned samples...")
        hv_fdc = compute_hypervolumes(samples_fdc, ref_point, device)
        ax.hist(hv_fdc, bins=50, alpha=0.6, label='After Finetune', density=True)
        logging.info(f"Finetune HV - Mean: {hv_fdc.mean():.4f}, Std: {hv_fdc.std():.4f}")
    
    ax.set_xlabel('Hypervolume')
    ax.set_ylabel('Density')
    ax.set_title('Hypervolume Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f"figs/hypervolume_distribution_{sampling_set_n}.png")
    plt.close(fig)
    logging.info(f"Hypervolume plot saved to figs/hypervolume_distribution_{sampling_set_n}.png")


def plot_results(args, sampling_set_n, samples_before=None, samples_after=None, samples_fdc=None):
    """Plot the results based on available samples."""
    # Determine how many plots we need
    num_plots = 0
    if args.sample_before_pretrain and samples_before is not None:
        num_plots += 1
    if args.sample_after_pretrain and samples_after is not None:
        num_plots += sampling_set_n
    if args.sample_after_finetune and samples_fdc is not None:
        num_plots += sampling_set_n

    if num_plots == 0:
        return

    # Create figure with appropriate number of subplots
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))
    if isinstance(ax, np.ndarray):
        ax = ax.flatten().tolist()
    else:
        ax = [ax]

    plot_idx = 0

    # Plot before pretrain
    if args.sample_before_pretrain and samples_before is not None:
        ax[plot_idx].scatter(samples_before[:, 0], samples_before[:, 1], alpha=0.5)
        ax[plot_idx].set_title("Before pretrain")
        plot_idx += 1

    # Plot after pretrain
    if args.sample_after_pretrain and samples_after is not None:
        for i in range(sampling_set_n):
            ax[plot_idx].scatter(
                samples_after[:, i, 0].detach().cpu(),
                samples_after[:, i, 1].detach().cpu(),
                alpha=0.5,
            )
            ax[plot_idx].set_title(f"Point {i + 1} after pretrain")
            ax[plot_idx].set_xlim(-15, 15)
            ax[plot_idx].set_ylim(-15, 15)
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
            ax[plot_idx].set_xlim(-15, 15)
            ax[plot_idx].set_ylim(-15, 15)
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
    parser.add_argument("--sampling_set_n", type=int, default=5, help="Number of points in the sampling set")
    parser.add_argument("-sample-before-pretrain", action="store_true", help="Sample before pretraining")
    parser.add_argument("-pretrain", action="store_true", help="Pretrain the model")
    parser.add_argument("-sample-after-pretrain", action="store_true", help="Sample after pretraining")
    parser.add_argument("-finetune", action="store_true", help="Finetune the model")
    parser.add_argument("-sample-after-finetune", action="store_true", help="Sample after finetuning")
    parser.add_argument("-aggressive-logging", action="store_true", help="Enable aggressive tensor logging for debugging")
    parser.add_argument("--index", type=int, default=0, help="Index for config exploration (if applicable)")
    parser.add_argument("-use-wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()
    
    explore_chebyshev = {
        "seed": [2],
        "gamma_falloff": [0.0],
        "gamma": [0.1],
        "epsilon": [0.005],
        "num_md_iterations": [5],  # number of outer (mirror descent) steps
        "beta": [0.8],
        "first_variation_norm_clipping": [1., 10., 100., np.inf],
        "num_samples": [10000],
        "gradient_flipping": [True, False],
    }
    
    explore_adjoint_matching = {
            "num_iterations": [50],  # how often the adjoint matching is run after the bandit step (if use_bandit is set to false this is the only optimization step)
            "batch_size": [32],  # batch_size is both used for number samples - sampled per batch an then for updates
            "clip_grad_norm": [0.4],  # float - if 0.0 no clipping is done
            "clip_loss": [1e5],  # float - if 0.0 no clipping is done
            "lr": [0.01],
            "finetune_steps": [10],
            "sampling_num_samples": [256],
            "sampling_num_integration_steps": [40]
            }

    # Set aggressive logging flag
    set_aggressive_logging(args.aggressive_logging)
    
        
    exp_config, adj_config = get_config(explore_chebyshev, explore_adjoint_matching, args.index)
    
    # Convert to DotDict for both dictionary and attribute access
    config = DotDict(exp_config)
    config["adjoint_matching"] = DotDict(adj_config)
    num_samples = config.get("num_samples", 10000)
    batch_size = config.get("batch_size", 256)

    # Initialize wandb if requested
    if args.use_wandb:
        wandb_config = {
            "sampling_set_n": args.sampling_set_n,
            **config
        }
        init_wandb(wandb_config, project_name="multi-objective-diffusion")

    logging.info("Starting multi-objective diffusion experiment...")
    logging.info(f"Arguments: {args}")
    sampling_set_n = args.sampling_set_n

    # Setup model and device
    network = create_network(sampling_set_n)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    sde = VPSDE(0.1, 12)
    model = DiffusionModel(network, sde).to(device)

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
        samples_after = generate_samples(model, device, sampling_set_n, batch_size, num_samples, reshape=True)
        
        # save the samples
        torch.save(samples_after, f"models/samples_after_pretrain_{sampling_set_n}.pt")
        logging.info(f"Saved samples after pretrain to models/samples_after_pretrain_{sampling_set_n}.pt")

    if args.finetune:
        finetune_model(model, device, sampling_set_n, config, args)

    # Load finetuned model
    model.model.load_state_dict(
        torch.load(f"models/multi_obj_finetuned_{sampling_set_n}.pth", map_location=device)
    )

    if args.sample_after_finetune:
        samples_fdc = generate_samples(model, device, sampling_set_n, batch_size, num_samples, reshape=True)

    # Plot results
    plot_results(args, sampling_set_n, samples_before, samples_after, samples_fdc)
    
    # Compute and plot hypervolumes
    plot_hypervolumes(args, sampling_set_n, samples_after, samples_fdc, device)


if __name__ == "__main__":
    main()
