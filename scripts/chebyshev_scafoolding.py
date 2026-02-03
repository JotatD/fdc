from __future__ import annotations

import logging
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from scipy.special import gamma

from genexp.trainers.objective import ZDT5Torch

# Add this near the top, before creating the trainer
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


EPS = 1e-6  # Small constant for numerical stability


def sample_lambda_first_quadrant(shape=(1,)):
    theta = (torch.pi / 2) * (
        torch.rand(*shape, dtype=torch.float32).clamp(EPS, 1 - EPS)
    )
    return torch.stack([torch.sin(theta), torch.cos(theta)], axis=-1)


def sample_lambda_full_quadrant(shape=(1,)):
    theta = (2 * torch.pi) * torch.rand(*shape, dtype=torch.float32)
    return torch.stack([torch.sin(theta), torch.cos(theta)], axis=-1)


def load_trace_debug(trace_path: str) -> dict:
    with open(trace_path, "rb") as f:
        trace = pickle.load(f)
    debug_buffer = trace.get("debug_buffer", [])
    if not debug_buffer:
        raise ValueError("Trace file does not include a debug_buffer.")
    breakpoint()
    return debug_buffer[-1]


def replay_from_trace(
    trace_path: str,
    device: str | None = None,
) -> dict:
    last = load_trace_debug(trace_path)
    device = torch.device(device or "cpu")

    x_before = torch.tensor(
        last["x_before"], dtype=torch.float32, device=device, requires_grad=True
    )
    grad = (
        torch.tensor(last["grad"], dtype=torch.float32, device=device)
        if last.get("grad") is not None
        else None
    )
    rewards = (
        torch.tensor(last["rewards"], dtype=torch.float32, device=device)
        if last.get("rewards") is not None
        else None
    )
    lambda_sample = (
        torch.tensor(last["lambda_sample"], dtype=torch.float32, device=device)
        if last.get("lambda_sample") is not None
        else None
    )

    return {
        "step": last.get("step"),
        "x_before": x_before,
        "grad": grad,
        "loss": last.get("loss"),
        "hypervolume": last.get("hypervolume"),
        "rewards": rewards,
        "lambda_sample": lambda_sample,
        "diff_dot_stats": last.get("diff_dot_stats"),
        "s_lambda_stats": last.get("s_lambda_stats"),
    }


if __name__ == "__main__":
    # Fix all random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("mps")
    problem = ZDT5Torch(n=2, device=device)
    test_number = 2
    if test_number == 1:
        # test 1
        x = sample_lambda_first_quadrant((1,)).to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=1e-3)

        # Get problem bounds
        lb, ub = problem.bounds()

        print(f"Step {0}, x: {x.detach().cpu().numpy()}, loss: {problem.evaluate(x)}")
        for step in range(1000):
            optimizer.zero_grad()
            loss = -problem.evaluate(x)[0, 1]
            loss.backward()
            optimizer.step()

            # Project x back to bounds after optimization step
            print(
                f"Step {step + 1}, x: {x.detach().cpu().numpy()}, loss: {problem.evaluate(x)}"
            )
    elif test_number == 2:
        # test 2
        x = (
            (sample_lambda_full_quadrant((1, 5)) * (1e-3))
            .to(device)
            .requires_grad_(True)
        )
        optimizer = torch.optim.Adam([x], lr=5e-3)
        batch_size = x.shape[0]
        sampling_set_n = x.shape[1]

        # Store history of all 5 points
        history = []

        # Store hypervolume history
        hv_history = []

        # Store debug information for last N steps
        debug_buffer = []
        max_debug_buffer = 10

        for step in range(300):
            # Save current positions
            history.append(x.detach().cpu().numpy()[0].copy())  # Shape: (5, 2)

            optimizer.zero_grad()
            k = 2
            ck = (np.pi ** (k / 2)) / (2**k * gamma(k / 2 + 1))
            ref = torch.tensor([-11.0, -11.0], device=device)
            rewards = problem.evaluate(x.reshape(-1, 2)).reshape(
                batch_size, sampling_set_n, -1
            )
            MC_times = 500

            hypervolume_values = []

            lambda_ = sample_lambda_first_quadrant((batch_size, MC_times)).to(
                x.device, dtype=torch.float32
            )
            lambda_ = lambda_.unsqueeze(2)
            lambda_ = lambda_.expand(-1, -1, sampling_set_n, -1)
            rewards = rewards.unsqueeze(1)
            rewards = rewards.expand(-1, MC_times, -1, -1)
            diff_dot = torch.relu(((rewards - ref) * 1 / lambda_)) ** k
            s_lambda = -torch.logsumexp(-diff_dot, dim=3)
            hypervolume_estimates = torch.logsumexp(s_lambda, dim=2)
            hypervolume_values = ck * hypervolume_estimates.mean(dim=1)

            # Compute gradients with respect to x_reshaped using autograd.grad
            loss = -hypervolume_values.sum()
            loss.backward()

            # Store debug info before optimizer step
            debug_info = {
                "step": step + 1,
                "x_before": x.detach().cpu().numpy().copy(),
                "grad": x.grad.detach().cpu().numpy().copy()
                if x.grad is not None
                else None,
                "loss": loss.item(),
                "hypervolume": hypervolume_values.detach().cpu().numpy().copy(),
                "rewards": rewards.detach().cpu().numpy().copy(),
                "lambda_s": lambda_.detach().cpu().numpy().copy(),  # Sample of lambda
                "diff_dot_stats": {
                    "min": diff_dot.min().item(),
                    "max": diff_dot.max().item(),
                    "mean": diff_dot.mean().item(),
                    "has_nan": torch.isnan(diff_dot).any().item(),
                    "values": diff_dot.detach().cpu().numpy().copy(),
                },
                "s_lambda_stats": {
                    "min": s_lambda.min().item(),
                    "max": s_lambda.max().item(),
                    "mean": s_lambda.mean().item(),
                    "has_nan": torch.isnan(s_lambda).any().item(),
                },
            }

            # Keep rolling buffer of last N steps
            debug_buffer.append(debug_info)
            if len(debug_buffer) > max_debug_buffer:
                debug_buffer.pop(0)

            optimizer.step()

            # Store hypervolume for plotting
            hv_history.append(hypervolume_values.detach().cpu().numpy()[0])

            # Check for NaN after optimizer step
            if torch.isnan(x).any():
                breakpoint()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                trace_file = f"nan_trace_{timestamp}.pkl"

                print("\n" + "=" * 80)
                print("NaN DETECTED! Saving debug trace...")
                print("=" * 80)

                # Collect comprehensive trace
                trace = {
                    "nan_step": step + 1,
                    "x_after_step": x.detach().cpu().numpy().copy(),
                    "nan_locations": torch.isnan(x).cpu().numpy(),
                    "debug_buffer": debug_buffer,  # Last N steps of debug info
                    "hyperparameters": {
                        "lr": optimizer.param_groups[0]["lr"],
                        "k": k,
                        "ref": ref.cpu().numpy(),
                        "MC_times": MC_times,
                        "ck": ck,
                    },
                }

                # Save trace to file
                with open(trace_file, "wb") as f:
                    pickle.dump(trace, f)

                print(f"\nTrace saved to: {trace_file}")
                print(f"\nNaN occurred at step: {step + 1}")
                print(f"x after step (with NaN):\n{trace['x_after_step']}")
                print(f"\nNaN locations:\n{trace['nan_locations']}")
                print("\nLast step before NaN:")
                print(f"  x: {debug_buffer[-1]['x_before']}")
                print(f"  grad: {debug_buffer[-1]['grad']}")
                print(f"  loss: {debug_buffer[-1]['loss']}")
                print(f"  hypervolume: {debug_buffer[-1]['hypervolume']}")
                print(f"  diff_dot stats: {debug_buffer[-1]['diff_dot_stats']}")
                print(f"  s_lambda stats: {debug_buffer[-1]['s_lambda_stats']}")
                print("\n" + "=" * 80)

                # Exit or continue based on preference
                raise RuntimeError(
                    f"NaN detected at step {step + 1}. Debug trace saved to {trace_file}"
                )

            print(
                f"Step {step + 1}, ||x||: {torch.diag(x.detach()[0] @ x.detach()[0].T)}, HV: {hypervolume_values.detach().cpu().numpy()}"
            )

        # Convert history to numpy array: (num_steps, 5, 2)
        history = np.array(history)
        hv_history = np.array(hv_history)

        # Plot trajectories and hypervolume progression
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Left plot: trajectories
        # Plot unit circle (Pareto set for ZDT5)
        theta = np.linspace(0, 2 * np.pi, 200)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax1.plot(
            circle_x,
            circle_y,
            "k--",
            linewidth=2,
            alpha=0.5,
            label="Unit Circle (Pareto Set)",
        )

        colors = ["red", "blue", "green", "orange", "purple"]

        for i in range(5):
            # Plot trajectory for point i
            trajectory = history[:, i, :]  # Shape: (num_steps, 2)
            ax1.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors[i],
                alpha=0.6,
                linewidth=2,
                label=f"Point {i + 1}",
            )
            # Mark starting point
            ax1.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                color=colors[i],
                s=100,
                marker="o",
                edgecolors="black",
                linewidths=2,
                zorder=5,
            )
            # Mark ending point
            ax1.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color=colors[i],
                s=100,
                marker="*",
                edgecolors="black",
                linewidths=2,
                zorder=5,
            )

        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.set_title("Trajectories of 5 Points During Optimization")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal", adjustable="box")

        # Right plot: hypervolume progression
        steps = np.arange(1, len(hv_history) + 1)
        ax2.plot(steps, hv_history, linewidth=2, color="darkblue")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Hypervolume")
        ax2.set_title("Hypervolume Progression")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("point_trajectories.png", dpi=150)
        print("\nTrajectory plot saved as 'point_trajectories.png'")
        plt.show()

    if test_number == 3:
        # test 3: replay from trace
        trace_path = "nan_trace_20260203_131922.pkl"
        replay_data = replay_from_trace(trace_path, device="cpu")
        print("Replayed data from trace:")
        for key, value in replay_data.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: Tensor of shape {value.shape}")
            else:
                print(f"{key}: {value}")

    elif test_number == 4:
        # test 4: sample many sets of 5 points on unit circle and compute hypervolume
        n_samples = 1000  # Number of sets to sample
        n_points = 5  # Points per set

        hypervolumes = []
        ref_point = torch.tensor([-11.0, -11.0], device=device)

        print(f"Sampling {n_samples} sets of {n_points} points on unit circle...")

        for i in range(n_samples):
            # Sample 5 points uniformly on unit circle
            angles = (
                torch.rand(n_points, dtype=torch.float32, device=device) * 2 * np.pi
            )
            x_samples = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

            # Evaluate objectives for these points
            objectives = problem.evaluate(x_samples)  # Shape: (5, 2)

            # Compute hypervolume using BoTorch
            # Note: BoTorch assumes maximization and reference point is the lower bound
            hv = Hypervolume(ref_point=ref_point)
            hv_value = hv.compute(objectives)

            hypervolumes.append(hv_value)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{n_samples} samples...")

        hypervolumes = np.array(hypervolumes)

        # Plot the distribution of hypervolumes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of hypervolume values
        ax1.hist(hypervolumes, bins=50, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Hypervolume")
        ax1.set_ylabel("Frequency")
        ax1.set_title(
            f"Distribution of Hypervolumes\n({n_samples} random sets of {n_points} points)"
        )
        ax1.axvline(
            hypervolumes.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {hypervolumes.mean():.2f}",
        )
        ax1.axvline(
            hypervolumes.max(),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Max: {hypervolumes.max():.2f}",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Line plot showing progression
        ax2.plot(range(1, n_samples + 1), hypervolumes, linewidth=0.5, alpha=0.6)
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Hypervolume")
        ax2.set_title("Hypervolume Values Across Samples")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("hypervolume_distribution.png", dpi=150)
        print("\nHypervolume statistics:")
        print(f"  Mean: {hypervolumes.mean():.4f}")
        print(f"  Std:  {hypervolumes.std():.4f}")
        print(f"  Min:  {hypervolumes.min():.4f}")
        print(f"  Max:  {hypervolumes.max():.4f}")
        print("\nPlot saved as 'hypervolume_distribution.png'")
        plt.show()
