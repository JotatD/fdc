from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import gamma

from genexp.trainers.objective import ZDT5Torch

# Add this near the top, before creating the trainer
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def sample_lambda_first_quadrant(shape=(1,)):
    theta = (torch.pi / 2) * torch.rand(*shape, dtype=torch.float32)
    return torch.stack([torch.sin(theta), torch.cos(theta)], axis=-1)


def sample_lambda_full_quadrant(shape=(1,)):
    theta = (2 * torch.pi) * torch.rand(*shape, dtype=torch.float32)
    return torch.stack([torch.sin(theta), torch.cos(theta)], axis=-1)


if __name__ == "__main__":
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
        for step in range(10000):
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

        for step in range(200):
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
            optimizer.step()
            print(
                f"Step {step + 1}, ||x||: {torch.diag(x.detach()[0] @ x.detach()[0].T)}, HV: {hypervolume_values.detach().cpu().numpy()}"
            )

        # Convert history to numpy array: (num_steps, 5, 2)
        history = np.array(history)

        # Plot trajectories
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot unit circle (Pareto set for ZDT5)
        theta = np.linspace(0, 2 * np.pi, 200)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(
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
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors[i],
                alpha=0.6,
                linewidth=2,
                label=f"Point {i + 1}",
            )
            # Mark starting point
            ax.scatter(
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
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color=colors[i],
                s=100,
                marker="*",
                edgecolors="black",
                linewidths=2,
                zorder=5,
            )

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Trajectories of 5 Points During Optimization")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig("point_trajectories.png", dpi=150)
        print("\nTrajectory plot saved as 'point_trajectories.png'")
        plt.show()
