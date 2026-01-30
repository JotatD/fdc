from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = torch.device("cpu")


@dataclass
class ZDTProblemTorch:
    n: int

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lb = torch.zeros(self.n, dtype=torch.float32, device=DEVICE)
        ub = torch.ones(self.n, dtype=torch.float32, device=DEVICE)
        return lb, ub

    def validate(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=torch.float32, device=DEVICE)
        if X.ndim != 2 or X.shape[1] != self.n:
            raise ValueError(f"X must have shape (N, {self.n}), got {tuple(X.shape)}")
        return X

    def f1(self, X: torch.Tensor) -> torch.Tensor:
        return X[:, 0]

    def g(self, X: torch.Tensor) -> torch.Tensor:
        if self.n < 2:
            raise ValueError("ZDT problems require n >= 2.")
        return 1.0 + 9.0 * torch.sum(X[:, 1:], dim=1) / (self.n - 1)

    def h(self, f1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f2(self, X: torch.Tensor) -> torch.Tensor:
        f1 = self.f1(X)
        g = self.g(X)
        return g * self.h(f1, g)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        X = self.validate(X)
        f1 = self.f1(X)
        f2 = self.f2(X)
        # Maximization variant: negate minimization objectives.
        return torch.stack([-f1, -f2], dim=1)

    def sample_pareto_set_X(self, N: int) -> torch.Tensor:
        """Sample N points from the Pareto set in decision space.
        Returns shape (N, n) tensor of decision variables."""
        rng = np.random.default_rng()
        X = np.zeros((N, self.n), dtype=np.float32)
        X[:, 0] = rng.random(N)
        return torch.from_numpy(X).to(dtype=torch.float32, device=DEVICE)

    def sample_pareto_set(self, N: int) -> torch.Tensor:
        """Sample N points from the Pareto set and evaluate them.
        Returns shape (N, 2) tensor of [f1, f2] values."""
        X_tensor = self.sample_pareto_set_X(N)
        return self.evaluate(X_tensor)

    def sample_pareto_front(self, N: int) -> torch.Tensor:
        """Sample N points analytically from the Pareto front.
        Returns shape (N, 2) tensor of [f1, f2] values."""
        f1 = torch.linspace(0.0, 1.0, N, dtype=torch.float32, device=DEVICE)
        g = torch.ones_like(f1)
        f2 = g * self.h(f1, g)
        # Maximization variant: negate minimization objectives.
        return torch.stack([-f1, -f2], dim=1)


class ZDT1Torch(ZDTProblemTorch):
    def h(self, f1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sqrt(f1 / g)


class ZDT2Torch(ZDTProblemTorch):
    def h(self, f1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.0 - (f1 / g) ** 2


class ZDT3Torch(ZDTProblemTorch):
    def h(self, f1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sqrt(f1 / g) - (f1 / g) * torch.sin(10.0 * torch.pi * f1)


class ZDT4Torch(ZDTProblemTorch):
    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lb = torch.full((self.n,), -5.0, dtype=torch.float32, device=DEVICE)
        ub = torch.full((self.n,), 5.0, dtype=torch.float32, device=DEVICE)
        lb[0] = 0.0
        ub[0] = 1.0
        return lb, ub

    def g(self, X: torch.Tensor) -> torch.Tensor:
        if self.n < 2:
            raise ValueError("ZDT problems require n >= 2.")
        Xi = X[:, 1:]
        return 1.0 + 10.0 * (self.n - 1) + torch.sum(
            Xi ** 2 - 10.0 * torch.cos(4.0 * torch.pi * Xi), dim=1
        )

    def h(self, f1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sqrt(f1 / g)


class ZDT5Torch(ZDTProblemTorch):
    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lb = torch.full((self.n,), -1.0, dtype=torch.float32, device=DEVICE)
        ub = torch.full((self.n,), 1.0, dtype=torch.float32, device=DEVICE)
        return lb, ub

    def f1(self, X: torch.Tensor) -> torch.Tensor:
        # Map x1 from [-1, 1] to [0, 1] so the Pareto front matches ZDT1.
        return 0.5 * (X[:, 0] + 1.0)

    def g(self, X: torch.Tensor) -> torch.Tensor:
        if self.n < 2:
            raise ValueError("ZDT problems require n >= 2.")
        r2 = X[:, 0] ** 2 + X[:, 1] ** 2
        return 1.0 + 9.0 * torch.abs(r2 - 1.0)

    def h(self, f1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.sqrt(f1 / g)

    def sample_pareto_set_X(self, N: int) -> torch.Tensor:
        rng = np.random.default_rng()
        theta = rng.random(N) * 2.0 * np.pi
        X = np.zeros((N, self.n), dtype=np.float32)
        X[:, 0] = np.cos(theta)
        X[:, 1] = np.sin(theta)
        return torch.from_numpy(X).to(dtype=torch.float32, device=DEVICE)


def sample_uniform(problem: ZDTProblemTorch, N: int, rng: np.random.Generator = None) -> torch.Tensor:
    rng = rng or np.random.default_rng()
    lb, ub = problem.bounds()
    lb_np = lb.cpu().numpy()
    ub_np = ub.cpu().numpy()
    samples = rng.random((N, problem.n)).astype(np.float32) * (ub_np - lb_np) + lb_np
    return torch.from_numpy(samples).to(dtype=torch.float32, device=DEVICE)


def main(output_path: str = "zdt_pareto.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    problems = [
        ("ZDT1", ZDT1Torch(n=2)),
        ("ZDT2", ZDT2Torch(n=2)),
        ("ZDT3", ZDT3Torch(n=2)),
        ("ZDT4", ZDT4Torch(n=2)),
        ("ZDT5", ZDT5Torch(n=2)),
    ]

    N_samples = 200
    N_front = 400
    
    for name, prob in problems:
        # Sample from Pareto set and front
        Xp_sampled = prob.sample_pareto_set_X(N_samples).cpu().numpy()
        Fp_sampled = prob.evaluate(torch.from_numpy(Xp_sampled)).cpu().numpy()
        Fp_front = prob.sample_pareto_front(N_front).cpu().numpy()
        
        # Create a new figure for each problem
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Pareto frontier (objective space)
        axes[0].plot(Fp_front[:, 0], Fp_front[:, 1], label='Pareto Front')
        axes[0].scatter(Fp_sampled[:, 0], Fp_sampled[:, 1], s=20, label='Sampled Points')
        axes[0].set_title(f'{name} - Pareto Frontier (Objective Space, maximize)')
        axes[0].set_xlabel('f1 (maximize)')
        axes[0].set_ylabel('f2 (maximize)')
        axes[0].grid(True)
        axes[0].legend()

        # Plot Pareto set (parameter space) - reconstruct x1 from f1
        axes[1].scatter(Xp_sampled[:, 0], Xp_sampled[:, 1], s=20)
        axes[1].set_title(f'{name} - Pareto Set (Parameter Space)')
        axes[1].set_xlabel('x1')
        axes[1].set_ylabel('x2')
        axes[1].grid(True)

        # Save the figure
        output_file = f'./data/{name}_pareto.png'
        plt.tight_layout()
        fig.savefig(output_file)
        plt.close(fig)
        print(f"Saved: {output_file}")

    return output_path


if __name__ == "__main__":
    out = main("./data/zdt_pareto.png")
    print(out, "saved:", os.path.exists(out))
