from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from genexp.trainers.objective import ZDT5Torch


def make_grid(side_length: float, grid_size: int) -> np.ndarray:
    half = 0.5 * side_length
    xs = np.linspace(-half, half, grid_size, dtype=np.float32)
    ys = np.linspace(-half, half, grid_size, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([xv.ravel(), yv.ravel()], axis=1)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="ZDT5 grid sampling and plots.")
    parser.add_argument("--grid-size", type=int, default=300, help="Points per axis.")
    parser.add_argument("--side-length", type=float, default=6, help="Square side length.")
    parser.add_argument(
        "--circle-tol",
        type=float,
        default=0.05,
        help="Tolerance for marking points near the unit circle.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(args.side_length, args.grid_size)
    X = torch.from_numpy(grid)

    problem = ZDT5Torch(n=2)
    F = problem.evaluate(X).cpu().numpy()

    r = np.sqrt(grid[:, 0] ** 2 + grid[:, 1] ** 2)
    near_circle = np.abs(r - 1.0) <= args.circle_tol

    # Plot 1: decision space
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(grid[:, 0], grid[:, 1], s=6, alpha=0.6, label="Grid points")
    ax1.scatter(
        grid[near_circle, 0],
        grid[near_circle, 1],
        s=18,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=0.8,
        label="Near unit circle",
    )
    ax1.set_title("Decision Space: Uniform Grid")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_aspect("equal", "box")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(out_dir / "zdt5_decision_space.png")
    plt.close(fig1)

    # Plot 2: objective space heatmap
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    sc = ax2.scatter(
        F[:, 0],
        F[:, 1],
        c=F[:, 1],
        cmap="viridis",
        s=8,
        alpha=0.7,
    )
    ax2.scatter(
        F[near_circle, 0],
        F[near_circle, 1],
        s=22,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=0.8,
        label="From near unit circle",
    )
    ax2.set_title("Objective Space: ZDT5 Values")
    ax2.set_xlabel("f1")
    ax2.set_ylabel("f2")
    ax2.grid(True, alpha=0.3)
    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label("f2 value")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(out_dir / "zdt5_objective_space.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
