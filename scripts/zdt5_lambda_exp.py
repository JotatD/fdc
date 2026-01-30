from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from genexp.trainers.objective import ZDT5Torch


def sample_unit_disk(n):
    theta = 2 * np.pi * np.random.rand(n)
    r = np.sqrt(np.random.rand(n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def main() -> None:
    parser = argparse.ArgumentParser(description="ZDT5 grid sampling and plots.")
    parser.add_argument("--max-iterations", type=int, default=5000, help="Points per axis.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    problem = ZDT5Torch(n=2)
    samples = np.random.rand(5, 2).astype(np.float32)
    ref = np.array([[-1.0, -1.0]])
    reward_set = problem.evaluate(torch.from_numpy(samples)).cpu().numpy()
    max_iterations = args.max_iterations
    chebyshev_values = []
    for i in range(max_iterations):
        #sample a vector from the 2d unit ball 
        lambda_ = torch.from_numpy(sample_unit_disk(1)).squeeze(0).to(dtype=torch.float32)
        
        s_lambdas = []
        for rew in reward_set:
            obj = (rew - ref).dot(lambda_.cpu().numpy())
            s_lambda = obj.min()
            s_lambdas.append(s_lambda)
        chebyshev_values.append(np.max(s_lambdas))
    chebyshev_values = np.array(chebyshev_values)
    running_means = chebyshev_values.cumsum(axis=0) / np.arange(1, max_iterations + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(running_means, label="Running Mean of Chebyshev Values")
    plt.xlabel("Iterations")
    plt.ylabel("Running Mean Chebyshev Value")
    plt.title("Running Mean of Chebyshev Values over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "zdt5_chebyshev_running_mean.png")
    plt.close()

if __name__ == "__main__":
    main()
