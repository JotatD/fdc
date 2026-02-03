from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from scipy.special import gamma

from genexp.trainers.objective import ZDT5Torch


def sample_lambda_first_quadrant():
    theta = (np.pi / 2) * np.random.rand()
    return torch.tensor([np.cos(theta), np.sin(theta)], dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZDT5 grid sampling and plots.")
    parser.add_argument(
        "--max-iterations", type=int, default=10000, help="Points per axis."
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

    problem = ZDT5Torch(n=2)
    samples = np.random.rand(5, 2).astype(np.float32)
    ref = torch.tensor([-11.0, -11.0])
    reward_set = problem.evaluate(torch.from_numpy(samples)).cpu()
    hv = Hypervolume(ref).compute(reward_set)
    print(hv)
    k = 2  # the reward space
    c_k = (np.pi ** (k / 2)) / (2**k * gamma(k / 2 + 1))

    max_iterations = args.max_iterations
    hypervolume_values = []
    hypervolume_approximations = []
    for i in range(max_iterations):
        # sample a vector from the 2d unit ball first quadrant
        lambda_ = sample_lambda_first_quadrant()

        s_lambdas = []
        s_lambda_approximations = []
        for rew in reward_set:
            obj = torch.relu((rew - ref) * (1 / lambda_)) ** k
            s_lambda = obj.min()
            s_lambdas.append(s_lambda)

            s_lambda_2 = -torch.logsumexp(-obj, dim=0)
            s_lambda_approximations.append(s_lambda_2)
            # print(f"s_lambda: {s_lambda} -- s_lambda_2: {s_lambda_2}")

        hypervolume_values.append(np.max(s_lambdas))
        hypervolume_approximations.append(
            torch.logsumexp(torch.tensor(s_lambda_approximations), dim=0)
        )
        # print(f"hypervalue: {hypervolume_values[-1]} -- approx: {hypervolume_approximations[-1]}")
    hypervolume_values = np.array(hypervolume_values)
    hypervolume_approximations = torch.stack(hypervolume_approximations).cpu().numpy()
    running_means = hypervolume_values.cumsum(axis=0) / np.arange(1, max_iterations + 1)
    hypervolume_approximations = hypervolume_approximations.cumsum(axis=0) / np.arange(
        1, max_iterations + 1
    )
    running_means *= c_k
    hypervolume_approximations *= c_k

    plt.figure(figsize=(8, 6))
    plt.plot(running_means, label="Running Mean of Hypervolume Values")
    plt.plot(
        hypervolume_approximations,
        label="Running Mean of Hypervolume Approximations",
        linestyle="--",
    )
    plt.plot(hv * np.ones_like(running_means), label="Hypervolume", linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Running Mean Hypervolume Value")
    plt.title("Running Mean of Hypervolume Values over Iterations")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "zdt5_hypervolume_running_mean.png")
    plt.close()


if __name__ == "__main__":
    main()
