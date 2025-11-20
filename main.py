import numpy as np
import matplotlib.pyplot as plt

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.objective import objective
from simulation.gradient_based.optimizer.optimizer_nlopt import optimizer


def main():
    weights = np.ones((ConfigSim.nx, ConfigSim.ny, 2))
    weights[..., 0] = 0.5
    weights[..., 1] = 1
    loss_hist = []
    for beta in [8, 32, 64, 256]:
        weights, loss = optimizer(weights, objective, beta)

        loss_hist += loss
        plt.plot(loss_hist)
        plt.savefig(f"plots/loss_{beta}.png.png")
        plt.close()


if __name__ == "__main__":
    main()
