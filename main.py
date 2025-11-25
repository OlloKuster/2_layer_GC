import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.objective import objective_f
from simulation.gradient_based.optimizer.optimizer_nlopt import optimizer
from util.structure_pillars import get_positions


def main():
    td.config.logging_level = "ERROR"
    weights = np.random.uniform(0, 1, (ConfigSim.nx, ConfigSim.ny))
    # weights = np.ones_like(weights) * 0.5
    positions = get_positions((ConfigSim.rho_size[0], ConfigSim.rho_size[1]), ConfigSim.size_pillars, ConfigSim.nx,  ConfigSim.ny)
    loss_hist = []
    for beta in [100]:
        weights, loss = optimizer(weights, positions, objective_f, beta)

        loss_hist += loss
        plt.plot(loss_hist)
        plt.savefig(f"plots/loss_{beta}.png.png")
        plt.close()


if __name__ == "__main__":
    main()
