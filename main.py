import numpy as np
import matplotlib.pyplot as plt

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.objective import objective_f
from simulation.gradient_based.optimizer.optimizer_nlopt import optimizer
from util.structure_pillars import get_positions


def main():
    weights = np.random.uniform(0, 1, ConfigSim.nx*ConfigSim.ny)
    positions = get_positions((ConfigSim.rho_size[0], ConfigSim.rho_size[1]), ConfigSim.size_pillars, ConfigSim.nx,  ConfigSim.ny)
    loss_hist = []
    for beta in [1e3]:
        weights, loss = optimizer(weights, positions, objective_f, beta)

        loss_hist += loss
        plt.plot(loss_hist)
        plt.savefig(f"plots/loss_{beta}.png.png")
        plt.close()


if __name__ == "__main__":
    main()
