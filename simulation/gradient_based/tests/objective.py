import numpy as np
import matplotlib.pyplot as plt
import tidy3d.web as web

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.objective import objective
from simulation.gradient_based.simulation import simulation


def test():
    weights = np.random.rand(ConfigSim.nx, ConfigSim.ny) * ConfigSim.nz
    # weights = np.zeros_like(weights) * 150
    # weights[0, 0] = 150
    v_em = objective(weights, 1, 100)
    print(v_em)
    return


if __name__ == "__main__":
    test()