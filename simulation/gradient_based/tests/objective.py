import numpy as np
import matplotlib.pyplot as plt
import tidy3d.web as web
from autograd import value_and_grad

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.objective import objective_f
from simulation.gradient_based.simulation import simulation
from util.structure_pillars import get_positions


def test():
    weights = np.random.uniform(0, 1, ConfigSim.nx*ConfigSim.ny)
    positions = get_positions((ConfigSim.rho_size[0], ConfigSim.rho_size[1]), ConfigSim.size_pillars)
    # v_em = objective(weights, 1, positions, 1)
    objective = objective_f(positions, 1, 100)
    dJ = value_and_grad(objective)
    val, grad = dJ(weights)
    print(val)
    print(grad)
    return


if __name__ == "__main__":
    test()