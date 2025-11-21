import numpy as np
import matplotlib.pyplot as plt
import tidy3d.web as web
import autograd.numpy as anp

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.simulation import simulation
from util.structure_pillars import get_positions


def test():
    weights = np.ones((ConfigSim.nx, ConfigSim.ny))
    # weights[..., 0] = 1
    # weights[..., 1] = 0.1
    weights = np.random.uniform(0, 1, (ConfigSim.nx*ConfigSim.ny))
    positions = get_positions((ConfigSim.rho_size[0], ConfigSim.rho_size[1]), ConfigSim.size_pillars, ConfigSim.nx,  ConfigSim.ny)
    sim = simulation(weights, positions, 1e3)

    sim.plot_eps(z=ConfigSim.mon_pos_z, freq=ConfigSim.freq0)
    plt.show()
    sim.plot_eps(y=0, freq=ConfigSim.freq0)
    plt.show()

    job = web.Job(simulation=sim, task_name="test")
    estimated_cost = web.estimate_cost(job.task_id)

    print(estimated_cost)

    return


if __name__ == "__main__":
    test()