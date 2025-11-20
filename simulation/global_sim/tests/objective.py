import numpy as np
import jax

from simulation.global_sim.config import ConfigSim
from simulation.global_sim.objective import objective


def test():
    jax.config.update("jax_enable_x64", True)

    matrix = np.random.random((ConfigSim.ny, ConfigSim.nx))

    obj = objective(matrix)

    print(obj)


if __name__ == "__main__":
    test()
