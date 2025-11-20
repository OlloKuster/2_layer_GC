import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import jax

from simulation.global_sim.config import ConfigSim
from simulation.global_sim.simulation import simulation
from simulation.global_sim.structure_pillars import generate_pillars


def test():
    jax.config.update("jax_enable_x64", True)

    matrix = np.random.random((ConfigSim.ny, ConfigSim.nx))

    E, eps = simulation(matrix)

    plt.imshow(np.abs(eps)[:, eps.shape[1]//2].T, origin='lower', cmap='binary')
    plt.imshow(np.abs(E[1])[:, eps.shape[1]//2].T, origin='lower', cmap='magma', alpha=0.5)
    plt.show()
    p = pv.Plotter()
    data = pv.wrap(eps)
    p.add_volume(data, cmap='binary')
    p.show()

if __name__ == "__main__":
    test()