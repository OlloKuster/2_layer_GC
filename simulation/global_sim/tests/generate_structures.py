import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from simulation.global_sim.structure_pillars import generate_pillars


def test():
    matrix = np.random.random((20, 20))*2

    rho = generate_pillars(matrix, 10, 10, 4)
    plt.imshow(rho[..., 0].T, origin='lower')
    plt.show()

    p = pv.Plotter()
    data = pv.wrap(rho)
    p.add_volume(data, cmap='binary')
    p.show()


if __name__ == "__main__":
    test()