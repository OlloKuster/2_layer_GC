import pyswarms
import numpy as np
import jax
import h5py
import matplotlib.pyplot as plt

from simulation.global_sim.config import ConfigSim
from simulation.global_sim.objective import objective

jax.config.update("jax_enable_x64", True)
def optimizer():
    def objective_f(params):
        foms = []
        for j in range(params.shape[0]):
            fom, E, eps, mon_pos = objective(params[1].reshape((ConfigSim.ny, ConfigSim.nx)))
            plt.imshow(eps[mon_pos].T, origin='lower', cmap='binary')
            plt.imshow(np.abs(E[0])[mon_pos].T, origin='lower', cmap='magma')
            plt.savefig(f"/scratch/local/okuster/Code/01_Side_Projects/3D_Coupler_Barz/results/wg/field_{ConfigSim.i:04}.png")
            plt.close()
            plt.imshow(eps[:, eps.shape[1]//2].T, origin='lower', cmap='binary')
            plt.imshow(np.abs(E[1])[:, eps.shape[1]//2].T, origin='lower', cmap='magma')
            plt.savefig(f"/scratch/local/okuster/Code/01_Side_Projects/3D_Coupler_Barz/results/middle/field_{ConfigSim.i:04}.png")
            plt.close()
            plt.imshow(eps[:, eps.shape[1]//2].T, origin='lower', cmap='binary')
            plt.savefig(f"/scratch/local/okuster/Code/01_Side_Projects/3D_Coupler_Barz/results/middle/eps_{ConfigSim.i:04}.png")
            plt.close()
            ConfigSim.i += 1
            foms.append(-fom)
        return np.array(foms)

    init_weights = np.ones((2, ConfigSim.nx * ConfigSim.ny))
    bounds = (0 * np.ones((ConfigSim.nx * ConfigSim.ny)), 2 * np.ones(ConfigSim.nx * ConfigSim.ny))

    options = {"c1": 1, "c2": 1, "w": 0.5}

    optimizer = pyswarms.global_best.GlobalBestPSO(
        n_particles=1, dimensions=init_weights.shape[1], options=options, bounds=bounds, init_pos=init_weights
    )

    fom, w_opt = optimizer.optimize(objective_f, iters=1000)

    with h5py.File(f"/scratch/local/okuster/Code/01_Side_Projects/3D_Coupler_Barz/results/data.h5", "w") as f:
        grp = f.create_group("Results")
        grp.create_dataset("w_opt", data=w_opt)
        f.close()

    return fom


optimizer()