import jax.numpy as jnp

from simulation.global_sim.config import ConfigSim
from simulation.global_sim.simulation import simulation


def objective(weights):
    E, eps = simulation(weights)
    mon_pos_x = -int(jnp.ceil(ConfigSim.resolution*(ConfigSim.wg_length - 0.5)))
    mon_size_y = int(jnp.ceil(ConfigSim.wg_width * ConfigSim.resolution))*2
    mon_pos_z_start = int(jnp.ceil((ConfigSim.thickness_substrate - ConfigSim.rho_size[2]) * ConfigSim.resolution))
    mon_pos_z_end = int(jnp.ceil((ConfigSim.thickness_substrate + 1.5*ConfigSim.rho_size[2]) * ConfigSim.resolution))
    E_flux = jnp.abs(jnp.sum(E[0][mon_pos_x,
                                  int(jnp.ceil(eps.shape[1] - mon_size_y) / 2):int(jnp.ceil((eps.shape[1] + mon_size_y) / 2)),
                                  mon_pos_z_start:mon_pos_z_end]))
    return E_flux, E, eps, mon_pos_x