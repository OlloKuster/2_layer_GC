import jax.numpy as jnp
import numpy as np
import jaxwell

from simulation.global_sim.config import ConfigSim
from simulation.global_sim.structure_pillars import generate_pillars


def gaussian_3d(size, k, amplitude, mu, sigma):
    x = np.linspace(0, size[0], size[0])
    y = np.linspace(0, size[1], size[1])
    z = np.linspace(0, size[2], size[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    real_part = np.exp(- 0.5 / sigma ** 2 * ((xx - mu[0]) ** 2 + (yy - mu[1]) ** 2 + (zz - mu[2]) ** 2))
    imaginary_part = np.exp(1j * (k[0] * (xx - mu[0]) + k[1] * (yy - mu[1])))
    return amplitude * real_part * imaginary_part


def simulation(weights):
    rho = generate_pillars(weights,
                           ConfigSim.rho_size[0] * ConfigSim.resolution,
                           ConfigSim.rho_size[1] * ConfigSim.resolution / 2,
                           ConfigSim.rho_size[2] * ConfigSim.resolution)

    eps = (ConfigSim.refr_index[2] ** 2 - ConfigSim.refr_index[0] ** 2) * rho + ConfigSim.refr_index[0] ** 2
    eps = jnp.concat((eps, jnp.flip(eps, axis=1)), axis=1)

    buffer = int(jnp.ceil(ConfigSim.buffer * ConfigSim.resolution))
    wg_length = int(jnp.ceil(ConfigSim.wg_length * ConfigSim.resolution))
    wg_width = int(jnp.ceil(ConfigSim.wg_width * ConfigSim.resolution))
    thickness_sub = int(jnp.ceil(ConfigSim.thickness_substrate * ConfigSim.resolution))

    eps = jnp.pad(eps,
                  [(buffer, wg_length)] + [(buffer, buffer)] + [(0, buffer)],
                  mode='constant',
                  constant_values=ConfigSim.refr_index[0] ** 2)
    eps = jnp.pad(eps,
                  [(0, 0)] * 2 + [(thickness_sub, 0)],
                  mode='constant',
                  constant_values=ConfigSim.refr_index[1] ** 2)
    eps = eps.at[-wg_length:,
          int(jnp.ceil(eps.shape[1] - wg_width) / 2):int(jnp.ceil((eps.shape[1] + wg_width) / 2)),
          thickness_sub:int(jnp.ceil(thickness_sub + rho.shape[2]))].set(ConfigSim.refr_index[2] ** 2)

    source = gaussian_3d((int(jnp.ceil(ConfigSim.lx * ConfigSim.resolution)),
                          int(jnp.ceil(ConfigSim.ly * ConfigSim.resolution)),
                          1),
                         (0, 0),
                         1,
                         (buffer + rho.shape[0] / 2, eps.shape[1]/2, thickness_sub + rho.shape[2] + 2 * ConfigSim.resolution),
                         rho.shape[0]/2)
    source = source / jnp.linalg.norm(source)
    source = jnp.pad(source,
                     [(0, 0)] * 2 + [(int(jnp.ceil((thickness_sub + rho.shape[2] + ConfigSim.buffer / 2 * ConfigSim.resolution - 1))),
                                      (int(eps.shape[2] - thickness_sub - rho.shape[2] - ConfigSim.buffer / 2 * ConfigSim.resolution)))])
    b_zero = jnp.zeros(source.shape, jnp.complex128)
    eps_r = (eps, eps, eps)
    sources = (b_zero, source, b_zero)
    omega = 2 * jnp.pi / (ConfigSim.wavelength * ConfigSim.resolution)

    z = tuple(omega ** 2 * t for t in eps_r)
    b = tuple(jnp.complex128(-1j * omega * b) for b in sources)
    n_pml = ConfigSim.dpml * ConfigSim.resolution
    params = jaxwell.Params(
        pml_ths=((n_pml, n_pml), (n_pml, n_pml), (n_pml, n_pml)),
        pml_omega=omega,
        eps=1e-6,
        max_iters=1000000
    )
    E, _ = jaxwell.solve(params, z, b)
    return E, np.array(eps_r[0])
