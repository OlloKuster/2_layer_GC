from dataclasses import dataclass
from tidy3d import C_0
import numpy as np

@dataclass
class ConfigSim:
    wavelength = 1.55
    freq0 = C_0 / wavelength
    fwidth = freq0 / 100

    rho_size = (10, 10, 0.22)
    thickness_box = 2 # Buried Oxide Layer
    thickness_tox = 2 # Thin Oxide Layer
    thickness_substrate = 1
    wg_length = 2
    wg_width = 0.45
    wg_height = rho_size[2]
    buffer = 1 * wavelength
    lx = buffer + rho_size[0] + wg_length
    ly = buffer + rho_size[1] + buffer
    lz = thickness_substrate + thickness_box + rho_size[2] + thickness_tox

    src_pos = rho_size[2] / 2 + 0.75
    spot_size = 10.4

    mon_pos_x = lx / 2 - wg_length + 0.5
    mon_pos_z = -lz / 2 + thickness_substrate + thickness_box + wg_height / 2

    no_etch = rho_size[2]
    shallow_etch = 0.07


    eps_Si = 3.48**2
    eps_SiO2 = 1.44**2

    min_p_wvl = 10
    dl = 0.025
    run_time = 5 / fwidth

    size_pillars = 0.3

    nx = int(np.ceil(rho_size[0] // size_pillars))
    ny = int(np.ceil(rho_size[1] // size_pillars / 2))
    nz = 100
