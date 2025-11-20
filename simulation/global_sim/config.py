from dataclasses import dataclass


@dataclass
class ConfigSim:
    resolution = 20

    wavelength = 1.55
    freq0 = 1 / wavelength
    fwidth = freq0 / 10
    run_time = 50 / fwidth

    dpml = 0.5

    rho_size = (2, 2, 0.4)
    thickness_substrate = 1
    buffer = 1 * wavelength

    wg_width = 0.44
    wg_length = 2

    lx = buffer + rho_size[0] + wg_length
    ly = buffer + rho_size[1] + buffer
    lz = thickness_substrate + rho_size[2] + buffer

    pos_source = [-lx / 2 + buffer + rho_size[0] / 2, 0, lz / 2 - dpml - 0.1]
    size_source = [rho_size[0] / 4, rho_size[1] / 4, 0]

    pos_monitor = [lx/2 - wg_length / 2, 0, -lz / 2 + thickness_substrate + rho_size[2]/2]
    size_monitor = [rho_size[0] / 2, wg_width, 0]

    refr_index = (1., 1.44, 3.48)  # "cladding", substrate, material
    min_feature_size = 0.06

    dl = 10
    nx = 10
    ny = 20

    i = 0