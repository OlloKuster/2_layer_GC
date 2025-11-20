import tidy3d as td
from tidy3d.plugins.autograd import make_filter_and_project, rescale

import autograd.numpy as anp

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.sources_and_monitors import Sources, Monitors
from util.projections import tanh_filter_ag_f, ssp_proj_ag_f
from util.structure_pillars import generate_pillars


def simulation(weights, beta):

    rho = generate_pillars(weights, ConfigSim.nz * ConfigSim.shallow_etch // ConfigSim.no_etch, ConfigSim.nz + 1)    # rho = anp.repeat(weights[:, :, anp.newaxis], ConfigSim.nz, axis=2)
    # rho = anp.concatenate((rho, anp.flip(rho, axis=1)), axis=1)
    # rho = weights
    filter = tanh_filter_ag_f(alpha=0.5, beta=beta)
    rho_filt = filter(rho)
    eps = rescale(rho_filt, ConfigSim.eps_SiO2, ConfigSim.eps_Si)
    substrate = td.Structure(
        geometry=td.Box(center=(0,
                                0,
                                -ConfigSim.lz / 2 + ConfigSim.thickness_substrate / 2 - 0.5),
                        size=(td.inf, td.inf, ConfigSim.thickness_substrate + 1)),
        medium=td.Medium(permittivity=ConfigSim.eps_Si))
    waveguide = td.Structure(
        geometry=td.Box(center=(ConfigSim.lx / 2 - ConfigSim.wg_length / 2 + 0.5,
                                0,
                                -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.thickness_box + ConfigSim.wg_height / 2),
                        size=(ConfigSim.wg_length + 1, ConfigSim.wg_width, ConfigSim.wg_height)),
        medium=td.Medium(permittivity=ConfigSim.eps_Si)
    )

    custom_structure = td.Structure.from_permittivity_array(
        geometry=td.Box(center=(ConfigSim.lx / 2 - ConfigSim.wg_length - ConfigSim.rho_size[0] / 2,
                                ConfigSim.rho_size[1] / 4,
                                -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.thickness_box + ConfigSim.rho_size[2]/2),
                        size=(ConfigSim.rho_size[0], ConfigSim.rho_size[1] / 2, ConfigSim.rho_size[2])),
        eps_data=eps
    )
    grid_spec = td.GridSpec.auto(
        wavelength=ConfigSim.wavelength,
        min_steps_per_wvl=ConfigSim.min_p_wvl
    )

    sim = td.Simulation(
        size=[ConfigSim.lx, ConfigSim.ly, ConfigSim.lz],
        grid_spec=grid_spec,
        structures=[substrate, waveguide, custom_structure],
        sources=[Sources.source],
        monitors=[Monitors.fom_monitor, Monitors.field_monitor_xz,
                  Monitors.field_monitor_xy, Monitors.eps_monitor_xz, Monitors.eps_monitor_xy],
        run_time=ConfigSim.run_time,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        medium=td.Medium(permittivity=ConfigSim.eps_SiO2),
        symmetry=(0, -1, 0)
    )
    return sim
