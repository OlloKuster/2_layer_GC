import tidy3d as td
from tidy3d.plugins.autograd import make_filter_and_project, rescale

import autograd.numpy as anp

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.sources_and_monitors import Sources, Monitors
from util.projections import tanh_filter_ag_f, ssp_proj_ag_f, double_staircase_f
from util.structure_pillars import generate_pillars


def simulation(weights, positions, beta):
    filter = double_staircase_f(ConfigSim.shallow_etch, ConfigSim.no_etch, beta)
    gratings = []
    w = ConfigSim.size_pillars
    for h, (x, y) in zip(weights, positions):
        etch_depth = filter(h)
        grating = td.Box(
            center=(x, y, -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.thickness_box + etch_depth/2),
            size=(w, w, etch_depth)
        )
        gratings.append(grating)

    struct = td.Structure(geometry=td.GeometryGroup(geometries=gratings), medium=td.Medium(permittivity=ConfigSim.eps_Si))
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
    refine_box = td.MeshOverrideStructure(
        geometry=td.Box(center=(-w/2, ConfigSim.rho_size[1]/4 - w / 4, -ConfigSim.lz / 2 + ConfigSim.thickness_substrate + ConfigSim.thickness_box + ConfigSim.wg_height / 2),
                        size=(ConfigSim.rho_size[0]+0.2, ConfigSim.rho_size[1]/2+0.2, ConfigSim.rho_size[2]+ConfigSim.dl)),
        dl=[None, None, ConfigSim.dl]
    )
    grid_spec = td.GridSpec.auto(
        wavelength=ConfigSim.wavelength,
        min_steps_per_wvl=ConfigSim.min_p_wvl,
        override_structures=[refine_box]
    )

    sim = td.Simulation(
        size=[ConfigSim.lx, ConfigSim.ly, ConfigSim.lz],
        grid_spec=grid_spec,
        structures=[substrate, waveguide, struct],
        sources=[Sources.source],
        monitors=[Monitors.fom_monitor, Monitors.field_monitor_xz,
                  Monitors.field_monitor_xy, Monitors.eps_monitor_xz, Monitors.eps_monitor_xy],
        run_time=ConfigSim.run_time,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        medium=td.Medium(permittivity=ConfigSim.eps_SiO2),
        symmetry=(0, -1, 0)
    )
    return sim
