from dataclasses import dataclass
import tidy3d as td
import numpy as np

from simulation.gradient_based.config import ConfigSim


@dataclass
class Sources:
    source = td.GaussianBeam(
        center=(0, 0, ConfigSim.src_pos),
        size=(td.inf, td.inf, 0),
        source_time=td.GaussianPulse(freq0=ConfigSim.freq0, fwidth=ConfigSim.fwidth),
        pol_angle=np.pi / 2,
        angle_theta=0,
        direction="-",
        num_freqs=1,
        waist_radius=ConfigSim.spot_size / 2,
        name='source'
    )


@dataclass
class Monitors:
    mode_spec = td.ModeSpec(num_modes=1, target_neff=np.sqrt(ConfigSim.eps_Si))
    wavelengths = np.linspace(1.5, 1.6, 21)
    freqs = td.C_0 / wavelengths
    fom_monitor = td.ModeMonitor(
        center=[ConfigSim.mon_pos_x, 0, ConfigSim.mon_pos_z],
        size=[0, 3 * ConfigSim.wg_width, 3 * ConfigSim.wg_width],
        freqs=[ConfigSim.freq0],
        mode_spec=mode_spec,
        name="fom_monitor",
    )

    field_monitor_xz = td.FieldMonitor(
        center=(0, 0, ConfigSim.mon_pos_z),
        size=(td.inf, 0, td.inf),
        freqs=[ConfigSim.freq0],
        name="FieldMonitor_xz"
    )

    field_monitor_xy = td.FieldMonitor(
        center=(0, 0, ConfigSim.mon_pos_z),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSim.freq0],
        name="FieldMonitor_xy"
    )

    eps_monitor_xz = td.PermittivityMonitor(
        center=(0, 0, ConfigSim.mon_pos_z),
        size=(td.inf, 0, td.inf),
        freqs=[ConfigSim.freq0],
        name="PermittivityMonitor_xz"
    )

    eps_monitor_xy = td.PermittivityMonitor(
        center=(0, 0, ConfigSim.mon_pos_z),
        size=(td.inf, td.inf, 0),
        freqs=[ConfigSim.freq0],
        name="PermittivityMonitor_xy"
    )
