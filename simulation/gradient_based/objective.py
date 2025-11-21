import tidy3d.web as web
import autograd.numpy as anp

from simulation.gradient_based.config import ConfigSim
from simulation.gradient_based.simulation import simulation


def measure_transmission(sim_data):
    output_amps = sim_data["fom_monitor"].amps
    amp = output_amps.sel(direction="+", f=ConfigSim.freq0, mode_index=0).values
    return 1 - anp.sum(anp.abs(amp) ** 2)


def objective_f(positions, step_num, beta):
    def objective(weight):
        sim_em = simulation(weight, positions, beta)

        task_name = "grating_coupler_checkerboard"
        task_name += f"_step_{step_num:03}"
        sim_data = web.run(sim_em, task_name=task_name, folder_name="gc_checkerboard_10", verbose=False)

        v_em = measure_transmission(sim_data)
        return v_em

    return objective
