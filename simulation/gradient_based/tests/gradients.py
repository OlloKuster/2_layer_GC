import numpy as np
from autograd import value_and_grad
import autograd.numpy as anp
from simulation.gradient_based.config import ConfigSim
from util.projections import tanh_filter_ag_f, ssp_proj_ag_f
from util.structure_pillars import generate_pillars
import matplotlib.pyplot as plt


def test():
    weights = np.random.rand(ConfigSim.nx, ConfigSim.ny, 2)

    def pillar_f(weights_1, beta):
        filter = ssp_proj_ag_f(alpha=0.5, beta=beta, resolution=ConfigSim.nx / 50)
        return anp.sum(filter(generate_pillars(weights_1, ConfigSim.nz * ConfigSim.shallow_etch // ConfigSim.no_etch,
                                                ConfigSim.nz + 1)))
    vs = []
    for beta in [1, 1e1, 1e2, 1e3, anp.inf]:
        v, g = value_and_grad(pillar_f)(weights, beta)
        vs.append(np.linalg.norm(g))
        print(np.mean(g))
    plt.plot(vs)
    # plt.xscale('log')
    plt.show()


if __name__ == "__main__":
    test()
