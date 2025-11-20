import autograd
import autograd.numpy as anp
import numpy as np
import nlopt
import time
import matplotlib.pyplot as plt
import h5py

from simulation.gradient_based.optimizer.config import ConfigNlopt as config


def optimizer(weights, objective, beta):
    """
    The optimiser function
    :return:
    """
    loss_hist = []

    def f(x, g):
        start = time.time()
        x = anp.reshape(x, weights.shape)
        value, grad = autograd.value_and_grad(objective)(x, step_num=config.i, beta=beta)
        value = float(value)  # Requires np float and not jax.numpy float
        print(value)
        print(f"\tgrad_norm = {np.linalg.norm(grad):.4e}")
        config.i += 1
        loss_hist.append(value)
        if g.size > 0:
            g[:] = grad.ravel()
        end = time.time()
        print(f"time: {end-start}")
        return value

    opt = nlopt.opt(config.OPTIMISER, weights.size)
    opt.set_min_objective(f)
    opt.set_maxeval(config.MAXEVAL)
    # opt.set_ftol_abs(config.FTOL_ABS)
    # opt.set_ftol_rel(config.FTOL_REL)
    opt.set_upper_bounds(config.UPPER_BOUNDS)
    opt.set_lower_bounds(config.LOWER_BOUNDS)

    rho_opt = opt.optimize(weights.ravel())
    rho_opt = rho_opt.reshape(weights.shape)

    with h5py.File(
            f"plots/data_{beta}.h5",
            'w') as f:
        grp = f.create_group("gc")
        grp.create_dataset("rho", data=rho_opt)
        grp.create_dataset("loss", data=loss_hist)
        f.close()
    return rho_opt, loss_hist