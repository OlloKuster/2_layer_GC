import autograd
import autograd.numpy as anp
import h5py
import numpy as np
import matplotlib.pyplot as plt
import optax
import nlopt
from simulation.gradient_based.optimizer.config import ConfigOptax


def optimizer(weights, positions, objective_f):
    weights = np.array(weights)
    optimizer = optax.adam(learning_rate=ConfigOptax.learning_rate)
    opt_state = optimizer.init(weights)

    loss_hist = []
    rho_hist = [weights]

    betas = [8, 16, 32, 1e2]

    for beta in betas:
        for i in range(ConfigOptax.num_steps):

            if len(betas) > 1 and betas[0] == 1:
                beta = beta + i * 1
            objective = objective_f(positions=positions, step_num=i, beta=beta)
            value, gradient = autograd.value_and_grad(objective)(weights)

            print(f"step = {i + 1}")
            print(f"\tbeta = {beta:.4e}")
            print(f"\tval = {value:.4e}")
            print(f"\tgrad_norm = {np.linalg.norm(gradient):.4e}")

            updates, opt_state = optimizer.update(-gradient, opt_state, weights)
            weights[:] = optax.apply_updates(weights, updates)

            anp.clip(weights, 0.0, 1.0, out=weights)

            loss_hist.append(value)
            plt.plot(loss_hist)
            plt.savefig("plots/loss.png")
            plt.close()
            # rho_hist.append(rho)



    with h5py.File(
            f"plots/data.h5",
            'w') as f:
        grp = f.create_group("lens_3d")
        grp.create_dataset("rho", data=weights)
        grp.create_dataset("loss", data=loss_hist)
        f.close()


