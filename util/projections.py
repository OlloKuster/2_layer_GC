import autograd.numpy as anp


def sigmoid(x):
    return anp.exp(x) / (1. + anp.exp(x))


def double_staircase_f(h_1, h_2, beta):
    def double_staircase(x):
        return h_1 * sigmoid(beta * (x - h_1 / 2)) + (1 - h_1) * sigmoid(beta * (x - (h_2 + h_1) / 2))

    return double_staircase


def ssp_proj_ag_f(alpha, beta, resolution):
    f2bin = tanh_filter_ag_f(alpha, beta)

    def f2bin_smooth(rho):
        dx = dy = 1 / resolution
        R_smoothing = 0.55 * dx / 100
        rho_proj = f2bin(rho)
        rho_grad = anp.gradient(rho)
        rho_grad_norm2 = (rho_grad[0] / dx) ** 2 + (rho_grad[1] / dy) ** 2
        nonzero_norm = anp.abs(rho_grad_norm2) > 0
        rho_grad_norm = anp.sqrt(anp.where(nonzero_norm,
                                           rho_grad_norm2, 1))
        rho_grad_norm_eff = anp.where(nonzero_norm, rho_grad_norm, 1)
        d = (alpha - rho) / rho_grad_norm_eff
        needs_smoothing = nonzero_norm & (anp.abs(d) < R_smoothing)
        d_R = d / R_smoothing
        F = anp.where(needs_smoothing,
                      0.5 - 15 / 16 * d_R + 5 / 8 * d_R ** 3 - 3 / 16 * d_R ** 5,
                      1.0)
        F_minus = anp.where(needs_smoothing,
                            0.5 + 15 / 16 * d_R - 5 / 8 * d_R ** 3 + 3 / 16 * d_R ** 5,
                            1.0)
        rho_minus = rho - R_smoothing * rho_grad_norm_eff * F
        rho_plus = rho + R_smoothing * rho_grad_norm_eff * F_minus
        rho_minus_eff_proj = f2bin(rho_minus)
        rho_plus_eff_proj = f2bin(rho_plus)
        rho_proj_smoothed = (1 - F) * rho_minus_eff_proj + F * rho_plus_eff_proj
        return anp.where(needs_smoothing, rho_proj_smoothed, rho_proj)

    return f2bin_smooth


def tanh_filter_ag_f(alpha=0.5, beta=30):
    def f2bin(rho_0):
        """
        Binarises the values of x with parameters alpha and beta.
        :param rho_0: Array which will be binarised.
        :param alpha: Steepness of the binarisation function.
        :param beta: Origin of the binarisation function.
        :return: Binarised array of x.
        """
        if beta == anp.inf:
            return anp.where(rho_0 > alpha, 1.0, 0.0)
        else:
            num = anp.tanh(alpha * beta) + anp.tanh(beta * (rho_0 - alpha))
            denom = anp.tanh(alpha * beta) + anp.tanh(beta * (1 - alpha))
            proj = num / denom
            return proj

    return f2bin
