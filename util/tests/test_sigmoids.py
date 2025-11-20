import matplotlib.pyplot as plt
import numpy as np
from autograd import grad

from util.projections import double_staircase_f


def test():
    x = np.linspace(0, 1, 100)
    double_staircase = double_staircase_f(0.07, 0.22, 100)

    df = grad(double_staircase)
    grads = [df(xs) for xs in x]

    plt.plot(x, double_staircase(x))
    print(double_staircase(0.07))
    # plt.plot(x, grads)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test()