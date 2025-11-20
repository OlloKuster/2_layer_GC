import numpy as np


def generate_pillars(matrix, size_x, size_y, height):
    '''
    Generates a (non-differentiable) density distribution of pillars.
    Note: I am not entirely sure about
    :param matrix: Weight matrix (to be optimized).
    :param width_x: Width in x-direction of a single pillar (in px).
    :param width_y:
    :param height:
    :return:
    '''
    nx = size_x / len(matrix[0])
    ny = size_y / len(matrix[1])
    rho = np.zeros((int(size_x), int(size_y), int(height)))
    for i_x in range(len(matrix[1])):
        for i_y in range(len(matrix[0])):
            rho[int(i_x * nx):int((i_x + 1) * nx),
                int(i_y * ny):int((i_y + 1) * ny),
                0:int(round(matrix[i_x, i_y])*height / 2)] = 1
    return rho
