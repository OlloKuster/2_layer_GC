import autograd.numpy as anp
import numpy as np

from util.projections import double_staircase_f


def generate_pillars(matrix, height_1, height_2):
    '''
    Generates a (non-differentiable) density distribution of pillars.
    Note: I am not entirely sure about
    :param matrix: Weight matrix (to be optimized).
    :param width_x: Width in x-direction of a single pillar (in px).
    :param width_y:
    :param height:
    :return:
    '''
    layer_1 = anp.repeat(matrix[:, :, 0][:, :, anp.newaxis], height_1, axis=2)
    layer_2 = anp.repeat((matrix[:, :, 0] * matrix[:, :, 1])[:, :, anp.newaxis], height_2, axis=2)
    return anp.concatenate((layer_1, layer_2), axis=2)


def generate_pillars_(matrix, height_1, height_2):
    '''
    Generates a (non-differentiable) density distribution of pillars.
    Note: I am not entirely sure about
    :param layer_1: Weight matrix (to be optimized).
    :param width_x: Width in x-direction of a single pillar (in px).
    :param width_y:
    :param height:
    :return:
    '''
    mask = matrix[:, :, 0] > 0
    layer_1 = anp.concatenate((matrix[:, :, 0][..., anp.newaxis], (matrix[:, :, 0] * mask)[..., anp.newaxis]), axis=2)
    for i in range(1, int(height_1 - 1)):
        layer_1 = anp.concatenate((layer_1, (layer_1[:, :, i - 1] * mask)[..., anp.newaxis]), axis=2)
        mask = layer_1[:, :, -1]

    # mask = matrix[:, :, 1] > 0
    # layer_2 = anp.concatenate((matrix[:, :, 1][..., anp.newaxis], (matrix[:, :, 1] * mask)[..., anp.newaxis]), axis=2)
    # for i in range(1, int((height_2-height_1) - 1)):
    #     layer_2 = anp.concatenate((layer_2, (layer_2[:, :, i - 1] * mask)[..., anp.newaxis]), axis=2)
    #     mask = layer_2[:, :, -1]
    #
    # # layer_1 = anp.repeat(matrix[:, :, 0][:, :, anp.newaxis], height_1, axis=2)
    # # layer_2 = anp.repeat((matrix[:, :, 0] * matrix[:, :, 1])[:, :, anp.newaxis], height_2, axis=2)
    # res = np.concat((layer_1[..., :-1], layer_2[..., :-1]), axis=2)
    return layer_1

    # import matplotlib.pyplot as plt
    # plt.imshow(res[:, res.shape[1]//2].T, origin='lower')
    # plt.show()


def generate_pillars_fill(matrix, nz, height_1, height_2, beta):
    '''
    Generates a (non-differentiable) density distribution of pillars.
    Note: I am not entirely sure about
    :param matrix: Weight matrix (to be optimized).
    :param width_x: Width in x-direction of a single pillar (in px).
    :param width_y:
    :param height:
    :return:
    '''
    rho = anp.zeros((matrix.shape[0], matrix.shape[1], nz))
    projection = double_staircase_f(height_1 / nz, height_2 / nz, beta)
    for i_x in range(len(matrix[0]) - 1):
        for i_y in range(len(matrix[1]) - 1):
            rho[i_x, i_y,
            0:int(projection(matrix[i_x, i_y])._value)] = 1
    return rho
