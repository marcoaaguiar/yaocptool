import numpy as np


def sparsity(M, psi):
    """

    :param M:
    :param float psi:

    :return:
    :rtype:
    """
    N = np.empty_like(M)
    for row in range(len(N)):
        for col in range(len(N[row])):
            prob = np.random.rand()
            if prob < psi:
                N[row][col] = 0
            else:
                N[row][col] = 1

    return N * M


def grad_tanh(z):
    return np.diag(1 - np.tanh(z.flatten()) ** 2)

def normalize_std_deviation(data):
    means = np.atleast_2d(np.mean(data, axis=0))
    std = np.atleast_2d(np.std(data, axis=0))

    normal_input = (data - means) / std
    return normal_input
