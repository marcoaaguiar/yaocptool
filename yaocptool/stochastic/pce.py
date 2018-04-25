import math
from itertools import product

import numpy as np
from casadi import DM, SX, Function, mtimes, chol, solve, vertcat, log, exp
from scipy.stats.distributions import norm
from sobol import sobol_seq


def sample_parameter_normal_distribution(mean, covariance, n_samples=1):
    """Sample parameter using Sobol sampling with a normal distribution.

    :param mean:
    :param covariance:
    :param n_samples:
    :return:
    """
    if isinstance(mean, list):
        mean = vertcat(*mean)

    n_uncertain = mean.size1()

    # Uncertain parameter design
    sobol_design = sobol_seq.i4_sobol_generate(n_uncertain, n_samples, math.ceil(np.log2(n_samples)))
    sobol_samples = DM(sobol_design.T)
    for i in range(n_uncertain):
        sobol_samples[:, i] = norm(loc=0., scale=1.).ppf(sobol_samples[:, i])

    log_samples = SX.zeros(n_samples, n_uncertain)

    for i in range(n_samples):
        log_samples[i, :] = mean + mtimes(sobol_samples[i, :], chol(covariance)).T

    return log_samples


def sample_parameter_log_normal_distribution(mean, covariance, n_samples=1):
    """Sample parameter using Sobol sampling with a log-normal distribution.

    :param mean:
    :param covariance:
    :param n_samples:
    :return:
    """
    if isinstance(mean, list):
        mean = vertcat(*mean)

    mean_log = log(mean)
    log_samples = sample_parameter_normal_distribution(mean_log, covariance, n_samples)
    samples = exp(log_samples)
    return samples


def get_ls_factor(n_uncertain, n_samples, pc_order, lamb=0):
    # Uncertain parameter design
    sobol_design = sobol_seq.i4_sobol_generate(n_uncertain, n_samples, math.ceil(np.log2(n_samples)))
    sobol_samples = np.transpose(sobol_design)
    for i in range(n_uncertain):
        sobol_samples[:, i] = norm(loc=0., scale=1.).ppf(sobol_samples[:, i])

    # Polynomial function definition
    x = SX.sym('x')
    he0fcn = Function('He0fcn', [x], [1.])
    he1fcn = Function('He1fcn', [x], [x])
    he2fcn = Function('He2fcn', [x], [x ** 2 - 1])
    he3fcn = Function('He3fcn', [x], [x ** 3 - 3 * x])
    he4fcn = Function('He4fcn', [x], [x ** 4 - 6 * x ** 2 + 3])
    he5fcn = Function('He5fcn', [x], [x ** 5 - 10 * x ** 3 + 15 * x])
    he6fcn = Function('He6fcn', [x], [x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15])
    he7fcn = Function('He7fcn', [x], [x ** 7 - 21 * x ** 5 + 105 * x ** 3 - 105 * x])
    he8fcn = Function('He8fcn', [x], [x ** 8 - 28 * x ** 6 + 210 * x ** 4 - 420 * x ** 2 + 105])
    he9fcn = Function('He9fcn', [x], [x ** 9 - 36 * x ** 7 + 378 * x ** 5 - 1260 * x ** 3 + 945 * x])
    he10fcn = Function('He10fcn', [x], [x ** 10 - 45 * x ** 8 + 640 * x ** 6 - 3150 * x ** 4 + 4725 * x ** 2 - 945])
    helist = [he0fcn, he1fcn, he2fcn, he3fcn, he4fcn, he5fcn, he6fcn, he7fcn, he8fcn, he9fcn, he10fcn]

    # Calculation of factor for least-squares
    xu = SX.sym("xu", n_uncertain)
    exps = (p for p in product(range(pc_order + 1), repeat=n_uncertain) if sum(p) <= pc_order)
    exps.next()
    exps = list(exps)

    psi = SX.ones(math.factorial(n_uncertain + pc_order) / (math.factorial(n_uncertain) * math.factorial(pc_order)))
    for i in range(len(exps)):
        for j in range(n_uncertain):
            psi[i + 1] *= helist[exps[i][j]](xu[j])
    psi_fcn = Function('PSIfcn', [xu], [psi])

    nparameter = SX.size(psi)[0]
    psi_matrix = SX.zeros(n_samples, nparameter)
    for i in range(n_samples):
        psi_a = psi_fcn(sobol_samples[i, :])
        for j in range(SX.size(psi)[0]):
            psi_matrix[i, j] = psi_a[j]

    psi_t_psi = mtimes(psi_matrix.T, psi_matrix) + lamb * DM.eye(nparameter)
    chol_psi_t_psi = chol(psi_t_psi)
    inv_chol_psi_t_psi = solve(chol_psi_t_psi, SX.eye(nparameter))
    inv_psi_t_psi = mtimes(inv_chol_psi_t_psi, inv_chol_psi_t_psi.T)

    ls_factor = mtimes(inv_psi_t_psi, psi_matrix.T)
    ls_factor = DM(ls_factor)

    # Calculation of expectations for variance function
    n_sample_expectation_vector = 100000
    x_sample = np.random.multivariate_normal(np.zeros(n_uncertain), np.eye(n_uncertain), n_sample_expectation_vector)
    psi_squared_sum = DM.zeros(SX.size(psi)[0])
    for i in range(n_sample_expectation_vector):
        psi_squared_sum += psi_fcn(x_sample[i, :]) ** 2
    expectation_vector = psi_squared_sum / n_sample_expectation_vector

    return ls_factor, expectation_vector, psi_fcn
