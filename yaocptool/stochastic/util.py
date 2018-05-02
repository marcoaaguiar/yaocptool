import math
import numpy as np
from casadi import DM, SX, Function, mtimes, chol, solve, vertcat, log, exp
from scipy.stats.distributions import norm
from sobol import sobol_seq


def sample_parameter_normal_distribution_with_sobol(mean, covariance, n_samples=1):
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


def sample_parameter_log_normal_distribution_with_sobol(mean, covariance, n_samples=1):
    """Sample parameter using Sobol sampling with a log-normal distribution.

    :param mean:
    :param covariance:
    :param n_samples:
    :return:
    """
    if isinstance(mean, list):
        mean = vertcat(*mean)

    mean_log = log(mean)
    log_samples = sample_parameter_normal_distribution_with_sobol(mean_log, covariance, n_samples)
    samples = exp(log_samples)
    return samples
