import math

import numpy as np
import sobol_seq
from casadi import DM, mtimes, chol, vertcat, log, exp
from scipy.stats.distributions import norm


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
    sobol_design = sobol_seq.i4_sobol_generate(
        n_uncertain, n_samples, math.ceil(np.log2(n_samples))
    )
    sobol_samples = DM(sobol_design.T)
    for i in range(n_uncertain):
        sobol_samples[i, :] = norm(loc=0.0, scale=1).ppf(sobol_samples[i, :])

    unscaled_sample = DM.zeros(n_uncertain, n_samples)

    for i in range(n_samples):
        unscaled_sample[:, i] = mean + mtimes(chol(covariance).T, sobol_samples[:, i])

    return unscaled_sample


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
    log_samples = sample_parameter_normal_distribution_with_sobol(
        mean_log, covariance, n_samples
    )
    samples = exp(log_samples)
    return samples
