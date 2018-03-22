from itertools import product

import math
from casadi import sqrt, diag, DM, SX, Function, mtimes, chol, solve, MX, vertcat, log, exp
from scipy.stats.distributions import norm
from sobol import sobol_seq
import numpy as np


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
    sobol_samples = sobol_design.T
    for i in range(n_uncertain):
        sobol_samples[:, i] = norm(loc=0., scale=1.).ppf(sobol_samples[:, i])

    log_samples = DM.zeros(n_samples, n_uncertain)
    std = sqrt(diag(covariance))
    for i in range(n_samples):
        for j in range(n_uncertain):
            log_samples[i, j] = mean[j] + sobol_samples[i, j] * std[j]

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
    He0fcn = Function('He0fcn', [x], [1.])
    He1fcn = Function('He1fcn', [x], [x])
    He2fcn = Function('He2fcn', [x], [x ** 2 - 1])
    He3fcn = Function('He3fcn', [x], [x ** 3 - 3 * x])
    He4fcn = Function('He4fcn', [x], [x ** 4 - 6 * x ** 2 + 3])
    He5fcn = Function('He5fcn', [x], [x ** 5 - 10 * x ** 3 + 15 * x])
    He6fcn = Function('He6fcn', [x], [x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15])
    He7fcn = Function('He7fcn', [x], [x ** 7 - 21 * x ** 5 + 105 * x ** 3 - 105 * x])
    He8fcn = Function('He8fcn', [x], [x ** 8 - 28 * x ** 6 + 210 * x ** 4 - 420 * x ** 2 + 105])
    He9fcn = Function('He9fcn', [x], [x ** 9 - 36 * x ** 7 + 378 * x ** 5 - 1260 * x ** 3 + 945 * x])
    He10fcn = Function('He10fcn', [x], [x ** 10 - 45 * x ** 8 + 640 * x ** 6 - 3150 * x ** 4 + 4725 * x ** 2 - 945])
    Helist = [He0fcn, He1fcn, He2fcn, He3fcn, He4fcn, He5fcn, He6fcn, He7fcn, He8fcn, He9fcn, He10fcn]

    # Calculation of factor for least-squares
    xu = SX.sym("xu", n_uncertain)
    exps = (p for p in product(range(pc_order + 1), repeat=n_uncertain) if sum(p) <= pc_order)
    exps.next()
    exps = list(exps)

    PSI = SX.ones(math.factorial(n_uncertain + pc_order) / (math.factorial(n_uncertain) * math.factorial(pc_order)))
    for i in range(len(exps)):
        for j in range(n_uncertain):
            PSI[i + 1] *= Helist[exps[i][j]](xu[j])
    PSIfcn = Function('PSIfcn', [xu], [PSI])

    nparameter = SX.size(PSI)[0]
    PSImatrix = SX.zeros(n_samples, nparameter)
    for i in range(n_samples):
        PSIa = PSIfcn(sobol_samples[i, :])
        for j in range(SX.size(PSI)[0]):
            PSImatrix[i, j] = PSIa[j]

    PSITPSI = mtimes(PSImatrix.T, PSImatrix) + lamb*DM.eye(nparameter)
    cholPSITPSI = chol(PSITPSI)
    invcholPSITPSI = solve(cholPSITPSI, SX.eye(nparameter))
    invPSITPSI = mtimes(invcholPSITPSI, invcholPSITPSI.T)

    LSfactor = mtimes(invPSITPSI, PSImatrix.T)
    LSfactor = DM(LSfactor)

    # Calculation of expectations for variance function
    nsample = 100000
    Xsample = np.random.multivariate_normal(np.zeros(n_uncertain), np.eye(n_uncertain), nsample)
    PSIsquaredsum = DM.zeros(SX.size(PSI)[0])
    for i in range(nsample):
        PSIsquaredsum += PSIfcn(Xsample[i, :]) ** 2
    Expectationvector = PSIsquaredsum / nsample
    #    Expectationvector = MX([1.,1.,2.,1.,1.,2.])

    return LSfactor, Expectationvector, PSIfcn
