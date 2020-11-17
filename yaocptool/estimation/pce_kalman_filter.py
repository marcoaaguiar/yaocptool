# coding=utf-8
from math import factorial

from casadi import mtimes, vertcat, solve, DM, vec

from yaocptool.estimation.estimator_abstract import EstimatorAbstract
from yaocptool.modelling import SystemModel, DataSet
from yaocptool.stochastic import sample_parameter_normal_distribution_with_sobol
from yaocptool.stochastic.pce import get_ls_factor


class PCEKalmanFilter(EstimatorAbstract):
    def __init__(self, model, **kwargs):
        """

        :param SystemModel model: estimator model
        :param x_mean: a initial guess for the mean
        :param p_k: a initial guess for the covariance of the estimator
        :param h_function: a function that receives 3 parameters (x, y_algebraic, u) and returns an measurement.
        :param c_matrix: if h_function is not give, c_matrix is "C" from the measurement equation:
                         y_meas = C*[x y_alg] + D*u. note that it has to have n_x + n_y (algebraic) columns.
        :param DM r_v: process noise matrix
        :param DM r_n: measurement noise matrix
        :param int pc_order: Polynomial chaos order
        :param int n_samples: Number of samples to be used
        """
        self.model = model

        self.h_function = None
        self.c_matrix = None
        self.d_matrix = None

        self.implementation = "standard"

        self.x_mean = None
        self.p_k = None

        self.r_v = DM(0.0)
        self.r_n = DM(0.0)

        self.p = None
        self.theta = None
        self.y_guess = None

        self.n_samples = None
        self.n_uncertain = None
        self.pc_order = 4

        self._types_fixed = False
        self._checked = False
        EstimatorAbstract.__init__(self, **kwargs)
        self._fix_types()

        if self.n_uncertain is None:
            self.n_uncertain = self.x_mean.numel()

        if self.n_samples is None:
            self.n_samples = factorial(self.n_uncertain + self.pc_order) // (
                factorial(self.n_uncertain) * factorial(self.pc_order)
            )

        if self.n_samples < self.n_pol_parameters:
            raise ValueError(
                "Number of samples has to greater or equal to the number of polynomial parameters"
                '"n_samples"={}, n_pol_parameter={}'.format(
                    self.n_samples, self.n_pol_parameters
                )
            )

        self._ls_factor, _, _ = get_ls_factor(
            self.n_uncertain, self.n_samples, self.pc_order
        )

        self.dataset = DataSet(name=self.model.name)
        self.dataset.data["x"]["size"] = self.model.n_x
        self.dataset.data["x"]["names"] = [
            "est_" + self.model.x_sym[i].name() for i in range(self.model.n_x)
        ]

        self.dataset.data["P"]["size"] = self.model.n_x ** 2
        self.dataset.data["P"]["names"] = [
            "P_" + str(i) + str(j)
            for i in range(self.model.n_x)
            for j in range(self.model.n_x)
        ]

    @property
    def n_pol_parameters(self):
        n_pol_parameters = factorial(self.n_uncertain + self.pc_order) / (
            factorial(self.n_uncertain) * factorial(self.pc_order)
        )
        return n_pol_parameters

    def _fix_types(self):
        self.x_mean = vertcat(self.x_mean)
        self.p_k = vertcat(self.p_k)

        self.r_v = vertcat(self.r_v)
        self.r_n = vertcat(self.r_n)

        if self.p is not None:
            self.p = vertcat(self.p)

    def _check(self):
        if self.x_mean is None:
            raise ValueError('A initial condition for the "x_mean" must be provided')

        if self.p_k is None:
            raise ValueError('A initial condition for the "p_k" must be provided')

        if self.h_function is None and self.c_matrix is None:
            raise ValueError(
                'Neither a measurement function "h_function" or a measurement matrix "c_matrix" was given'
            )

        return True

    def estimate(self, t_k, y_k, u_k):
        if not self._checked:
            self._check()
            self._checked = True
        if not self._types_fixed:
            self._fix_types()
            self._types_fixed = True

        x_mean = self.x_mean
        x_cov = self.p_k

        (
            x_hat_k_minus,
            p_k_minus,
            y_hat_k_minus,
            p_yk_yk,
            k_gain,
        ) = self._priori_update(x_mean, x_cov, u=u_k, p=self.p, theta=self.theta)

        x_hat_k = x_hat_k_minus + mtimes(k_gain, (y_k - y_hat_k_minus))
        p_k = p_k_minus - mtimes(k_gain, mtimes(p_yk_yk, k_gain.T))

        self.x_mean = x_hat_k
        self.p_k = p_k

        self.dataset.insert_data("x", t_k, self.x_mean)
        self.dataset.insert_data("P", t_k, vec(self.p_k))

        return x_hat_k, p_k

    def _get_measurement_from_prediction(self, x, y, u):
        if self.h_function is not None:
            measurement_prediction = self.h_function(x, y, u)
        elif self.c_matrix is not None:
            measurement_prediction = mtimes(self.c_matrix, vertcat(x, y))
            if self.d_matrix is not None:
                measurement_prediction = measurement_prediction + mtimes(
                    self.d_matrix, u
                )
        else:
            raise ValueError(
                'Neither a measurement function "h_function" or a measurement matrix "c_matrix" was given'
            )
        return measurement_prediction

    def _priori_update(self, x_mean, x_cov, u, p, theta):
        x_samples = self._get_sampled_states(x_mean, x_cov)

        # Perform predictions via simulation
        simulation_results = []
        x_cal_x_k_at_k_minus_1 = []
        y_alg_cal_x_k_at_k_minus_1 = []
        for s in range(self.n_samples):
            x_0_i = DM(x_samples[s])
            simulation_results_i = self.model.simulate(
                x_0=x_0_i,
                t_0=self.t,
                t_f=self.t + self.t_s,
                u=u,
                p=p,
                theta=theta,
                y_0=self.y_guess,
            )
            simulation_results.append(simulation_results_i)
            x_cal_x_k_at_k_minus_1.append(simulation_results_i.final_condition()[0])
            y_alg_cal_x_k_at_k_minus_1.append(
                simulation_results[s].final_condition()[1]
            )

        # fit the polynomial for x

        a_x = []
        x_hat_k_minus = []
        for i in range(self.model.n_x):
            x_i_vector = vertcat(
                *[x_cal_x_k_at_k_minus_1[s][i] for s in range(self.n_samples)]
            )
            a_x.append(mtimes(self._ls_factor, x_i_vector))

        # get the mean for x
        for i in range(self.model.n_x):
            x_hat_k_minus.append(a_x[i][0])
        x_hat_k_minus = vertcat(*x_hat_k_minus)

        # get the covariance for x
        p_k_minus = DM.zeros(self.model.n_x, self.model.n_x)
        for i in range(self.model.n_x):
            for j in range(self.model.n_x):
                p_k_minus[i, j] = (
                    sum([a_x[i][k] * a_x[j][k] for k in range(1, self.n_samples)])
                    + self.r_v[i, j]
                )

        # calculate the measurement for each sample
        y_cal_k_at_k_minus_1 = []
        for s in range(self.n_samples):
            y_cal_k_at_k_minus_1.append(
                self._get_measurement_from_prediction(
                    x_cal_x_k_at_k_minus_1[s], y_alg_cal_x_k_at_k_minus_1[s], u
                )
            )
        n_meas = y_cal_k_at_k_minus_1[0].numel()

        # find the measurements estimate
        a_meas = []
        y_meas_hat_k_minus = []
        for i in range(n_meas):
            y_meas_i_vector = vertcat(
                *[y_cal_k_at_k_minus_1[s][i] for s in range(self.n_samples)]
            )
            a_meas.append(mtimes(self._ls_factor, y_meas_i_vector))

        # get the mean for the measurement
        for i in range(n_meas):
            y_meas_hat_k_minus.append(a_meas[i][0])
        y_meas_hat_k_minus = vertcat(*y_meas_hat_k_minus)

        # get the covariance for the meas
        p_yk_yk = DM.zeros(n_meas, n_meas)
        for i in range(n_meas):
            for j in range(n_meas):
                p_yk_yk[i, j] = sum(
                    [a_meas[i][k] * a_meas[j][k] for k in range(1, self.n_samples)]
                )
        p_yk_yk = p_yk_yk + self.r_n

        # get cross-covariance
        p_xk_yk = DM.zeros(self.model.n_x, n_meas)
        for i in range(self.model.n_x):
            for j in range(n_meas):
                p_xk_yk[i, j] = sum(
                    [a_x[i][k] * a_meas[j][k] for k in range(1, self.n_samples)]
                )

        # k_gain = mtimes(p_xk_yk, inv(p_yk_yk))
        k_gain = solve(p_yk_yk.T, p_xk_yk.T).T

        return x_hat_k_minus, p_k_minus, y_meas_hat_k_minus, p_yk_yk, k_gain

    def _get_sampled_states(self, x_mean, x_cov):
        x_samples = sample_parameter_normal_distribution_with_sobol(
            x_mean, x_cov, self.n_samples
        )
        return [x_samples[:, i] for i in range(self.n_samples)]
