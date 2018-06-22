# coding=utf-8
from casadi import chol, mtimes, vertcat, solve, DM, vec

from yaocptool.estimation.estimator_abstract import EstimatorAbstract
from yaocptool.modelling import SystemModel, DataSet


class UnscentedKalmanFilter(EstimatorAbstract):
    def __init__(self, model, **kwargs):
        """
            Unscented Kalman Filter. Two versions are implemented standard and square-root.
            Implemented based on [1] and [2].



        References:

        [1] Wan, E. A., & Van Der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation.
        In Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium
        (Cat. No.00EX373) (Vol. v, pp. 153–158). IEEE. http://doi.org/10.1109/ASSPCC.2000.882463

        [2] Merwe, R. Van Der, & Wan, E. a. (2001). The square-root unscented Kalman filter for state and
        parameter-estimation. 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing.
        Proceedings (Cat. No.01CH37221), 6, 1–4. http://doi.org/10.1109/ICASSP.2001.940586

        :param SystemModel model: estimator model
        :param x_mean: a initial guess for the mean
        :param p_k: a initial guess for the covariance of the estimator
        :param h_function: a function that receives 3 parameters (x, y_algebraic, u) and returns an measurement.
        :param c_matrix: if h_function is not give, c_matrix is "C" from the measurement equation:
                         y_meas = C*[x y_alg] + D*u. note that it has to have n_x + n_y (algebraic) columns.
        :param DM r_v: process noise matrix
        :param DM r_n: measurement noise matrix
        :param implementation: options: 'standard' or 'square-root'. (default: 'standard')
        """
        self.model = model

        self._ell = self.model.n_x
        self.n_sigma_points = 2 * self._ell + 1

        self.h_function = None
        self.c_matrix = None
        self.d_matrix = None

        self.implementation = 'standard'

        self.x_mean = None
        self.p_k = None

        self.r_v = 0.
        self.r_n = 0.

        self.p = None
        self.theta = None
        self.y_guess = None

        self._types_fixed = False
        self._checked = False

        EstimatorAbstract.__init__(self, **kwargs)

        self.dataset = DataSet(name=self.model.name)
        self.dataset.data['x']['size'] = self.model.n_x
        self.dataset.data['x']['names'] = ['est_' + self.model.x_sym[i].name() for i in range(self.model.n_x)]

        self.dataset.data['P']['size'] = self.model.n_x ** 2
        self.dataset.data['P']['names'] = ['P_' + str(i) + str(j) for i in range(self.model.n_x) for j in
                                           range(self.model.n_x)]

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
            raise ValueError('Neither a measurement function "h_function" or a measurement matrix "c_matrix" was given')

        return True

    def estimate(self, t_k, y_k, u_k):
        if not self._checked:
            self._check()
            self._checked = True
        if not self._types_fixed:
            self._fix_types()
            self._types_fixed = True

        if self.implementation == 'standard':
            x_mean, p_k = self._estimate_standard_ukf(t_k, y_k, u_k)
        else:
            x_mean, p_k = self._estimate_square_root_ukf(t_k, y_k, u_k)

        self.x_mean = x_mean
        self.p_k = p_k

        self.dataset.insert_data('x', self.x_mean, t_k)
        self.dataset.insert_data('P', vec(self.p_k), t_k)

        return x_mean, p_k

    def _get_measurement_from_prediction(self, x, y, u):
        if self.h_function is not None:
            measurement_prediction = self.h_function(x, y, u)
        elif self.c_matrix is not None:
            d_matrix = 0. if self.d_matrix is None else self.d_matrix
            measurement_prediction = mtimes(self.c_matrix, vertcat(x, y)) + mtimes(d_matrix, u)
        else:
            raise ValueError('Neither a measurement function "h_function" or a measurement matrix "c_matrix" was given')
        return measurement_prediction

    def _get_sigma_points_and_weights(self, x_mean, x_cov):
        # Initialize variables
        sigma_points = []
        weights_m = []
        weights_c = []

        ell = self._ell

        # Tuning parameters
        alpha = 1e-3
        kappa = 0
        beta = 2
        lamb = alpha ** 2 * (ell + kappa)

        # Sigma points
        sqr_root_matrix = chol((ell + lamb) * x_cov).T
        sigma_points.append(x_mean)
        for i in range(self.n_sigma_points - 1):
            ind = i % ell
            sign = 1 if i < ell else -1
            sigma_points.append(x_mean + sign * sqr_root_matrix[:, ind])

        # Weights
        weights_m.append(lamb / (ell + lamb))
        weights_c.append(lamb / (ell + lamb) + (1 - alpha ** 2 + beta))
        for i in range(self.n_sigma_points - 1):
            weights_m.append(1 / (2 * (ell + lamb)))
            weights_c.append(1 / (2 * (ell + lamb)))

        return sigma_points, weights_m, weights_c

    def _priori_update_standard(self, x_mean, x_cov, u, p, theta):
        # obtain the weights
        sigma_points, weights_m, weights_c = self._get_sigma_points_and_weights(x_mean, x_cov)

        # Perform predictions via simulation
        simulation_results = []
        x_cal_x_k_at_k_minus_1 = []
        y_alg_cal_x_k_at_k_minus_1 = []
        for i in range(self.n_sigma_points):
            x_0_i = sigma_points[i]
            simulation_results_i = self.model.simulate(x_0=x_0_i, t_0=self.t, t_f=self.t + self.t_s,
                                                       u=u, p=p, theta=theta, y_0=self.y_guess)
            simulation_results.append(simulation_results_i)
            x_cal_x_k_at_k_minus_1.append(simulation_results_i.final_condition()[0])
            y_alg_cal_x_k_at_k_minus_1.append(simulation_results[i].final_condition()[1])

        # Obtain the statistics
        x_hat_k_minus = sum([weights_m[i] * x_cal_x_k_at_k_minus_1[i] for i in range(self.n_sigma_points)])

        p_k_minus = sum(
            [weights_c[i] * mtimes((x_cal_x_k_at_k_minus_1[i] - x_hat_k_minus),
                                   (x_cal_x_k_at_k_minus_1[i] - x_hat_k_minus).T)
             for i in range(self.n_sigma_points)]) + self.r_v

        y_cal_k_at_k_minus_1 = []
        for i in range(self.n_sigma_points):
            y_cal_k_at_k_minus_1.append(self._get_measurement_from_prediction(x_cal_x_k_at_k_minus_1[i],
                                                                              y_alg_cal_x_k_at_k_minus_1[i], u))

        y_hat_k_minus = sum([weights_m[i] * y_cal_k_at_k_minus_1[i] for i in range(self.n_sigma_points)])

        p_yk_yk = sum([weights_c[i] * mtimes((y_cal_k_at_k_minus_1[i] - y_hat_k_minus),
                                             (y_cal_k_at_k_minus_1[i] - y_hat_k_minus).T)
                       for i in range(self.n_sigma_points)]) + self.r_n

        p_xk_yk = sum([weights_c[i] * mtimes((x_cal_x_k_at_k_minus_1[i] - x_hat_k_minus),
                                             (y_cal_k_at_k_minus_1[i] - y_hat_k_minus).T)
                       for i in range(self.n_sigma_points)])

        # k_gain = mtimes(p_xk_yk, inv(p_yk_yk))
        k_gain = solve(p_yk_yk.T, p_xk_yk.T).T

        return x_hat_k_minus, p_k_minus, y_hat_k_minus, p_yk_yk, k_gain

    def _estimate_standard_ukf(self, t_k, y_k, u_k):
        x_mean = self.x_mean
        x_cov = self.p_k

        (x_hat_k_minus, p_k_minus,
         y_hat_k_minus, p_yk_yk, k_gain) = self._priori_update_standard(x_mean, x_cov, u=u_k,
                                                                        p=self.p, theta=self.theta)

        x_hat_k = x_hat_k_minus + mtimes(k_gain, (y_k - y_hat_k_minus))
        p_k = p_k_minus - mtimes(k_gain, mtimes(p_yk_yk, k_gain.T))

        return x_hat_k, p_k

    def _estimate_square_root_ukf(self, t_k, y_k, u_k):
        raise NotImplementedError

    def cholupdate(self, R, x, sign):
        import numpy as np
        p = np.size(x)
        x = x.T
        for k in range(p):
            if sign == '+':
                r = np.sqrt(R[k, k] ** 2 + x[k] ** 2)
            elif sign == '-':
                r = np.sqrt(R[k, k] ** 2 - x[k] ** 2)
            c = r / R[k, k]
            s = x[k] / R[k, k]
            R[k, k] = r
            if sign == '+':
                R[k, k + 1:p] = (R[k, k + 1:p] + s * x[k + 1:p]) / c
            elif sign == '-':
                R[k, k + 1:p] = (R[k, k + 1:p] - s * x[k + 1:p]) / c
            x[k + 1:p] = c * x[k + 1:p] - s * R[k, k + 1:p]
        return R
