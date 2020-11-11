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

        self.implementation = "standard"

        self.x_mean = None
        self.p_k = None

        self.r_v = 0.0
        self.r_n = 0.0

        self.p = None
        self.theta = None
        self.y_guess = None

        self._types_fixed = False
        self._checked = False

        EstimatorAbstract.__init__(self, **kwargs)

        # Data set
        self.dataset = DataSet(name=self.model.name)
        self.dataset.data["x"]["size"] = self.model.n_x
        self.dataset.data["x"]["names"] = [
            "ukf_" + self.model.x_sym[i].name() for i in range(self.model.n_x)
        ]

        self.dataset.data["y"]["size"] = self.model.n_y
        self.dataset.data["y"]["names"] = [
            "ukf_" + self.model.y_sym[i].name() for i in range(self.model.n_y)
        ]

        self.dataset.data["P"]["size"] = self.model.n_x ** 2
        self.dataset.data["P"]["names"] = [
            "ukf_P_" + str(i) + str(j)
            for i in range(self.model.n_x)
            for j in range(self.model.n_x)
        ]

        self.dataset.data["P_y"]["size"] = self.model.n_y ** 2
        self.dataset.data["P_y"]["names"] = [
            "ukf_P_y_" + str(i) + str(j)
            for i in range(self.model.n_y)
            for j in range(self.model.n_y)
        ]

        self.dataset.data["meas"]["size"] = self.n_meas
        self.dataset.data["meas"]["names"] = [
            "ukf_meas_" + str(i) for i in range(self.n_meas)
        ]

        # Choose the UKF implementation
        if self.implementation == "standard":
            self.estimate = self._estimate_standard_ukf
        else:
            self.estimate = self._estimate_square_root_ukf

    @property
    def n_meas(self):
        """Number of measurements

        :rtype: int
        :return: Number of measurements
        """
        if self.h_function is not None:
            return self.h_function.numel_out()
        elif self.c_matrix is not None:
            return self.c_matrix.shape[0]
        else:
            raise Exception(
                "The estimator has no measurements information, neither 'h_function' or 'c_matrix' "
                "were given."
            )

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
        raise Exception("This function should have been replaced")

    def _estimate_standard_ukf(self, t_k, y_k, u_k):
        if not self._checked:
            self._check()
            self._checked = True
        if not self._types_fixed:
            self._fix_types()
            self._types_fixed = True

        x_mean = self.x_mean
        x_cov = self.p_k

        # obtain the weights
        sigma_points, weights_m, weights_c = self._get_sigma_points_and_weights(
            x_mean, x_cov
        )

        # Obtain the unscented transformation points via simulation
        simulation_results = []
        x_ut_list = []
        y_ut_list = []
        x_aug_ut_list = []
        meas_ut_list = []
        for i in range(self.n_sigma_points):
            x_0_i = sigma_points[i]
            simulation_results_i = self.model.simulate(
                x_0=x_0_i,
                t_0=self.t,
                t_f=self.t + self.t_s,
                u=u_k,
                p=self.p,
                theta=self.theta,
                y_0=self.y_guess,
            )
            simulation_results.append(simulation_results_i)
            x_ut_list.append(simulation_results_i.final_condition()["x"])
            y_ut_list.append(simulation_results[i].final_condition()["y"])
            x_aug_ut_list.append(vertcat(x_ut_list[-1], y_ut_list[-1]))

            meas_ut_list.append(
                self._get_measurement_from_prediction(x_ut_list[i], y_ut_list[i], u_k)
            )

        # Obtain the means
        x_aug_pred = sum(
            weights_m[i] * x_aug_ut_list[i] for i in range(self.n_sigma_points)
        )

        x_pred = x_aug_pred[: self.model.n_x]
        meas_pred = sum(
            weights_m[i] * meas_ut_list[i] for i in range(self.n_sigma_points)
        )

        # Compute the covariances
        cov_x_aug_pred = sum(
            weights_c[i]
            * mtimes((x_aug_ut_list[i] - x_aug_pred), (x_aug_ut_list[i] - x_aug_pred).T)
            for i in range(self.n_sigma_points)
        )
        cov_x_aug_pred[: self.model.n_x, : self.model.n_x] += self.r_v

        cov_meas_pred = (
            sum(
                weights_c[i]
                * mtimes((meas_ut_list[i] - meas_pred), (meas_ut_list[i] - meas_pred).T)
                for i in range(self.n_sigma_points)
            )
            + self.r_n
        )

        cov_xmeas_pred = sum(
            weights_c[i]
            * mtimes((x_aug_ut_list[i] - x_aug_pred), (meas_ut_list[i] - meas_pred).T)
            for i in range(self.n_sigma_points)
        )

        # Calculate the gain
        k_gain = solve(cov_meas_pred.T, cov_xmeas_pred.T).T

        # Correct prediction of the state estimation
        x_mean = x_aug_pred + mtimes(k_gain, (y_k - meas_pred))
        meas_corr = self._get_measurement_from_prediction(
            x_mean[: self.model.n_x], x_mean[self.model.n_x :], u_k
        )
        print("Predicted state: {}".format(x_pred))
        print("Prediction error: {}".format(y_k - meas_pred))
        print("Correction: {}".format(mtimes(k_gain, (y_k - meas_pred))))
        print("Corrected state: {}".format(x_mean))
        print("Measurement: {}".format(y_k))
        print("Corrected Meas.: {}".format(meas_corr))

        # Correct covariance prediction
        cov_x_aug = cov_x_aug_pred - mtimes(k_gain, mtimes(cov_meas_pred, k_gain.T))

        self.x_mean = x_mean
        self.p_k = cov_x_aug

        # Save variables in local object
        self._x_aug = x_mean
        self.x_mean = x_mean[: self.model.n_x]
        self._p_k_aug = cov_x_aug_pred
        self.p_k = cov_x_aug[: self.model.n_x, : self.model.n_x]

        # Save in the data set
        self.dataset.insert_data("x", t_k, self.x_mean)
        self.dataset.insert_data("y", t_k, x_mean[self.model.n_x :])
        self.dataset.insert_data("meas", t_k, meas_corr)
        self.dataset.insert_data("P", t_k, vec(self.p_k))
        self.dataset.insert_data(
            "P_y", t_k, cov_x_aug[self.model.n_x :, self.model.n_x :]
        )

        return x_mean, cov_x_aug

    def _estimate_square_root_ukf(self, t_k, y_k, u_k):
        raise NotImplementedError

    @staticmethod
    def cholupdate(r_matrix, x, sign):
        import numpy as np

        p = np.size(x)
        x = x.T
        for k in range(p):
            if sign == "+":
                r = np.sqrt(r_matrix[k, k] ** 2 + x[k] ** 2)
            elif sign == "-":
                r = np.sqrt(r_matrix[k, k] ** 2 - x[k] ** 2)
            else:
                raise ValueError(
                    "sign can be '-' or '+', value given = {}".format(sign)
                )
            c = r / r_matrix[k, k]
            s = x[k] / r_matrix[k, k]
            r_matrix[k, k] = r
            if sign == "+":
                r_matrix[k, k + 1 : p] = (r_matrix[k, k + 1 : p] + s * x[k + 1 : p]) / c
            elif sign == "-":
                r_matrix[k, k + 1 : p] = (r_matrix[k, k + 1 : p] - s * x[k + 1 : p]) / c
            x[k + 1 : p] = c * x[k + 1 : p] - s * r_matrix[k, k + 1 : p]
        return r_matrix

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
        for _ in range(self.n_sigma_points - 1):
            weights_m.append(1 / (2 * (ell + lamb)))
            weights_c.append(1 / (2 * (ell + lamb)))

        return sigma_points, weights_m, weights_c

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
