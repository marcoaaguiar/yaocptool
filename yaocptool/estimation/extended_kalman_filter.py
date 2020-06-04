from casadi import DM, mtimes, jacobian, Function, vertcat, vec, inv, solve, horzcat, SX

from yaocptool import expm
from yaocptool.modelling import SystemModel, DataSet
from .estimator_abstract import EstimatorAbstract


class ExtendedKalmanFilter(EstimatorAbstract):
    def __init__(self, model, **kwargs):
        """Extended Kalman Filter for ODE and DAE systems implementation based on (1)

        (1) Mandela, R. K., Narasimhan, S., & Rengaswamy, R. (2009).
        Nonlinear State Estimation of Differential Algebraic Systems. Proceedings of the 2009 ADCHEM (Vol. 42). IFAC.
        http://doi.org/10.3182/20090712-4-TR-2008.00129

        :param SystemModel model: filter model
        :param float t_s: sampling time
        :param float t: current estimator time
        :param DM x_mean: current state estimation
        :param DM p_k: current covariance estimation
        :param DM r_v: process noise covariance matrix
        :param DM r_n: measurement noise covariance matrix
        :param DM y_guess: initial guess of the algebraic variables for the model simulation
        """
        self.model = model

        self.h_function = None  # casadi.Function
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

        self.verbosity = 1

        self._types_fixed = False
        self._checked = False

        EstimatorAbstract.__init__(self, **kwargs)

        self._p_k_aug = DM.eye(self.model.n_x + self.model.n_y)
        self._p_k_aug[: self.model.n_x, : self.model.n_x] = self.p_k
        if self.model.n_y > 0:
            if self.y_guess is not None:
                self._x_aug = vertcat(self.x_mean, self.y_guess)
            else:
                raise ValueError(
                    "If the model has algebraic it is necessary to provide a initial guess for it "
                    "('y_guess')."
                )
        else:
            self._x_aug = self.x_mean

        (
            self.augmented_model,
            self.a_aug_matrix_func,
            self.gamma_func,
            self.p_update_func,
        ) = self._create_augmented_model_and_p_update(model)

        # Data set
        self.dataset = DataSet(name=self.model.name)
        self.dataset.data["x"]["size"] = self.model.n_x
        self.dataset.data["x"]["names"] = [
            "ekf_" + self.model.x_sym[i].name() for i in range(self.model.n_x)
        ]

        self.dataset.data["y"]["size"] = self.model.n_y
        self.dataset.data["y"]["names"] = [
            "ekf_" + self.model.y_sym[i].name() for i in range(self.model.n_y)
        ]

        self.dataset.data["P"]["size"] = self.model.n_x ** 2
        self.dataset.data["P"]["names"] = [
            "ekf_P_" + str(i) + str(j)
            for i in range(self.model.n_x)
            for j in range(self.model.n_x)
        ]

        self.dataset.data["P_y"]["size"] = self.model.n_y ** 2
        self.dataset.data["P_y"]["names"] = [
            "ekf_P_y_" + str(i) + str(j)
            for i in range(self.model.n_y)
            for j in range(self.model.n_y)
        ]

        self.dataset.data["meas"]["size"] = self.n_meas
        self.dataset.data["meas"]["names"] = [
            "ekf_meas_" + str(i) for i in range(self.n_meas)
        ]

    @property
    def n_meas(self):
        """ Number of measurements

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
        if not self._checked:
            self._check()
            self._checked = True
        if not self._types_fixed:
            self._fix_types()
            self._types_fixed = True

        # Simulate the system to obtain the prediction for the states and algebraic variables
        sim_results = self.model.simulate(
            x_0=self.x_mean,
            t_0=self.t,
            t_f=self.t + self.t_s,
            u=u_k,
            p=self.p,
            theta=self.theta,
            y_0=self.y_guess,
            integrator_type="implicit",
        )
        x_pred, y_pred, u_f = sim_results.final_condition()
        x_aug_f = vertcat(x_pred, y_pred)

        # Covariance
        all_sym_values = self.model.put_values_in_all_sym_format(
            t=self.t, x=x_pred, y=y_pred, p=self.p, theta=self.theta
        )
        all_sym_values = all_sym_values

        a_aug_matrix = self.a_aug_matrix_func(*all_sym_values)
        transition_matrix = expm(a_aug_matrix * self.t_s)

        gamma_matrix = self.gamma_func(*all_sym_values)
        p_pred = self.p_update_func(
            transition_matrix, self._p_k_aug, gamma_matrix, self.r_v
        )

        # Obtain the matrix G (linearized measurement model)
        measurement_sym = self._get_measurement_from_prediction(
            self.model.x_sym, self.model.y_sym, self.model.u_sym
        )
        dh_dx_aug_sym = jacobian(
            measurement_sym, vertcat(self.model.x_sym, self.model.y_sym)
        )
        f_dh_dx_aug = Function(
            "dh_dx_aug",
            [self.model.x_sym, self.model.y_sym, self.model.u_sym],
            [dh_dx_aug_sym],
        )

        g_matrix = f_dh_dx_aug(x_pred, y_pred, u_f)

        # Obtain the filter gain
        k_gain = mtimes(
            mtimes(p_pred, g_matrix.T),
            inv(mtimes(g_matrix, mtimes(p_pred, g_matrix.T)) + self.r_n),
        )

        # Get measurement prediction
        meas_pred = self._get_measurement_from_prediction(x_pred, y_pred, u_f)

        # Correct prediction of the state estimation
        x_mean = x_aug_f + mtimes(k_gain, (y_k - meas_pred))
        meas_corr = self._get_measurement_from_prediction(
            x_mean[: self.model.n_x], x_mean[self.model.n_x :], u_f
        )
        if self.verbosity > 1:
            print("Predicted state: {}".format(x_pred))
            print("Prediction error: {}".format(y_k - meas_pred))
            print("Correction: {}".format(mtimes(k_gain, (y_k - meas_pred))))
            print("Corrected state: {}".format(x_mean))
            print("Measurement: {}".format(y_k))
            print("Corrected Meas.: {}".format(meas_corr))

        # Correct covariance prediction
        p_k = mtimes(
            DM.eye(self.augmented_model.n_x) - mtimes(k_gain, g_matrix), p_pred
        )

        # Save variables in local object
        self._x_aug = x_mean
        self.x_mean = x_mean[: self.model.n_x]
        self._p_k_aug = p_k
        self.p_k = p_k[: self.model.n_x, : self.model.n_x]

        # Save in the data set
        self.dataset.insert_data("x", t_k, self.x_mean)
        self.dataset.insert_data("y", t_k, x_mean[self.model.n_x :])
        self.dataset.insert_data("meas", t_k, meas_corr)
        self.dataset.insert_data("P", t_k, vec(self.p_k))
        self.dataset.insert_data("P_y", t_k, p_k[self.model.n_x :, self.model.n_x :])

        return self.x_mean, self.p_k

    def _create_augmented_model_and_p_update(self, model):
        """

        :type model: SystemModel
        """
        aug_model = SystemModel("aug_linearized_EKF_model")
        aug_model.include_state(self.model.x_sym)
        aug_model.include_state(self.model.y_sym)
        aug_model.include_control(self.model.u_sym)
        aug_model.include_parameter(self.model.p_sym)
        aug_model.include_theta(self.model.theta_sym)

        # remove u_par (self.model.u_par model.u_par)
        all_sym = list(self.model.all_sym)

        # Mean
        a_func = Function(
            "A_matrix", all_sym, [jacobian(self.model.ode, self.model.x_sym)]
        )
        b_func = Function(
            "B_matrix", all_sym, [jacobian(self.model.ode, self.model.y_sym)]
        )
        c_func = Function(
            "C_matrix", all_sym, [jacobian(self.model.alg, self.model.x_sym)]
        )
        d_func = Function(
            "D_matrix", all_sym, [jacobian(self.model.alg, self.model.y_sym)]
        )

        x_lin = aug_model.create_parameter("x_lin", self.model.n_x)
        y_lin = aug_model.create_parameter("y_lin", self.model.n_y)

        all_sym[1] = x_lin
        all_sym[2] = y_lin

        a_matrix = a_func(*all_sym)
        b_matrix = b_func(*all_sym)
        c_matrix = c_func(*all_sym)
        d_matrix = d_func(*all_sym)

        a_aug_matrix = vertcat(
            horzcat(a_matrix, b_matrix),
            horzcat(
                mtimes(-solve(d_matrix, c_matrix), a_matrix),
                mtimes(-solve(d_matrix, c_matrix), b_matrix),
            ),
        )

        x_aug = vertcat(model.x_sym, model.y_sym)
        aug_model.include_equations(ode=[mtimes(a_aug_matrix, x_aug)])

        # Covariance
        gamma = vertcat(DM.eye(self.model.n_x), -solve(d_matrix, c_matrix))

        trans_matrix_sym = SX.sym("trans_matrix", a_aug_matrix.shape)
        p_matrix_sym = SX.sym(
            "p_matrix", (model.n_x + model.n_y, model.n_x + model.n_y)
        )
        gamma_matrix_sym = SX.sym("gamma_matrix", gamma.shape)
        q_matrix_sym = SX.sym("q_matrix", (self.model.n_x, self.model.n_x))

        p_kpp = mtimes(
            trans_matrix_sym, mtimes(p_matrix_sym, trans_matrix_sym.T)
        ) + mtimes(gamma_matrix_sym, mtimes(q_matrix_sym, gamma_matrix_sym.T))

        a_aug_matrix_func = Function("trans_matrix", all_sym, [a_aug_matrix])
        gamma_func = Function("gamma", all_sym, [gamma])
        p_update_func = Function(
            "p_update",
            [trans_matrix_sym, p_matrix_sym, gamma_matrix_sym, q_matrix_sym],
            [p_kpp],
        )

        return aug_model, a_aug_matrix_func, gamma_func, p_update_func

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
