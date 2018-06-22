from casadi import DM, mtimes, jacobian, Function, vertcat, reshape, vec, inv, solve

from yaocptool.modelling import SystemModel, DataSet
from .estimator_abstract import EstimatorAbstract


class ExtendedKalmanFilter(EstimatorAbstract):
    def __init__(self, model, **kwargs):
        """

        :param SystemModel model:
        """
        self.model = model

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
        self._p_k_aug = DM.eye(self.model.n_x + self.model.n_y)
        self._p_k_aug[:self.model.n_x, :self.model.n_x] = self.p_k
        self.augmented_model = self._create_augmented_model(model)

        self.dataset = DataSet(name=self.model.name)
        self.dataset.data['x']['size'] = self.model.n_x
        self.dataset.data['x']['names'] = ['est_' + self.model.x_sym[i].name() for i in range(self.model.n_x)]

        self.dataset.data['P']['size'] = self.model.n_x ** 2
        self.dataset.data['P']['names'] = ['P_' + str(i) + str(j) for i in range(self.model.n_x) for j in
                                           range(self.model.n_x)]

    def _create_augmented_model(self, model):
        """

        :type model: SystemModel
        """
        aug_model = model.get_copy()

        # means
        ell = vertcat(DM.eye(self.model.n_x),
                      -solve(jacobian(self.model.alg, self.model.y_sym),
                             jacobian(self.model.alg, self.model.x_sym)))

        p_vec = aug_model.create_state('P', (model.n_x + model.n_y) ** 2)
        p_matrix = reshape(p_vec, ((model.n_x + model.n_y), (model.n_x + model.n_y)))

        x_aug = vertcat(model.x_sym, model.y_sym)
        f_aug = vertcat(model.ode, model.alg)

        df_aug_dx_aug = jacobian(f_aug, x_aug)

        dp_dt = mtimes(df_aug_dx_aug, p_matrix) + mtimes(p_matrix, df_aug_dx_aug.T) + mtimes(ell,
                                                                                             mtimes(self.r_v, ell.T))
        aug_model.include_system_equations(ode=[vec(dp_dt)])
        return aug_model

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
        x_aug_hat_minus, p_hat_minus, y_hat_minus, k_gain, dh_dx_aug = self._priori_update(self.x_mean, self._p_k_aug,
                                                                                           u_k, self.p, self.theta)

        x_mean = x_aug_hat_minus + mtimes(k_gain, (y_k - y_hat_minus))
        p_k = mtimes(DM.eye(self.model.n_x + self.model.n_y) - mtimes(k_gain, dh_dx_aug),
                     mtimes(p_hat_minus,
                            (DM.eye(self.model.n_x + self.model.n_y) - mtimes(k_gain, dh_dx_aug)).T)
                     ) + mtimes(k_gain, mtimes(self.r_n, k_gain.T))

        self.x_mean = x_mean[:self.model.n_x]
        self._p_k_aug = p_k
        self.p_k = p_k[:self.model.n_x, :self.model.n_x]

        self.dataset.insert_data('x', self.x_mean, t_k)
        self.dataset.insert_data('P', vec(self.p_k), t_k)

        return self.x_mean, self.p_k

    def _priori_update(self, x_k, p_k, u, p, theta):
        x_0 = vertcat(x_k, vec(p_k))
        sim_results = self.augmented_model.simulate(x_0=x_0, t_0=self.t, t_f=self.t + self.t_s,
                                                    u=u, p=p, theta=theta, y_0=self.y_guess)
        x_f, y_f, u_f = sim_results.final_condition()
        x_hat_aug_minus = x_f[:self.model.n_x + self.model.n_y]
        x_hat_minus = x_f[:self.model.n_x]
        p_hat_minus = reshape(x_f[self.model.n_x:],
                              ((self.model.n_x + self.model.n_y), (self.model.n_x + self.model.n_y)))

        dh_dx_aug_sym = jacobian(self._get_measurement_from_prediction(self.model.x_sym,
                                                                       self.model.y_sym,
                                                                       self.model.u_sym),
                                 vertcat(self.model.x_sym, self.model.y_sym))
        f_dh_dx_aug = Function('dh_dx_aug', [self.model.x_sym, self.model.y_sym, self.model.u_sym], [dh_dx_aug_sym])

        dh_dx_aug = f_dh_dx_aug(x_hat_minus, y_f, u_f)
        k_gain = mtimes(mtimes(p_hat_minus, dh_dx_aug.T),
                        inv(self.r_n + mtimes(dh_dx_aug, mtimes(p_hat_minus, dh_dx_aug.T))))
        y_hat_minus = self._get_measurement_from_prediction(x_hat_minus, y_f, u_f)
        return x_hat_aug_minus, p_hat_minus, y_hat_minus, k_gain, dh_dx_aug

    def _get_measurement_from_prediction(self, x, y, u):
        if self.h_function is not None:
            measurement_prediction = self.h_function(x, y, u)
        elif self.c_matrix is not None:
            d_matrix = 0. if self.d_matrix is None else self.d_matrix
            measurement_prediction = mtimes(self.c_matrix, vertcat(x, y)) + mtimes(d_matrix, u)
        else:
            raise ValueError('Neither a measurement function "h_function" or a measurement matrix "c_matrix" was given')
        return measurement_prediction
