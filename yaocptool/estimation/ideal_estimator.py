from casadi import DM

from yaocptool.estimation.estimator_abstract import EstimatorAbstract


class IdealEstimator(EstimatorAbstract):
    def __init__(self, t_s: float, t_0: float, n_x: int):
        self.t_s = t_s
        self.t = t_0

        self.n_x = n_x

    def estimate(self, t_k, y_k, u_k):
        """
        Estimate the state given the measurement y_k and the control u_k
        :param DM t_k: time of the measurements
        :param DM y_k: measurement
        :param DM u_k: control
        """

        x_k = y_k[: self.n_x]
        cov_x_k = DM.zeros(x_k.shape)

        return x_k, cov_x_k
