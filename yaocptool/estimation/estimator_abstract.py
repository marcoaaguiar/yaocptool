class EstimatorAbstract:
    def __init__(self, **kwargs):
        self.t_s = 1.0
        self.t = 0.0

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def estimate(self, t_k, y_k, u_k):
        """
        Estimate the state given the measurement y_k and the control u_k
        :param DM t_k: time of the measurements
        :param DM y_k: measurement
        :param DM u_k: control
        """
        raise NotImplementedError
