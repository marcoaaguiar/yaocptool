class DiscretizationSchemeBase:
    def __init__(self, solution_method):
        self.solution_method = solution_method

    @property
    def model(self):
        return self.solution_method.model

    @property
    def problem(self):
        return self.solution_method.problem

    @property
    def degree(self):
        return self.solution_method.degree

    @property
    def degree_control(self):
        return self.solution_method.degree_control

    @property
    def finite_elements(self):
        return self.solution_method.finite_elements

    def splitXandU(self, results_vector, all_subinterval=False):
        x_values, _, u_values = self.splitXYandU(results_vector, all_subinterval)
        return x_values, u_values

    def splitXYandU(self, V, all_subinterval=False):
        raise NotImplementedError

    def discretize(self, x_0=None, p=[], theta=None):
        raise NotImplementedError
