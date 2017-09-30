class DiscretizationSchemeBase:
    def __init__(self, solution_method):
        """

        :type solution_method: yaocptool.methods.solutionmethodsbase.SolutionMethodsBase
        """
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

    @property
    def delta_t(self):
        return self.solution_method.delta_t

    @property
    def time_breakpoints(self):
        return self.solution_method.time_breakpoints

    def splitXandU(self, results_vector, all_subinterval=False):
        x_values, _, u_values = self.splitXYandU(results_vector, all_subinterval)
        return x_values, u_values

    def splitXYandU(self, V, all_subinterval=False):
        raise NotImplementedError

    def discretize(self, x_0=None, p=[], theta=None):
        raise NotImplementedError

    def set_data_to_optimization_result_from_raw_data(self, optimization_result, raw_solution_dict):
        """
        Set the raw data received from the solver and put it in the Optimization Result object
        :type optimization_result: yaocptool.methods.optimizationresult.OptimizationResult
        :type raw_solution_dict: dict
        """
        raise NotImplemented
