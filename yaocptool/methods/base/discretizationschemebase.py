# noinspection PyUnresolvedReferences
from typing import List
from casadi import DM, vertcat


class DiscretizationSchemeBase:
    def __init__(self, solution_method):
        """

        :type solution_method: yaocptool.methods.base.solutionmethodsbase.SolutionMethodsBase
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

    def split_x_and_u(self, results_vector, all_subinterval=False):
        x_values, _, u_values = self.split_x_y_and_u(results_vector, all_subinterval)
        return x_values, u_values

    def vectorize(self, vector):
        if len(vector) > 0:
            if not isinstance(vector[0], list):
                return vertcat(*vector)
            else:
                return vertcat(*[self.vectorize(sub_vector) for sub_vector in vector])
        else:
            return vertcat(vector)

    def split_x_y_and_u(self, v, all_subinterval=False):
        raise NotImplementedError

    def discretize(self, x_0=None, p=None, theta=None):
        raise NotImplementedError

    def create_nlp_symbolic_variables(self):
        """
        Create the symbolic variables that will be used by the NLP problem
        :rtype: (DM, List[List[DM]], List(List(DM)), List(DM), DM, DM, DM)
        """
        raise NotImplementedError

    def _create_variables_bound_vectors(self):
        """
        Return two items: the vector of lower bounds and upper bounds
        :rtype: (DM, DM)
        """
        raise NotImplementedError

    def get_system_at_given_times(self, x, y, u, time_dict=None, p=None, theta=None, functions=None,
                                  start_at_t_0=False):
        # TODO: calculate quadratures, for error evaluation of aug lagrange
        raise NotImplementedError

    def set_data_to_optimization_result_from_raw_data(self, optimization_result, raw_solution_dict):
        """
        Set the raw data received from the solver and put it in the Optimization Result object
        :type optimization_result: yaocptool.methods.optimizationresult.OptimizationResult
        :type raw_solution_dict: dict
        """
        raise NotImplementedError
