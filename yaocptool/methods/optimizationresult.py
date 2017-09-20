import matplotlib.pyplot as plt
from casadi import horzcat, DM
from typing import List

class OptimizationResult:
    def __init__(self, raw_solution_dict=None, solution_method=None, **kwargs):
        # solution_method:
        self.raw_solution_dict = raw_solution_dict
        self.raw_decision_variables = raw_solution_dict['x'] # type: List[DM]
        self.finite_elements = -1
        self.degree = -1
        self.degree_control = -1
        self.x_values = []
        self.y_values = []
        self.u_values = []
        self.t_0 = -1
        self.t_f = -1
        self.discretization_scheme = ''
        self.time_breakpoints = []
        self.collocation_points = []

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if solution_method is not None:
            self._get_attributes_from_solution_method(solution_method)
        self._extract_from_raw_data(self.raw_decision_variables, solution_method)

    @property
    def is_collocation(self):
        if self.discretization_scheme == '':
            raise Exception('discretization_scheme not defined yet')
        return self.discretization_scheme == 'collocation'

    @property
    def is_valid(self):
        for attr in ['finite_elements', 'degree', 'degree_control', 't_0', 't_f', 'time_breakpoints']:
            if getattr(self, attr) < 0:
                raise Exception('{} attribute {} is lower than 0'.format(self.__class__.__name__, attr))
        return True

    def _get_attributes_from_solution_method(self, solution_method):
        for attr in ['finite_elements', 'degree', 'degree_control', 'time_breakpoints']:
            attr_value = getattr(solution_method, attr)
            setattr(self, attr, attr_value)

        self.collocation_points = solution_method.collocation_points(solution_method.degree)

        for attr in ['t_0', 't_f']:
            attr_value = getattr(solution_method.problem, attr)
            setattr(self, attr, attr_value)

        self.discretization_scheme = solution_method



    def _extract_from_raw_data(self, raw_data, solution_method):
        self.x_values, self.y_values, self.u_values = solution_method.splitXYandU(raw_data)

    # Plot
    def plot(self, plot_list, t_states=None):
        if self.is_valid:
            if self.is_collocation:
                self.plot_collocation(plot_list, t_states)
            else:
                self.plot_multiple_shooting(plot_list, t_states)

    def plot_multiple_shooting(self, plot_list, t_states=None):
        x_values, y_values, u_values = self.x_values, self.y_values, self.u_values
        t_x = self.time_breakpoints
        t_yu = self.time_breakpoints[:-1]

        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)
            if 'x' in entry:
                for i in entry['x']:
                    plt.step(t_x, horzcat(*x_values)[i, :].T, where='post')
                plt.legend(['x[' + repr(i) + ']' for i in entry['x']])
            if 'y' in entry:
                for i in entry['y']:
                    plt.step(t_yu, horzcat(*y_values)[i, :].T, where='post')
                plt.legend(['y[' + repr(i) + ']' for i in entry['y']])

            if 'u' in entry:
                for i in entry['u']:
                    plt.step(t_yu, horzcat(*u_values)[i, :].T, where='post')
                plt.legend(['u[' + repr(i) + ']' for i in entry['u']])
            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            k += 1
        plt.show()

    def plot_collocation(self, plot_list, t_states=None):
        pass
