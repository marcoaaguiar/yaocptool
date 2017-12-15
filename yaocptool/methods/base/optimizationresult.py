from collections import defaultdict

import matplotlib.pyplot as plt
from casadi import horzcat, DM
from typing import List


# TODO: Fix plot u data

class OptimizationResult:
    def __init__(self, **kwargs):
        # Raw Information
        self.raw_solution_dict = {}
        self.raw_decision_variables = None  # type: List[DM]

        # Data from the method
        self.method_name = ''  # type: str
        self.discretization_scheme = ''  # type: str
        self.finite_elements = -1  # type: int
        self.degree = -1
        self.degree_control = -1

        # From problem
        self.problem_name = ''
        self.t_0 = -1
        self.t_f = -1

        self.time_breakpoints = []
        self.collocation_points = []

        self.objective = None  # type: DM
        self.constraints_values = None  # type: DM

        self.x_names = []
        self.y_names = []
        self.z_names = []
        self.u_names = []

        self.x_breakpoints_data = {'values': [], 'time': []}
        self.y_breakpoints_data = {'values': [], 'time': []}
        self.u_breakpoints_data = {'values': [], 'time': []}
        self.x_interpolation_data = {'values': [], 'time': []}
        self.y_interpolation_data = {'values': [], 'time': []}
        self.u_interpolation_data = {'values': [], 'time': []}
        self.other_data = defaultdict(lambda: {'values': [], 'time': []})

        self.x_0 = []
        self.theta = {}
        self.p = []

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    @property
    def is_collocation(self):
        if self.discretization_scheme == '':
            raise Exception('discretization_scheme not defined yet')
        return self.discretization_scheme == 'collocation'

    @property
    def is_valid(self):
        for attr in ['finite_elements', 'degree', 'degree_control', 't_0', 't_f']:
            if getattr(self, attr) < 0:
                raise Exception('{} attribute {} is lower than 0'.format(self.__class__.__name__, attr))
        if len(self.time_breakpoints) < 1:
            raise Exception('{} attribute {} (list) is empty'.format(self.__class__.__name__, 'time_breakpoints'))
        for attr in ['objective']:
            if getattr(self, attr) is None:
                raise Exception('{} attribute {} is None'.format(self.__class__.__name__, attr))
        return True

    # Plot
    def plot(self, plot_list):
        if isinstance(plot_list, dict):
            plot_list = [plot_list]

        if self.is_valid:
            self._plot_interpolation(plot_list)

    @staticmethod
    def _plot_entry(t_vector, data_vector, row, label='', plot_style='plot'):
        if plot_style not in ['plot', 'step']:
            raise ValueError('Plot style not recognized: "{}". Allowed : "plot" and "step"'.format(plot_style))
        if plot_style == 'plot':
            return plt.plot(t_vector.T, data_vector[row, :].T, label=label)
        elif plot_style == 'step':
            return plt.step(t_vector.T, data_vector[row, :].T, label=label, where='post')

    def _plot_interpolation(self, plot_list):
        x_values = self.x_interpolation_data['values']
        y_values = self.y_interpolation_data['values']
        u_values = self.u_interpolation_data['values']

        x_values = horzcat(*[horzcat(*x_values[l]) for l in range(self.finite_elements)])
        y_values = horzcat(*[horzcat(*y_values[l]) for l in range(self.finite_elements)])
        u_values = horzcat(*[horzcat(*u_values[l]) for l in range(self.finite_elements)])

        t_x = self.x_interpolation_data['time']
        t_y = self.y_interpolation_data['time']
        t_u = self.u_interpolation_data['time']

        t_x = horzcat(*[horzcat(*t_x[l]) for l in range(self.finite_elements)])
        t_y = horzcat(*[horzcat(*t_y[l]) for l in range(self.finite_elements)])
        t_u = horzcat(*[horzcat(*t_u[l]) for l in range(self.finite_elements)])

        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)
            lines = []
            # Plot optimization x data
            if 'x' in entry:
                for l in entry['x']:
                    line = self._plot_entry(t_x, x_values, l, label=self.x_names[l], plot_style='plot')
                    lines.append(line)

            # Plot optimization y data
            if 'y' in entry:
                for l in entry['y']:
                    line = self._plot_entry(t_y, y_values, l, label=self.y_names[l], plot_style='plot')
                    lines.append(line)

            # Plot optimization u data
            if 'u' in entry:
                for l in entry['u']:
                    plot_style = 'step' if self.degree_control == 1 else 'plot'
                    line = self._plot_entry(t_u, u_values, l, label=self.u_names[l], plot_style=plot_style)
                    lines.append(line)

            # Plot optimization any other data included in the OptimizationResult
            for key in set(entry.keys()).difference(['x', 'y', 'u']):
                for l in entry[key]:
                    entry_time = horzcat(
                        *[horzcat(*self.other_data[key]['times'][i]) for i in range(self.finite_elements)])
                    entry_values = horzcat(
                        *[horzcat(*self.other_data[key]['values'][i]) for i in range(self.finite_elements)])
                    self._plot_entry(entry_time, entry_values, l, label=key + '[' + repr(l) + ']', plot_style='plot')
            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            plt.legend()
        plt.show()
