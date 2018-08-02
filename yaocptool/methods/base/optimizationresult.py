from collections import defaultdict
from functools import partial

try:
    import matplotlib.pyplot as plt
except:
    print('Failed to import matplotlib. Make sure that it is properly installed')

from casadi import horzcat, DM
from typing import List


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
        self.theta_opt_names = []

        self.x_interpolation_data = {'values': [], 'time': []}
        self.y_interpolation_data = {'values': [], 'time': []}
        self.u_interpolation_data = {'values': [], 'time': []}

        self.other_data = defaultdict(partial(defaultdict, {'values': [], 'time': []}))

        self.x_0 = []
        self.theta = {}
        self.p = []
        self.p_opt = []
        self.theta_opt = []
        self.eta = []

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
    @staticmethod
    def _plot_entry(t_vector, data_vector, row, label='', plot_style='plot', **kwargs):
        if plot_style not in ['plot', 'step']:
            raise ValueError('Plot style not recognized: "{}". Allowed : "plot" and "step"'.format(plot_style))
        if plot_style == 'plot':
            return plt.plot(t_vector.T, data_vector[row, :].T, label=label, **kwargs)
        elif plot_style == 'step':
            return plt.step(t_vector.T, data_vector[row, :].T, label=label, where='post', **kwargs)

    def first_control(self):
        """Return the first element of the control vector

        :rtype: DM
        """
        u_0 = self.u_interpolation_data['values'][0][0]
        return u_0

    def plot(self, plot_list, figures=None, show=True):
        """Plot the optimization result.
        It takes as input a list of dictionaries, each dictionary represents a plot.  In the dictionary use keyword 'x'
        to specify which states you want to print, the value of the dictionary should be a list of state to be printed.
        The keywords that are accepted are: 'x', 'y', 'u'
        :param list plot_list: List of dictionaries to generate the plots.
        :param list figures: OPTIONAL: list of figures to be plotted in. If not provided it will create new figures.
        :param bool show: OPTIONAL: select if matplotlib.pyplot.show should be applied after the plots.
        """
        if isinstance(plot_list, dict):
            plot_list = [plot_list]

        used_figures = []

        if self.is_valid:
            x_values = self.x_interpolation_data['values']
            y_values = self.y_interpolation_data['values']
            u_values = self.u_interpolation_data['values']

            x_values = horzcat(*[horzcat(*x_values[l]) for l in range(self.finite_elements)])
            y_values = horzcat(*[horzcat(*y_values[l]) for l in range(self.finite_elements)])
            u_values = horzcat(*[horzcat(*u_values[l]) for l in range(self.finite_elements)])
            theta_opt_values = horzcat(*self.theta_opt)

            t_x = self.x_interpolation_data['time']
            t_y = self.y_interpolation_data['time']
            t_u = self.u_interpolation_data['time']

            t_x = horzcat(*[horzcat(*t_x[l]) for l in range(self.finite_elements)])
            t_y = horzcat(*[horzcat(*t_y[l]) for l in range(self.finite_elements)])
            t_u = horzcat(*[horzcat(*t_u[l]) for l in range(self.finite_elements)])

            for k, entry in enumerate(plot_list):
                if figures is not None:
                    fig = plt.figure(figures[k].number)
                else:
                    fig = plt.figure(k)

                used_figures.append(fig)

                lines = []

                # Plot optimization x data
                if 'x' in entry:
                    x_indices = entry['x']
                    if x_indices == 'all':
                        x_indices = range(x_values.shape[0])
                    # for ind, value in enumerate(x_indices):
                    #     if isinstance(value, str) and not value == 'all':
                    #         x_indices.insert(self.x_names.index(value), x_indices.index(value))

                    for l in x_indices:
                        line = self._plot_entry(t_x, x_values, l, label=self.x_names[l], plot_style='plot')
                        lines.append(line)

                # Plot optimization y data
                if 'y' in entry:
                    y_indices = entry['y']
                    if y_indices == 'all':
                        y_indices = range(y_values.shape[0])
                    for l in y_indices:
                        line = self._plot_entry(t_y, y_values, l, label=self.y_names[l], plot_style='plot')
                        lines.append(line)

                # Plot optimization u data
                if 'u' in entry:
                    u_indices = entry['u']
                    if u_indices == 'all':
                        u_indices = range(u_values.shape[0])
                    for l in u_indices:
                        plot_style = 'step' if self.degree_control == 1 else 'plot'
                        line = self._plot_entry(t_u, u_values, l, label=self.u_names[l], plot_style=plot_style)
                        lines.append(line)

                # Plot theta_op
                if 'theta_opt' in entry:
                    theta_indices = entry['theta_opt']
                    if theta_indices == 'all':
                        theta_indices = range(theta_opt_values.shape[0])

                    for l in theta_indices:
                        plot_style = 'step'
                        line = self._plot_entry(horzcat(*self.time_breakpoints[:-1]), theta_opt_values, l,
                                                label=self.theta_opt_names[l],
                                                plot_style=plot_style)
                        lines.append(line)

                # Plot optimization any other data included in the OptimizationResult
                for key in set(entry.keys()).difference(['x', 'y', 'u', 'theta_opt']):
                    for l in entry[key]:
                        entry_time = horzcat(
                            *[horzcat(*self.other_data[key]['times'][i]) for i in range(self.finite_elements)])
                        entry_values = horzcat(
                            *[horzcat(*self.other_data[key]['values'][i]) for i in range(self.finite_elements)])
                        self._plot_entry(entry_time, entry_values, l, label=key + '[' + repr(l) + ']',
                                         plot_style='plot')
                plt.grid()
                axes = fig.axes
                axes[0].ticklabel_format(useOffset=False)
                plt.legend(ncol=4)

        if show:
            plt.show()

        return used_figures
