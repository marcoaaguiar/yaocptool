from collections import defaultdict
from functools import partial
from yaocptool.modelling import DataSet
from casadi import horzcat, DM

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Failed to import matplotlib. Make sure that it is properly installed')
    plt = None


class OptimizationResult:
    def __init__(self, **kwargs):
        # Raw Information
        self.raw_solution_dict = {}
        self.raw_decision_variables = None  # type: list

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

        self.objective_opt_problem = None  # type: DM
        self.v_final = None  # type: DM
        self.x_c_final = None  # type: DM
        self.constraints_values = None  # type: DM

        self.x_names = []
        self.y_names = []
        self.u_names = []
        self.theta_opt_names = []

        self.x_data = {'values': [], 'time': []}
        self.y_data = {'values': [], 'time': []}
        self.u_data = {'values': [], 'time': []}

        self.other_data = defaultdict(partial(dict, [('values', []), ('time', [])]))

        self.x_0 = DM([])
        self.theta = {}
        self.p = DM([])
        self.p_opt = DM([])
        self.theta_opt = []
        self.eta = DM([])

        self._dataset = None

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = self.to_dataset()
        return self._dataset

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

    def first_control(self):
        """Return the first element of the control vector

        :rtype: DM
        """
        u_0 = self.u_data['values'][0][0]
        return u_0

    def to_dataset(self):
        """
            Return a dataset with the data of x, y, and u

        :rtype: DataSet
        """
        dataset = DataSet(name=self.problem_name + '_dataset')

        dataset.max_delta_t = (self.t_f - self.t_0) / self.finite_elements
        dataset.plot_style = 'plot'

        dataset.create_entry('x', size=len(self.x_names), names=self.x_names)
        dataset.create_entry('y', size=len(self.y_names), names=self.y_names)
        dataset.create_entry('u', size=len(self.u_names), names=self.u_names)
        dataset.create_entry('theta_opt', size=len(self.theta_opt_names), names=self.theta_opt_names, plot_style='step')

        for entry in self.other_data:
            size = self.other_data[entry]['values'][0][0].shape[0]
            dataset.create_entry(entry, size=size, names=[entry + '_' + str(i) for i in range(size)])

        if self.degree_control > 1:
            dataset.data['u']['plot_style'] = 'plot'
        else:
            dataset.data['u']['plot_style'] = 'step'

        x_times = self.x_data['time'] + [[self.t_f]]
        for el in range(self.finite_elements + 1):
            time = horzcat(*x_times[el])
            values = horzcat(*self.x_data['values'][el])
            dataset.insert_data('x', time, values)

        for el in range(self.finite_elements):
            time_y = horzcat(*self.y_data['time'][el])
            values_y = horzcat(*self.y_data['values'][el])
            dataset.insert_data('y', time_y, values_y)

            time_u = horzcat(*self.u_data['time'][el])
            values_u = horzcat(*self.u_data['values'][el])
            dataset.insert_data('u', time_u, values_u)

        dataset.insert_data('theta_opt', time=horzcat(*self.time_breakpoints[:-1]), value=horzcat(*self.theta_opt))

        for entry in self.other_data:
            for el in range(self.finite_elements):
                dataset.insert_data(entry,
                                    time=horzcat(*self.other_data[entry]['time'][el]),
                                    value=horzcat(*self.other_data[entry]['values'][el]))

        return dataset

    def plot(self, plot_list, figures=None, show=True):
        """Plot the optimization result.
        It takes as input a list of dictionaries, each dictionary represents a plot.  In the dictionary use keyword 'x'
        to specify which states you want to print, the value of the dictionary should be a list of state to be printed.
        The keywords that are accepted are: 'x', 'y', 'u'
        :param list plot_list: List of dictionaries to generate the plots.
        :param list figures: OPTIONAL: list of figures to be plotted in. If not provided it will create new figures.
        :param bool show: OPTIONAL: select if matplotlib.pyplot.show should be applied after the plots.
        """
        return self.dataset.plot(plot_list, figures, show)
