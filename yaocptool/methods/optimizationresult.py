import matplotlib.pyplot as plt
from casadi import horzcat

class OptimizationResult:
    def __init__(self, V_sol=None, solution_method=None, **kwargs):

        self.V_sol = V_sol
        self.finite_elements = -1
        self.degree = -1
        self.degree_control = -1
        self.X = []
        self.Y = []
        self.U = []
        self.t_0 = -1
        self.t_f = -1
        self.discretization_method = ''
        self.time_breakpoints = []

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if solution_method is not None:
            self.get_attributes_from_solution_method(solution_method)

    @property
    def is_collocation(self):
        if self.discretization_method == '':
            raise Exception('discretization_method not defined yet')
        return self.discretization_method == 'collocation'

    def get_attributes_from_solution_method(self, solution_method):
        for attr in ['finite_elements', 'degree', 'degree_control', 't_0', 't_f', 'time_breakpoints']:
            attr_value = getattr(solution_method, attr)
            setattr(self, attr, attr_value)
        self.discretization_method = solution_method

    def is_valid(self):
        for attr in ['finite_elements', 'degree', 'degree_control', 't_0', 't_f', 'time_breakpoints']:
            if getattr(self, attr) < 0:
                raise Exception('{} attribute {} is lower than 0'.format(self.__class__.__name__, attr))

    def plot(self, plot_list, t_states=None):
        if self.is_collocation:
            self.plot_collocation()
        if isinstance(plot_list, int):
            plot_list = [plot_list]
        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)
            if 'x' in entry:
                for i in entry['x']:
                    plt.plot(t_states, horzcat(*X)[i, :].T)
                plt.legend(['x[' + `i` + ']' for i in entry['x']])
            if 'y' in entry:
                for i in entry['y']:
                    plt.plot(t_states[:len(U)], horzcat(*Y)[i, :].T)
                plt.legend(['y[' + `i` + ']' for i in entry['y']])

            if 'u' in entry:
                for i in entry['u']:
                    plt.plot(t_states[:len(U)], horzcat(*U)[i, :].T)
                plt.legend(['u[' + `i` + ']' for i in entry['u']])
            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            k += 1
        # plt.ion()
        plt.show()
    def plot_collocation(self):
        raise
