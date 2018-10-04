from __future__ import print_function
import copy
from collections import defaultdict
from functools import partial
from casadi import horzcat, vertcat, DM, sum2

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Failed to import matplotlib. Make sure that it is properly installed')
    plt = None


class DataSet:
    def __init__(self, name='dataset', **kwargs):
        """
            Generic time dependent data storage.
            The data is stored in the self.data dictionary.
            self.data['entry_name']['time'] is a row vector
            self.data['entry_name']['values'] is a matrix with the same number of columns as the time vector, and
            rows equal to self.data['entry_name']['size'].
            The data can be more easily managed using create_entry, get_entry, insert_data.


        :param str name: name of th dataset
        :param str plot_style: default plot style. plot = linear interpolation, step = piecewise constant
        ('plot' | 'step')
        :param float max_sampling_time: maximum expected distance between two time data. This is used to detect
        discontinuity on the data, and plot it separately.
        """
        self.name = name
        self.data = defaultdict(partial(dict, [('time', DM([])), ('names', None), ('values', DM([])), ('size', None)]))
        self.plot_style = 'step'
        self.max_delta_t = None

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def create_entry(self, entry, size, names=None):
        """
            Create an entry in the dataset
        :param entry: entry name
        :param size: number of rows in the vector
        :param list names: name for each row, it should be a list with size 'size'. If 'names' is not given,
        then the name list [entry_1, entry_2, ..., entry_size]
        """
        if names is None:
            names = [entry + '_' + str(i) for i in range(size)]
        self.data[entry]['size'] = size
        self.data[entry]['names'] = names

    def get_entry(self, entry):
        """
            Return the time and values for a given entry.
        :param str entry: entry name
        :return: entry time, entry value
        :rtype: tuple
        """
        return self.data[entry]['time'], self.data[entry]['values']

    def get_entry_names(self, entry):
        """
            Get list of names of an entry
        :param entry:
        :rtype: list
        """
        return self.data[entry]['names']

    def get_entry_size(self, entry):
        """
            Get size of an entry
        :param entry:
        :return:
        """
        return self.data[entry]['size']

    def insert_data(self, entry, value, time):
        value = vertcat(value)

        # TODO: Here is a correction for a BUG on CASADI horzcat with DM([]), refactor when CASADI corrects this
        if self.data[entry]['values'].numel() > 0:
            self.data[entry]['values'] = horzcat(self.data[entry]['values'], value)
            self.data[entry]['time'] = horzcat(self.data[entry]['time'], time)
        else:
            self.data[entry]['values'] = value
            self.data[entry]['time'] = time

        if self.data[entry]['size'] is None:
            self.data['entry']['size'] = value.size1()

    def get_copy(self):
        """
            Return a copy of this dataset. The copy is not connected to the original data set, therefore changes in one
            of the dataset will not affect the other.
        :return:
        """
        dataset_copy = copy.copy(self)
        dataset_copy.data = copy.deepcopy(self.data)

        return dataset_copy

    def extend(self, other_dataset):
        """Extend this DataSet with another DataSet. They don't need to be ordered, after the merging a chronological
        sort of the data is performed.

        :param DataSet other_dataset:
        :return:
        """

        for entry in other_dataset.data:
            if not self.get_entry_size(entry) == other_dataset.get_entry_size(entry):
                raise ValueError('The size of the same entry is different in the two datasets, '
                                 '{}!={}'.format(self.get_entry_size(entry), other_dataset.get_entry_size(entry)))
            if not entry in self.data:
                self.create_entry(entry, size=other_dataset.get_entry_size(entry),
                                  names=other_dataset.get_entry_names(entry))
            self.insert_data(entry, value=other_dataset.data[entry]['values'], time=other_dataset.data[entry]['time'])

        self.sort()

    def sort(self, entries=None):
        """
            Sort the dataset for given 'entries' in an chronological order, this can be used when data is not inserted
            in an ordered fashion.

        :param list entries: list of entries to be sorted, if this parameter is no given all the entries will be sorted.
        """
        if entries is None:
            entries = self.data.keys()

        for entry in entries:
            time = [self.data[entry]['time'][i] for i in range(self.data[entry]['time'].shape[1])]
            values = [self.data[entry]['values'][:, i] for i in range(self.data[entry]['values'].shape[1])]
            time, values = (list(t) for t in zip(*sorted(zip(time, values), key=lambda point: point[0])))
            self.data[entry]['time'] = horzcat(*time)
            self.data[entry]['values'] = horzcat(*values)

    def _plot_entry(self, t_vector, data_vector, row, label='', plot_style='plot'):
        t_vector, data_vector = self._find_discontinuity_for_plotting(t_vector, data_vector)
        print(t_vector)
        if plot_style not in ['plot', 'step']:
            raise ValueError('Plot style not recognized: "{}". Allowed : "plot" and "step"'.format(plot_style))
        if plot_style == 'plot':
            return plt.plot(t_vector.T, data_vector[row, :].T, label=label)
        elif plot_style == 'step':
            return plt.step(t_vector.T, data_vector[row, :].T, label=label, where='post')

    def _find_discontinuity_for_plotting(self, time, values):
        tolerance = 1.2
        if self.max_delta_t is None:
            delta_t = sum2(time) / time.numel()
        else:
            delta_t = self.max_delta_t
        out_time = time[0]
        out_values = values[:, 0]

        for i in range(1, time.shape[1]):
            delta_t_comparison = time[i] - time[i - 1]
            if delta_t_comparison > (1 + tolerance) * delta_t:
                # include NaN
                out_time = horzcat(out_time, DM.nan())
                out_values = horzcat(out_values, DM.nan(values.shape[0], 1))

            out_time = horzcat(out_time, time[i])
            out_values = horzcat(out_values, values[:, i])

            if self.max_delta_t is None:
                delta_t = delta_t_comparison
        return out_time, out_values

    def plot(self, plot_list, figures=None, show=True):
        """Plot DataSet information.
        It takes as input a list of dictionaries, each dictionary represents a plot.  In the dictionary use keyword 'x'
        to specify which states you want to print, the value of the dictionary should be a list of index of the states
        to be printed.
        :param list plot_list: List of dictionaries to generate the plots.
        :param list figures: list of figures to be plotted on top (optional)
        :param bool show: if the plotted figures should be shown after plotting (optional, default=True).
        """

        if isinstance(plot_list, dict):
            plot_list = [plot_list]

        used_figures = []

        for k, entry_dict in enumerate(plot_list):
            if figures is not None:
                fig = plt.figure(figures[k].number)
            else:
                fig = plt.figure(k)

            used_figures.append(fig)

            lines = []
            # Plot optimization x data
            for entry in entry_dict:
                indexes = entry_dict[entry]
                # if it is 'all'
                if indexes == 'all':
                    indexes = range(self.data[entry]['size'])

                # if it is a variable name
                names = self.get_entry_names(entry)
                for i, item in enumerate(indexes):
                    if isinstance(item, (str, unicode)):
                        indexes[i] = names.index(item)

                # plot entry/indexes
                for l in indexes:
                    line = self._plot_entry(self.data[entry]['time'], self.data[entry]['values'], l,
                                            label=self.data[entry]['names'][l], plot_style=self.plot_style)
                    lines.append(line)

            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            plt.legend()

        plt.interactive(True)
        if show:
            plt.show()

        return used_figures
