from __future__ import print_function

import copy
import os
import pickle
import re
from collections import defaultdict
from functools import partial

from casadi import horzcat, DM, sum2

from yaocptool.config import PLOT_INTERACTIVE

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
        :param bool find_discontinuity: Default: True. If True, it will try to find discontinuity on the data, and plot
            with gaps where data is missing/not available, instead of a line connecting all data points.
        :param float max_sampling_time: maximum expected distance between two time data. This is used to detect
            discontinuity on the data, and plot it separately.
        """
        self.name = name
        self.data = defaultdict(partial(dict, [('time', DM([])), ('names', None), ('values', DM([])), ('size', None)]))

        self.plot_style = 'step'
        self.max_delta_t = None
        self.find_discontinuity = True

        for (k, val) in kwargs.items():
            setattr(self, k, val)

    def create_entry(self, entry, size, names=None, plot_style=None):
        """
            Create an entry in the dataset

        :param entry: entry name
        :param size: number of rows in the vector
        :param list names: name for each row, it should be a list with size 'size'. If 'names' is not given, then the
            name list [entry_1, entry_2, ..., entry_size]
        :param str plot_style: ('plot' | 'step') choose if the plot will be piecewise constant (step) or a first order
            interpolation (plot).
        """
        if names is None:
            names = [entry + '_' + str(i) for i in range(size)]
        self.data[entry]['size'] = size
        self.data[entry]['names'] = names
        if plot_style is not None:
            self.data[entry]['plot_style'] = plot_style

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

    def insert_data(self, entry, time, value):
        """Insert data on the dataset

        :param str entry: entry name ('x', 'y', 'u', ...)
        :param float|list|DM time: time or time vector of the data
        :param DM value: vector or matrix of values for the entry (column represent time)
        """
        if isinstance(time, list):
            time = horzcat(*time)
        if isinstance(time, (float, int)):
            time = DM(time)

        if not time.shape[1] == value.shape[1]:
            raise ValueError('Number of columns of "time" and "value" should be the same, '
                             'time.shape={} and value.shape={}'.format(time.shape[1], value.shape))

        self.data[entry]['values'] = horzcat(self.data[entry]['values'], value)
        self.data[entry]['time'] = horzcat(self.data[entry]['time'], time)

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
            if entry not in self.data:
                self.create_entry(entry, size=other_dataset.get_entry_size(entry),
                                  names=other_dataset.get_entry_names(entry))
            self.insert_data(entry, time=other_dataset.data[entry]['time'], value=other_dataset.data[entry]['values'])

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
            entry_length = self.data[entry]['time'].shape[1]
            time = [self.data[entry]['time'][i] for i in range(entry_length)]
            values = [self.data[entry]['values'][:, i] for i in range(self.data[entry]['values'].shape[1])]
            time, values = (list(t) for t in zip(*sorted(zip(time, values), key=lambda point: point[0])))
            self.data[entry]['time'] = horzcat(*time)
            self.data[entry]['values'] = horzcat(*values)

    def save(self, file_path):
        """
            Save this object in the "file_path" using pickle (.p extension).
            It can be retrieved using using pickle.load

        :param str file_path: path with file name of the file to be saved. Example: files/result.p
        """
        directory = os.path.abspath(os.path.dirname(file_path))
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb+') as f:
            pickle.dump(self, f)

    def _plot_entry(self, t_vector, data_vector, row, label='', plot_style='plot'):
        if self.find_discontinuity:
            t_vector, data_vector = self._find_discontinuity_for_plotting(t_vector, data_vector)

        if plot_style not in ['plot', 'step']:
            raise ValueError('Plot style not recognized: "{}". Allowed : "plot" and "step"'.format(plot_style))

        if plot_style == 'plot':
            return plt.plot(t_vector.T, data_vector[row, :].T, label=label)
        if plot_style == 'step':
            return plt.step(t_vector.T, data_vector[row, :].T, label=label, where='post')

        raise ValueError('"plot_style" not recognized. Given {}, available: "step" or "plot"'.format(plot_style))

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

    def plot(self, plot_list, figures=None, show=True, exact=False):
        """Plot DataSet information.
        It takes as input a list of dictionaries, each dictionary represents a plot. In the dictionary use keyword 'x',
        'u', 'y', etc., to specify which entry you want to print, the value of the dictionary should be a list of
        index, names, or regex expressions of the entries to be printed. Alternatively, instead of a list a string
        'all' can be passed so all the elements of the entry are plotted.

        Usage:
            result.plot({'x':'all', 'y':'all', 'u':'all'})  # print all variables in a single plot
            result.plot([{'x':[0,3]}, {'y':'all'}, {'u':['v_1', 'q_out']}])  # create 3 plots, with the selected vars
            result.plot('all')  # create 3 plots one with all 'x', one with all 'y', and one with all 'u'

        :param list|str plot_list: List of dictionaries to generate the plots.
        :param list figures: list of figures to be plotted on top (optional)
        :param bool show: if the plotted figures should be shown after plotting (optional, default=True).
        :param bool exact: If true only precise match of entry elements will be plotted, otherwise a regex match will
            be used
        """

        if plot_list == 'all':
            plot_list = [{'x': 'all'}, {'y': 'all'}, {'u': 'all'}]

        if isinstance(plot_list, dict):
            plot_list = [plot_list]

        used_figures = []

        # for each plot asked
        for k, entry_dict in enumerate(plot_list):
            if figures is not None:
                fig = plt.figure(figures[k].number)
            else:
                fig = plt.figure(k)
            used_figures.append(fig)

            lines = []
            # for each entry (dict key) e.g. 'x', 'y'
            for entry in entry_dict:
                if entry not in self.data:
                    raise Exception(f"Entry '{entry}' not found in the dataset. Entries: {self.data.keys()}")

                # if a custom plot style was asked, otherwise use default
                if 'plot_style' in self.data[entry]:
                    plot_style = self.data[entry]['plot_style']
                else:
                    plot_style = self.plot_style

                indexes_or_names = entry_dict[entry]
                # if it is 'all'
                if indexes_or_names == 'all':
                    indexes_or_names = range(self.data[entry]['size'])

                var_names = self.get_entry_names(entry)
                # identify variables with regex
                if not exact:
                    for regex in indexes_or_names[:]:
                        regex_ind = indexes_or_names.index(regex)
                        if isinstance(regex, ("".__class__, u"".__class__)):
                            indexes_or_names[regex_ind:regex_ind + 1] = [v_name for v_name in var_names if
                                                                         re.match(regex, v_name)]

                # if it is a variable name
                for i, item in enumerate(indexes_or_names):
                    if isinstance(item, ("".__class__, u"".__class__)):
                        indexes_or_names[i] = var_names.index(item)

                # plot entry/indexes
                for ind in indexes_or_names:
                    line = self._plot_entry(self.data[entry]['time'], self.data[entry]['values'], ind,
                                            label=self.data[entry]['names'][ind], plot_style=plot_style)
                    lines.append(line)

            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            plt.legend()

        plt.interactive(PLOT_INTERACTIVE)
        if show:
            plt.show()

        return used_figures
