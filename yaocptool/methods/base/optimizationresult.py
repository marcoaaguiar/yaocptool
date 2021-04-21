from collections import defaultdict
from contextlib import suppress
from functools import partial
from typing import Any, Dict, List, Optional, Union

from casadi import DM, horzcat

from yaocptool.modelling import DataSet
from yaocptool.optimization.abstract_optimization_problem import (
    ExtendedOptiResultDictType,
)


class OptimizationResult:
    def __init__(self, **kwargs):
        # Raw Information
        self.raw_solution_dict: ExtendedOptiResultDictType = {}  # type: ignore
        self.raw_decision_variables: Optional[List] = None

        # Data from the method
        self.method_name: str = ""
        self.discretization_scheme: str = ""
        self.finite_elements: Optional[int] = None
        self.degree: Optional[int] = None
        self.degree_control: Optional[int] = None

        # From problem
        self.problem_name = ""
        self.t_0: Optional[float] = None
        self.t_f: Optional[float] = None

        # From solver
        self.stats = None
        self.success = None

        self.time_breakpoints = []
        self.collocation_points = []

        self.objective_opt_problem: DM = None  # type: ignore
        self.v_final = None  # type: DM|None
        self.x_c_final = None  # type: DM|None
        self.constraints_values = None  # type: DM|None

        self.x_names = []
        self.y_names = []
        self.u_names = []
        self.theta_opt_names = []

        self.x_data = {"values": [], "time": []}
        self.y_data = {"values": [], "time": []}
        self.u_data = {"values": [], "time": []}

        self.other_data = defaultdict(partial(dict, [("values", []), ("time", [])]))
        self.statistics: Dict[str, Any] = {}

        self.x_0 = DM()
        self.theta = {}
        self.p = DM()
        self.p_opt = DM()
        self.theta_opt = []
        self.eta = DM()

        self._dataset = None

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    @property
    def dataset(self) -> DataSet:
        if self._dataset is None:
            self._dataset = self.to_dataset()
        return self._dataset

    @property
    def is_valid(self) -> bool:
        for attr in ["finite_elements", "degree", "degree_control", "t_0", "t_f"]:
            if getattr(self, attr) < 0:
                raise Exception(
                    "{} attribute {} is lower than 0".format(
                        self.__class__.__name__, attr
                    )
                )
        if len(self.time_breakpoints) < 1:
            raise Exception(
                "{} attribute {} (list) is empty".format(
                    self.__class__.__name__, "time_breakpoints"
                )
            )
        for attr in ["objective"]:
            if getattr(self, attr) is None:
                raise Exception(
                    "{} attribute {} is None".format(self.__class__.__name__, attr)
                )
        return True

    def first_control(self) -> DM:
        """Return the first element of the control vector

        :rtype: DM
        """
        return self.u_data["values"][0][0]

    def get_variable(
        self, var_type: str, indices: Union[int, List[int]]
    ) -> List[List[DM]]:
        """Get all the data for a variable (var_type

        :param str var_type: variable type ('x', 'y', 'u'
        :param int|list of int indices: variable indices
        """

        if var_type == "x":
            data = self.x_data["values"]
        elif var_type == "y":
            data = self.y_data["values"]
        elif var_type == "u":
            data = self.u_data["values"]
        else:
            raise NotImplementedError

        return [[vect[indices] for vect in list_vec] for list_vec in data]

    def get_variable_by_name(self, name: str) -> List[List[DM]]:
        """
        Get all the data for a variable (var_type

        """
        variable = None
        with suppress(ValueError):
            variable = self.get_variable("x", self.x_names.index(name))
        with suppress(ValueError):
            variable = self.get_variable("y", self.y_names.index(name))
        with suppress(ValueError):
            variable = self.get_variable("u", self.u_names.index(name))

        if variable is not None:
            return variable

        raise ValueError(f'Varialble with name "{name}" not found.')

    def to_dataset(self) -> DataSet:
        """
            Return a dataset with the data of x, y, and u

        :rtype: DataSet
        """
        dataset = DataSet(
            name=self.problem_name + "_dataset",
        )

        dataset.plot_style = "plot"

        dataset.create_entry("x", size=len(self.x_names), names=self.x_names)
        dataset.create_entry("y", size=len(self.y_names), names=self.y_names)
        dataset.create_entry("u", size=len(self.u_names), names=self.u_names)
        dataset.create_entry(
            "theta_opt",
            size=len(self.theta_opt_names),
            names=self.theta_opt_names,
            plot_style="step",
        )

        for entry in self.other_data:
            size = self.other_data[entry]["values"][0][0].shape[0]
            dataset.create_entry(
                entry, size=size, names=[entry + "_" + str(i) for i in range(size)]
            )

        dataset.data["u"].plot_style = "plot" if self.degree_control > 1 else "step"

        x_times = self.x_data["time"] + [[self.t_f]]
        if len(self.x_names) > 0:
            for el in range(self.finite_elements + 1):
                time = horzcat(*x_times[el])
                values = horzcat(*self.x_data["values"][el])
                dataset.insert_data("x", time, values)

        for el in range(self.finite_elements):
            if len(self.y_names) > 0:
                time_y = horzcat(*self.y_data["time"][el])
                values_y = horzcat(*self.y_data["values"][el])
                dataset.insert_data("y", time_y, values_y)

            if len(self.u_names) > 0:
                time_u = horzcat(*self.u_data["time"][el])
                values_u = horzcat(*self.u_data["values"][el])
                dataset.insert_data("u", time_u, values_u)

        if len(self.theta_opt_names) > 0:
            dataset.insert_data(
                "theta_opt",
                time=horzcat(*self.time_breakpoints[:-1]),
                value=horzcat(*self.theta_opt),
            )

        for entry in self.other_data:
            for el in range(self.finite_elements):
                dataset.insert_data(
                    entry,
                    time=horzcat(*self.other_data[entry]["time"][el]),
                    value=horzcat(*self.other_data[entry]["values"][el]),
                )

        return dataset

    @property
    def plot(self):
        return self.dataset.plot
