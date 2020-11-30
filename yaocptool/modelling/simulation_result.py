from typing import List

from casadi import DM
from yaocptool.modelling import DataSet


class SimulationResult(DataSet):
    def __init__(
        self,
        n_x: int,
        n_y: int,
        n_u: int,
        x_names: List[str] = None,
        y_names: List[str] = None,
        u_names: List[str] = None,
        **kwargs
    ):
        self.model_name = ""
        self.delta_t = 1
        self.t_0 = 0
        self.t_f = 1
        self.finite_elements = 0

        DataSet.__init__(self, **kwargs)
        self.plot_style = "plot"

        self.create_entry("x", size=n_x, names=x_names)
        self.create_entry("y", size=n_y, names=y_names)
        self.create_entry("u", size=n_u, names=u_names)

    @property
    def x(self) -> List[DM]:
        return [
            self.data["x"].values[:, i] for i in range(self.data["x"].values.shape[1])
        ]

    @property
    def y(self) -> List[DM]:
        return [
            self.data["y"].values[:, i] for i in range(self.data["y"].values.shape[1])
        ]

    @property
    def u(self) -> List[DM]:
        return [
            self.data["u"].values[:, i] for i in range(self.data["u"].values.shape[1])
        ]

    @property
    def t(self) -> DM:
        return self.data["x"].time

    @property
    def n_x(self) -> int:
        return self.data["x"].size

    @property
    def n_y(self) -> int:
        return self.data["y"].size

    @property
    def n_u(self) -> int:
        return self.data["u"].size

    @property
    def x_names(self) -> List[str]:
        return self.data["x"].names

    @property
    def y_names(self) -> List[str]:
        return self.data["y"].names

    @property
    def u_names(self) -> List[str]:
        return self.data["u"].names

    def final_condition(self):
        """Return the simulation final condition in the form of a tuple (x_f, y_f, u_f)

        :rtype: DM, DM, DM
        """
        return {
            "x": self.data["x"].values[:, -1],
            "y": self.data["y"].values[:, -1],
            "u": self.data["u"].values[:, -1],
        }

    def extend(self, other_sim_result: "SimulationResult"):
        """Extend this SimulationResult with other SimulationResult.
        It is only implemented for the case where the other simulation result starts at the end of this
        simulation. That is this.t_f == other_sim_result.t_0 .

        :param SimulationResult other_sim_result:
        :return:
        """
        list_of_attributes_to_check = ["n_x", "n_y", "n_u"]
        for attr in list_of_attributes_to_check:
            if getattr(self, attr) != getattr(other_sim_result, attr):
                raise Exception(
                    "Attribute {} is no equal for both SimulationResults: {}!={}".format(
                        attr, getattr(self, attr), getattr(other_sim_result, attr)
                    )
                )

        self.t_f = max(self.t_f, other_sim_result.t_f)
        self.t_0 = min(self.t_0, other_sim_result.t_0)
        self.finite_elements += other_sim_result.finite_elements

        DataSet.extend(self, other_sim_result)
