# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 10:50:48 2016

@author: marco
"""
import collections
import copy
from contextlib import suppress
from typing import Any, Dict, List, Optional, Tuple, Union
from yaocptool.util.util import create_constant_theta

from casadi import (
    DM,
    MX,
    SX,
    Function,
    horzcat,
    jacobian,
    mtimes,
    rootfinder,
    substitute,
    vertcat,
)

from yaocptool import (
    config,
    find_variables_in_vector_by_name,
    find_variables_indices_in_vector,
)
from yaocptool.modelling import DAESystem, SimulationResult
from yaocptool.modelling.mixins import (
    AlgebraicMixin,
    ContinuousStateMixin,
    ControlMixin,
    ParameterMixin,
)


class SystemModel(ContinuousStateMixin, AlgebraicMixin, ControlMixin, ParameterMixin):
    t = SX.sym("t")
    tau = SX.sym("tau")

    def __init__(self, name: str = "model", model_name_as_prefix: bool = False):
        r"""
            Continuous-time Dynamic System Model

        .. math::
            \dot{x} = f(x,y,u,t,p,\\theta) \\\\
            g(x,y,u,t,p,\\theta) = 0\\\\

        x - states
        y - algebraic
        u - control
        p - constant parameters
        theta - parameters dependent of the finite_elements (e.g.: disturbances)

        Note: when vectorizing the parameters order is [ p; theta; u_par]

        :param name: model name
        :param bool model_name_as_prefix: if true all variables create will have the model name as prefix
            e.g.: 'tank_h', where 'tank' is model name and 'h' is the state created
        """
        super().__init__()
        self.name = name

        self.model_name_as_prefix = model_name_as_prefix
        self.has_adjoint_variables = False

        self.verbosity = 0

    @property
    def system_type(self) -> str:
        """
            Return the system type.

        :return: 'ode'  if an ODE system and 'dae' if an DAE system
        :rtype: str
        """
        with suppress(AttributeError):
            if self.n_y > 0:
                return "dae"
        return "ode"

    @property
    def x_sys_sym(self) -> SX:
        if self.has_adjoint_variables:
            return self.x[: int(self.n_x // 2)]
        else:
            return self.x

    @property
    def lamb_sym(self) -> SX:
        if self.has_adjoint_variables:
            return self.x[self.n_x // 2 :]
        else:
            return SX([])

    @property
    def all_sym(self) -> List[SX]:
        return [
            self.t,
            self.x,
            self.y,
            self.p,
            self.theta,
            self.u_par,
        ]

    @property
    def t_sym(self) -> SX:
        return self.t

    @t_sym.setter
    def t_sym(self, value: SX):
        self.t = value

    @property
    def tau_sym(self) -> SX:
        return self.tau

    @tau_sym.setter
    def tau_sym(self, value: SX):
        self.tau = value

    def __str__(self) -> str:
        return f'{self.__class__.__name__}("{self.name}")'

    def name_variable(self, name: str) -> str:
        if self.model_name_as_prefix:
            return f"{self.name}_{name}"
        return name

    def print_summary(self):
        """
        Print model summary when using print(model)

        :return:
        """
        s = ""
        s += "=" * 20 + "\n"
        s += "Model Name: {:>23}".format(self.name)
        s += "| System type:                            {:>3}".format(self.system_type)
        s += "\n"
        s += "-" * 20 + "\n"
        s += "Number of states (x):         {:4} | Number of algebraic (y):               {:4}".format(
            self.n_x, self.n_y
        )
        s += "\n"
        s += "Number of controls (u):       {:4} |".format(self.n_u)
        s += "\n"
        s += "Number of parameters (p):     {:4} | Number of finite elem. param. (theta): {:4}".format(
            self.n_p, self.n_theta
        )
        s += "\n"
        s += "-" * 20 + "\n"
        s += "Number of ODE:                {:4} | Number of algebraic eq.:               {:4}".format(
            self.ode.numel(), self.alg.numel()
        )
        s += "\n"
        s += "=" * 20 + "\n"

        print(s)

    def print_variables(self):
        """
        Print list of variable in the model (x, y, u, p, theta)
        """
        var_name_space = 20
        column_size = var_name_space + 4

        n_lines = max(self.n_x, self.n_y, self.n_u, self.n_p, self.n_theta)
        x_names = self.x_names + [""] * (n_lines - self.n_x)
        y_names = self.y_names + [""] * (n_lines - self.n_y)
        u_names = self.u_names + [""] * (n_lines - self.n_u)
        p_names = self.p_names + [""] * (n_lines - self.n_p)
        theta_names = self.theta_names + [""] * (n_lines - self.n_theta)

        header_separator = "=|="
        header = ""
        header += " states (x) ".center(column_size, "=") + header_separator
        header += " algebraic (y) ".center(column_size, "=") + header_separator
        header += " input (u) ".center(column_size, "=") + header_separator
        header += " parameter (p) ".center(column_size, "=") + header_separator
        header += " theta param (theta) ".center(column_size, "=") + header_separator
        print(header)

        for i in range(n_lines):
            line = ""
            if i < self.n_x:
                line += "{:>2}: {:<" + str(var_name_space) + "}" + " | "
                line = line.format(i, x_names[i])
            else:
                line += " " * (var_name_space + 4) + " | "

            if i < self.n_y:
                line += "{:>2}: {:<" + str(var_name_space) + "}" + " | "
                line = line.format(i, y_names[i])
            else:
                line += " " * (var_name_space + 4) + " | "

            if i < self.n_u:
                line += "{:>2}: {:<" + str(var_name_space) + "}" + " | "
                line = line.format(i, u_names[i])
            else:
                line += " " * (var_name_space + 4) + " | "

            if i < self.n_p:
                line += "{:>2}: {:<" + str(var_name_space) + "}" + " | "
                line = line.format(i, p_names[i])
            else:
                line += " " * (var_name_space + 4) + " | "

            if i < self.n_theta:
                line += "{:>2}: {:<" + str(var_name_space) + "}" + " | "
                line = line.format(i, theta_names[i])
            else:
                line += " " * (var_name_space + 4) + " | "

            print(line)
        print(("=" * column_size + header_separator) * 5)

    def include_equations(self, *args: SX, **kwargs: Union[SX, List[SX]]):
        """
            Include model equations, (ordinary) differential equation and algebraic equation (ode and alg)

        :param list|casadi.SX ode: (ordinary) differential equation
        :param list|casadi.SX alg: algebraic equation
        """
        super().include_equations(*args, **kwargs)

    def include_variables(
        self,
        x: Optional[Union[SX, List[SX]]] = None,
        y: Optional[Union[SX, List[SX]]] = None,
        u: Optional[Union[SX, List[SX]]] = None,
        p: Optional[Union[SX, List[SX]]] = None,
        theta: Optional[Union[SX, List[SX]]] = None,
    ):
        """
            Include variables (x, y, u, p, and theta) to the model.

        :param SX|list|None x: state variable
        :param SX|list|None y: algebraic variable
        :param SX|list|None u: control/input variable
        :param SX|list|None p: parameter variable
        :param SX|list|None theta: element dependent variable
        """
        if x is not None:
            if isinstance(x, list):
                x = vertcat(*x)
            self.include_state(x)

        if y is not None:
            if isinstance(y, list):
                y = vertcat(*y)
            self.include_algebraic(y)

        if u is not None:
            if isinstance(u, list):
                u = vertcat(*u)
            self.include_control(u)

        if p is not None:
            if isinstance(p, list):
                p = vertcat(*p)
            self.include_parameter(p)

        if theta is not None:
            if isinstance(theta, list):
                theta = vertcat(*theta)
            self.include_theta(theta)

    def replace_variable(self, original: SX, replacement: SX):
        """
        Replace a variable or parameter by an variable or expression.

        :param SX|list replacement:
        :param SX|list original: and replacement, and also variable type which
            describes which type of variable is being remove to it from the
            counters. Types: 'x', 'y', 'u', 'p', 'ignore'
        """
        super().replace_variable(original, replacement)

    def get_variable_by_name(
        self, name: str = "", var_type: Optional[str] = None
    ) -> SX:
        """
            Return a variable with a specified name (Regex accepted).

        If no or multiple variables are found with the specified 'name' an ValueError exception is raised.
        To specify the search in a single variable type (x/y/u/p/theta) use the 'var_type'

        :param str name: variable name/regex
        :param str var_type: variable type (optional)
        :return: variable
        :rtype: SX
        """
        result = self.get_variables_by_names(name, var_type)
        # if only one is found return it
        if len(result) == 1:
            return result[-1]
        # if multiple where found raise exception
        if len(result) > 1:
            raise ValueError(
                "Multiple variables where found with the name: {}. Found: {}".format(
                    name, result
                )
            )
        # if none was found raise exception
        raise ValueError("No variable was found with name: {}".format(name))

    def get_variables_by_names(
        self,
        names: Union[str, List[str]] = "",
        var_type: Optional[Union[str, List[str]]] = None,
    ) -> List[SX]:
        """

        :param str|list of str names: list of variables names
        :param str var_type:
        """
        # if only one name is passed
        if not isinstance(names, list):
            names = [names]
        # if no specific type is passed, look into all
        if var_type is None:
            var_type = ["x", "y", "u", "p", "theta"]
        # make the passed one into a list so we can iterate over
        if not isinstance(var_type, list):
            var_type = [var_type]

        result = []
        for var_t in var_type:
            var = getattr(self, var_t)
            result.extend(find_variables_in_vector_by_name(names, var))

        return result

    def has_variable(self, var: SX) -> bool:
        """

        :param casadi.SX var: variable to be checked if it is in the SystemModel
        """

        ind = find_variables_indices_in_vector(
            var,
            vertcat(
                self.x,
                self.y,
                self.u,
                self.p,
                self.theta,
                self.u_par,
            ),
        )
        return len(ind) > 0

    def is_parametrized(self) -> bool:
        """
            Check if the model is parametrized.

        :rtype bool:
        """
        # if no u is provided (checking if the model is parametrized)
        return len(self._parametrized_controls) > 0

    def include_models(self, models: Union["SystemModel", List["SystemModel"]]):
        """
            Include model or list of models into this model. All the variables and functions will be included.

        :param models: models to be included
        """
        if not isinstance(models, list):
            models = [models]

        for model in models:
            # include variables
            self.include_state(model.x, x_0=model.x_0)
            self.include_algebraic(model.y)
            self.include_control(model.u)
            self.include_parameter(model.p)
            self.include_theta(model.theta)

            # include equations
            self.include_equations(ode=model.ode, x=model.x, alg=model.alg)

            # replace model time variables with this model time variables
            self.replace_variable(model.t, self.t)
            self.replace_variable(model.tau, self.tau)

    def merge(
        self,
        models_list: Union["SystemModel", List["SystemModel"]],
        connecting_equations: Optional[SX] = None,
    ):
        if not isinstance(models_list, list):
            models_list = [models_list]

        self.include_models(models_list)

        if connecting_equations is not None:
            self.include_equations(alg=connecting_equations)

    def connect(self, u: SX, y: SX, replace: bool = False):
        """
        Connect an input 'u' to a algebraic variable 'y', u = y.
        The function will perform the following actions:
        - include an algebraic equation u - y = 0
        - remove 'u' from the input vector (since it is not a free variable anymore)
        - include 'u' into the algebraic vector, since it is an algebraic variable now.

        :param u: input variable
        :param y: algebraic variable
        """
        # check if same size
        assert (
            u.numel() != y.numel()
        ), f'Size of "u" and "y" "are not the same, u={u.numel()} and y={y.numel()}'

        if replace:
            self.replace_variable(u, y)
        else:
            self.include_equations(alg=[u - y])
            self.include_algebraic(u)

        self.remove_control(u)

    @staticmethod
    def put_values_in_all_sym_format(
        t: Any = None,
        x: Any = None,
        y: Any = None,
        p: Any = None,
        theta: Any = None,
        u_par: Any = None,
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        if t is None:
            t = DM(0, 1)
        if x is None:
            x = DM(0, 1)
        if y is None:
            y = DM(0, 1)
        if p is None:
            p = DM(0, 1)
        if theta is None:
            theta = DM(0, 1)
        if u_par is None:
            u_par = DM(0, 1)
        return t, x, y, p, theta, u_par

    @staticmethod
    def all_sym_names() -> Tuple[str, str, str, str, str, str]:
        return "t", "x", "y", "p", "theta", "u_par"

    def convert_expr_from_time_to_tau(self, expr: SX, t_k: float, t_kp1: float) -> SX:
        t = self.t
        tau = self.tau
        h = t_kp1 - t_k
        return substitute(expr, t, tau * h + t_k)

    def convert_expr_from_tau_to_time(self, expr: SX, t_k: float, t_kp1: float) -> SX:
        t = self.t
        tau = self.tau
        h = t_kp1 - t_k
        return substitute(expr, tau, (t - t_k) / h)

    def get_copy(self) -> "SystemModel":
        """
            Get a copy of this model.

        Uses copy.copy to get copy of this object.

        :rtype: SystemModel
        """
        copy_model = copy.copy(self)
        copy_model._ode = dict(copy_model._ode)
        return copy_model

    def get_deepcopy(self) -> "SystemModel":
        """
            Get a deep copy of this model, differently from "get_copy",
            the variables of the original copy and the
            hard copy will not be the same, i.e. model.x != copy.x

        :rtype: SystemModel
        """
        model_copy = SystemModel(
            name=self.name, model_name_as_prefix=self.model_name_as_prefix
        )
        x_copy = SX(0, 1)
        y_copy = SX(0, 1)
        u_copy = SX(0, 1)
        p_copy = SX(0, 1)
        theta_copy = SX(0, 1)
        u_par_copy = SX(0, 1)

        if self.n_x > 0:
            x_copy = vertcat(
                *[model_copy.create_state(self.x[i].name()) for i in range(self.n_x)]
            )
        if self.n_y > 0:
            y_copy = vertcat(
                *[
                    model_copy.create_algebraic_variable(self.y[i].name())
                    for i in range(self.n_y)
                ]
            )
        if self.n_u > 0:
            u_copy = vertcat(
                *[model_copy.create_control(self.u[i].name()) for i in range(self.n_u)]
            )

        if self.n_p > 0:
            p_copy = vertcat(
                *[
                    model_copy.create_parameter(self.p[i].name())
                    for i in range(self.n_p)
                ]
            )
        if self.n_theta > 0:
            theta_copy = vertcat(
                *[
                    model_copy.create_theta(self.theta[i].name())
                    for i in range(self.n_theta)
                ]
            )

        if self.n_u_par > 0:
            u_par_copy = vertcat(
                *[SX.sym(self.u_par[i].name()) for i in range(self.n_u_par)]
            )

        model_copy.include_equations(ode=self.ode, alg=self.alg)
        model_copy.u_par = self.u_par
        model_copy.u_expr = self.u_expr

        model_copy.replace_variable(self.x, x_copy)
        model_copy.replace_variable(self.y, y_copy)
        model_copy.replace_variable(self.u, u_copy)
        model_copy.replace_variable(self.p, p_copy)
        model_copy.replace_variable(self.theta, theta_copy)
        model_copy.replace_variable(self.u_par, u_par_copy)

        model_copy.has_adjoint_variables = self.has_adjoint_variables

        return model_copy

    def get_dae_system(self) -> DAESystem:
        """
        Return a DAESystem object with the model equations.
        """
        if self.system_type == "ode":
            kwargs = {
                "x": self.x,
                "ode": self.ode,
                "t": self.t,
                "tau": self.tau,
            }
        else:
            kwargs = {
                "x": self.x,
                "y": self.y,
                "ode": self.ode,
                "alg": self.alg,
                "t": self.t,
                "tau": self.tau,
            }
        if self.n_p + self.n_theta + self.u_par.numel() > 0:
            kwargs["p"] = vertcat(self.p, self.theta, self.u_par)

        return DAESystem(**kwargs)

    def simulate(
        self,
        x_0: Union[DM, List[float]],
        t_f: Union[float, List[float]],
        t_0: float = 0.0,
        u: Optional[Union[MX, DM, List[float]]] = None,
        p: Optional[Union[MX, DM, List[float]]] = None,
        theta: Optional[Dict[int, DM]] = None,
        y_0: Optional[Union[MX, DM, List[float]]] = None,
        integrator_type: Optional[str] = None,
        integrator_options: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """Simulate model.
            If t_f is a float, then only one simulation will be done. If t_f is a list of times, then a sequence of
            simulations will be done, that each t_f is the end of a finite element.

        :param list||DM x_0: Initial condition
        :param float||list t_f: Final time of the simulation, can be a list of final times for sequential simulation
        :param float t_0: Initial time
        :param list||DM u: Controls of the system to be simulated
        :param DM||SX||list p: Simulation parameters
        :param dict theta: Parameters theta, which varies for each simulation for sequential simulations.
                           If t_f is a list then theta has to have one entry for each k in [0,...,len(t_f)]
        :param y_0: Initial guess for the algebraic variables
        :param str integrator_type: 'implicit' or 'explicit'
        :param dict integrator_options: options to be passed to the integrator

        :rtype: SimulationResult
        """
        if integrator_type is None:
            integrator_type = "implicit"
        if integrator_options is None:
            integrator_options = {}

        if isinstance(x_0, list):
            x_0 = vertcat(*x_0)
        if isinstance(y_0, list):
            y_0 = vertcat(*y_0)
        if isinstance(u, list):
            u = vertcat(*u)
        if isinstance(p, list):
            p = vertcat(*p)
        if not isinstance(t_f, list):
            t_f = [t_f]

        if theta is None:
            theta = {k: DM(0, 1) for k in range(len(t_f))}
        if p is None:
            p = DM(0, 1)
        if u is None:  # if control is not given
            u = DM(0, 1)

        u_sim: Dict[int, Union[MX, DM]] = (
            u if isinstance(u, dict) else {el: u for el in range(len(t_f))}
        )
        if len(t_f) > 1 and len(u_sim) != len(t_f):
            raise ValueError(
                'If "t_f" is a list, the parameter "u" should be a list with same length of "t_f".'
                "len(t_f) = {}".format(len(t_f))
            )

        dae_sys = self.get_dae_system()

        t_x_list = [t_0]
        t_yu_list = []
        x_list = [x_0]
        y_list = []
        u_list = []
        t_k = t_0
        x_k = x_0
        y_k = y_0
        f_u = Function("f_u", self.all_sym, [self.u_expr])

        for k, t_kpp in enumerate(t_f):
            p_k = vertcat(p, theta[k], u_sim[k])
            result = dae_sys.simulate(
                x_0=x_k,
                t_f=t_kpp,
                t_0=t_k,
                p=p_k,
                y_0=y_k,
                integrator_type=integrator_type,
                integrator_options=integrator_options,
            )
            t_x_list.append(t_kpp)
            t_yu_list.append(t_kpp)

            t_k = t_kpp
            x_k = result["xf"]
            y_k = result["zf"]
            u_k = f_u(
                *self.put_values_in_all_sym_format(
                    t=t_kpp, x=x_k, y=y_k, p=p, theta=theta[k], u_par=u[k]
                )
            )

            x_list.append(result["xf"])
            y_list.append(result["zf"])
            u_list.append(u_k)

        t_x_list = horzcat(*t_x_list)
        t_yu_list = horzcat(*t_yu_list)

        simulation_result = SimulationResult(
            model_name=self.name,
            n_x=self.n_x,
            n_y=self.n_y,
            n_u=self.n_u,
            x_names=self.x_names,
            y_names=self.y_names,
            u_names=self.u_names,
            t_0=t_0,
            t_f=t_f[-1],
            finite_elements=len(t_f),
        )

        simulation_result.insert_data("x", time=t_x_list, value=horzcat(*x_list))
        simulation_result.insert_data("y", time=t_yu_list, value=horzcat(*y_list))
        simulation_result.insert_data("u", time=t_yu_list, value=horzcat(*u_list))

        return simulation_result

    def linearize(self, x_bar, u_bar):
        """
        Returns a linearized model at a given points (X_BAR, U_BAR)
        """
        a_matrix = Function("a_matrix", [self.x, self.u], [jacobian(self.ode, self.x)])(
            x_bar, u_bar
        )
        b_matrix = Function("b_matrix", [self.x, self.u], [jacobian(self.ode, self.u)])(
            x_bar, u_bar
        )

        linear_model = SystemModel(n_x=self.n_x, n_u=self.n_u)
        linear_model.include_equations(
            ode=[mtimes(a_matrix, linear_model.x) + mtimes(b_matrix, linear_model.u)]
        )
        linear_model.name = "linearized_" + self.name

        return linear_model

    def find_equilibrium(
        self, additional_eqs, guess=None, t_0=0.0, rootfinder_options=None
    ):
        """Find a equilibrium point for the model.
        This method solves the root finding problem:

            f(x,y,u,t_0) = 0
            g(x,y,u,t_0) = 0
            additional_eqs (x,y,u,t_0) = 0

        Use additional_eqs to specify the additional conditions remembering that dim(additional_eqs) = n_u,
        so the system can be well defined.
        If no initial guess is provided ("guess" parameter) a guess of ones will be used (not zero to avoid problems
        with singularities.

        Returns x_0, y_0, u_0

        :param dict rootfinder_options: options to be passed to rootfinder
        :param additional_eqs: SX
        :param guess: DM
        :param t_0: float
        :return: (DM, DM, DM)
        """
        if rootfinder_options is None:
            rootfinder_options = dict(
                nlpsol="ipopt", nlpsol_options=config.SOLVER_OPTIONS["nlpsol_options"]
            )
        if guess is None:
            guess = [1] * (self.n_x + self.n_y + self.n_u)
        if isinstance(additional_eqs, list):
            additional_eqs = vertcat(*additional_eqs)

        eqs = vertcat(self.ode, self.alg, additional_eqs)
        eqs = substitute(eqs, self.t, t_0)
        eqs = substitute(eqs, self.tau, 0)
        f_eqs = Function("f_equilibrium", [vertcat(*self.all_sym[1:-1])], [eqs])

        rf = rootfinder("rf_equilibrium", "nlpsol", f_eqs, rootfinder_options)
        res = rf(guess)
        return (
            res[: self.n_x],
            res[self.n_x : self.n_x + self.n_y],
            res[self.n_x + self.n_y :],
        )
