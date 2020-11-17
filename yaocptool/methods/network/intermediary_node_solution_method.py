from collections import defaultdict

from yaocptool.modelling.ocp import OptimalControlProblem
from casadi.casadi import (
    DM,
    fabs,
    Function,
    horzcat,
    horzsplit,
    mmax,
    MX,
    substitute,
    vec,
    vertcat,
    vertsplit,
)
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.augmented_lagrangian import AugmentedLagrangian
from yaocptool.util.util import create_constant_theta, find_variables_indices_in_vector


class IntermediaryNodeSolutionMethod(AugmentedLagrangian):
    _u_c_function = None

    @property
    def u_c_function(self):
        if self._u_c_function is None:
            self._u_c_function = self._create_solve_function()
        return self._u_c_function

    def _create_solve_function(self):
        mu_k, y_p1, u_p2, nu_p1_c, nu_c_p2 = self._get_symb_variables()
        u_c = (nu_c_p2 - nu_p1_c) / (2 * mu_k) + (y_p1 + u_p2) / 2

        return Function("f_u_c", [mu_k, y_p1, u_p2, nu_p1_c, nu_c_p2], [u_c])

    def _create_nu_update_func(self):
        mu_k, y_p1, u_p2, nu_p1_c, nu_c_p2 = self._get_symb_variables()

        vectorize = lambda var: horzcat(
            *[horzcat(*var[el]) for el in range(self.finite_elements)]
        )

        u_c = vectorize(
            [
                [MX.sym("u_c_{el}_{deg}") for deg in range(self.degree)]
                for el in range(self.finite_elements)
            ]
        )
        y_c = vectorize(
            [
                [MX.sym("y_c_{el}_{deg}") for deg in range(self.degree)]
                for el in range(self.finite_elements)
            ]
        )

        nu_p1_c_next = nu_p1_c + mu_k * (u_c - y_p1)
        nu_c_p2_next = nu_c_p2 + mu_k * (u_p2 - y_c)

        algebraic_error = u_c - y_p1
        equality_error = u_p2 - y_c

        return Function(
            "f_nu_update_and_error",
            [mu_k, y_p1, u_p2, nu_p1_c, nu_c_p2, u_c, y_c],
            [nu_p1_c_next, nu_c_p2_next, algebraic_error, equality_error],
        )

    def _get_symb_variables(self):
        mu_k = MX.sym("mu_k")

        y_p1 = [
            [MX.sym("y_p1_{el}_{deg}") for deg in range(self.degree)]
            for el in range(self.finite_elements)
        ]
        u_p2 = [
            [MX.sym("u_p2_{el}_{deg}") for deg in range(self.degree)]
            for el in range(self.finite_elements)
        ]

        nu_p1_c = [
            [MX.sym("nu_p1_{el}_{deg}") for deg in range(self.degree)]
            for el in range(self.finite_elements)
        ]
        nu_c_p2 = [
            [MX.sym("nu_p2_{el}_{deg}") for deg in range(self.degree)]
            for el in range(self.finite_elements)
        ]

        vectorize = lambda var: horzcat(
            *[horzcat(*var[el]) for el in range(self.finite_elements)]
        )
        y_p1, u_p2, nu_p1_c, nu_c_p2 = (
            vectorize(y_p1),
            vectorize(u_p2),
            vectorize(nu_p1_c),
            vectorize(nu_c_p2),
        )
        return mu_k, y_p1, u_p2, nu_p1_c, nu_c_p2

    def _compute_new_nu_and_error(
        self, p=None, theta=None, raw_solution_dict=None
    ) -> DM:
        if raw_solution_dict is None:
            raw_solution_dict = {}
        if theta is None:
            theta = {}
        if p is None:
            p = []

        if self.new_nu_func is None:
            self.new_nu_func = self._create_nu_update_func()

        errors = DM(0)

        if not self.options.debug_skip_compute_nu_and_error:
            split_vector = lambda ind: horzcat(
                *[
                    vertsplit(theta[el], self.degree)[ind].T
                    for el in range(self.options.finite_elements)
                ]
            )
            y_p1, u_p2, nu_p1_c, nu_c_p2 = (split_vector(i) for i in range(4))
            u_c = raw_solution_dict["x"]
            y_c = u_c

            (nu_p1_c_next, nu_c_p2_next, rel_alg, rel_eq) = self.new_nu_func(
                p[0], y_p1, u_p2, nu_p1_c, nu_c_p2, u_c, y_c
            )

            errors = DM(
                [mmax(fabs(row)) for row in vertsplit(vertcat(rel_alg, rel_eq), 1)]
            )

            nu_tilde = {
                el: vertcat(nu_p1_c_next_el, nu_c_p2_next_el)
                for el, (nu_p1_c_next_el, nu_c_p2_next_el) in enumerate(
                    zip(
                        horzsplit(nu_p1_c_next, self.degree),
                        horzsplit(nu_c_p2_next, self.degree),
                    )
                )
            }

            if not self.options.debug_skip_update_nu:
                if (
                    self.options.only_update_if_improve
                    and any((errors > 0).nz)
                    and self.last_violation_error is not None
                ):
                    for ind, (error, last_error) in enumerate(
                        zip(errors.nz, self.last_violation_error.nz)
                    ):
                        if (error - last_error) < 0:
                            self.last_violation_error[ind] = error
                            for el in self.nu_tilde:
                                self.nu_tilde[el][ind] = nu_tilde[el][ind]
                        else:
                            print("Error is greater than zero")
                else:
                    self.nu_tilde = nu_tilde
        return errors

    def solve(self, theta, p=None, *args, **kwargs):
        print(self.mu)
        theta_k = self.join_nu_to_theta(theta, self.nu)
        p_k = vertcat(p, self.mu) if p else self.mu

        split_vector = lambda ind: horzcat(
            *[
                vertsplit(theta_k[el], self.degree)[ind].T
                for el in range(self.options.finite_elements)
            ]
        )
        y_p1, u_p2, nu_p1_c, nu_c_p2 = (split_vector(i) for i in range(4))

        u_c = self.u_c_function(p_k[0], y_p1, u_p2, nu_p1_c, nu_c_p2)
        raw_solution_dict = {"x": u_c}

        return self.create_optimization_result(raw_solution_dict, p_k, theta_k)

    def create_optimization_result(self, raw_solution_dict, p, theta):
        optimization_result = OptimizationResult()
        optimization_result.method_name = self.__class__.__name__

        # From the solution_method
        for attr in [
            "finite_elements",
            "degree",
            "degree_control",
            "time_breakpoints",
            "discretization_scheme",
        ]:
            attr_value = getattr(self, attr)
            setattr(optimization_result, attr, attr_value)

        optimization_result.x_0 = None
        optimization_result.theta = theta
        optimization_result.p = p
        optimization_result.raw_solution_dict = raw_solution_dict

        # From the problem
        for attr in ["t_0", "t_f"]:
            attr_value = getattr(self.problem, attr)
            setattr(optimization_result, attr, attr_value)
        optimization_result.problem_name = self.problem.name

        # From model
        optimization_result.x_names = self.model.x_names
        optimization_result.y_names = self.model.y_names
        optimization_result.u_names = self.model.u_names
        optimization_result.theta_opt_names = [
            self.problem.theta_opt[i].name() for i in range(self.problem.n_theta_opt)
        ]

        optimization_result.u_data["time"] = self.time_interpolation_controls
        optimization_result.u_data["values"] = [
            horzsplit(element_values, 1)
            for element_values in horzsplit(raw_solution_dict["x"], self.degree)
        ]

        optimization_result.y_data["time"] = self.time_interpolation_controls
        optimization_result.y_data["values"] = [
            horzsplit(element_values, 1)
            for element_values in horzsplit(raw_solution_dict["x"], self.degree)
        ]
        optimization_result.objective_opt_problem = 0.0

        optimization_result.success = True
        optimization_result.stats = {}

        return optimization_result

    def _get_nu_func_symb_variables(self):
        _, y_p1, u_p2, _, _ = self._get_symb_variables()
        return [], y_p1, u_p2, [], []
