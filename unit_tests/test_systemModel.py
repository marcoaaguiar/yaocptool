from unittest import TestCase

from casadi import vertcat, SX, is_equal
from typing import List, Dict
from .models import create_siso, create_2x1_mimo, create_2x2_mimo
from yaocptool.modelling import SystemModel, OptimalControlProblem


class TestSystemModel(TestCase):
    def setUp(self):
        self.models = {}  # type: Dict[str, SystemModel]
        self.problems = {}  # type: Dict[str, OptimalControlProblem]
        for creator in [create_siso, create_2x1_mimo, create_2x2_mimo]:
            model, problem = creator()
            self.models[model.name] = model
            self.problems[model.name] = problem

        self.answer_test_system_type = {'SISO': 'ode', 'MIMO_2x1': 'ode', 'MIMO_2x2': 'ode'}
        self.answer_test_n_x = {'SISO': 1, 'MIMO_2x1': 2, 'MIMO_2x2': 2}
        self.answer_test_n_u = {'SISO': 1, 'MIMO_2x1': 1, 'MIMO_2x2': 2}
        self.answer_test_n_y = {'SISO': 0, 'MIMO_2x1': 0, 'MIMO_2x2': 0}
        self.answer_test_n_z = {'SISO': 0, 'MIMO_2x1': 0, 'MIMO_2x2': 0}
        self.answer_test_n_p = {'SISO': 0, 'MIMO_2x1': 0, 'MIMO_2x2': 0}
        self.answer_test_n_theta = {'SISO': 0, 'MIMO_2x1': 0, 'MIMO_2x2': 0}
        self.answer_test_n_yz = {'SISO': 0, 'MIMO_2x1': 0, 'MIMO_2x2': 0}

        # self.answer_test_n_x = {'SISO':, 'MIMO_2x1':, 'MIMO_2x2':}

    def test_system_type(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertEqual(model.system_type, self.answer_test_system_type[model_name])

    def test_n_x(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertEqual(model.n_x, self.answer_test_n_x[model_name])

    def test_n_y(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertEqual(model.n_y, self.answer_test_n_y[model_name])

    def test_n_z(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertEqual(model.n_z, self.answer_test_n_z[model_name])

    def test_n_u(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertEqual(model.n_u, self.answer_test_n_u[model_name])

    def test_n_p(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertEqual(model.n_p, self.answer_test_n_p[model_name])

    def test_n_theta(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertEqual(model.n_theta, self.answer_test_n_theta[model_name])

    def test_x_sys_sym(self):
        for model_name in self.models:
            model = self.models[model_name]
            if model.hasAdjointVariables:
                self.assertTrue(is_equal(model.x_sys_sym, model.x_sym[:model.n_x // 2]))
            else:
                self.assertTrue(is_equal(model.x_sys_sym, model.x_sym))

    def test_lamb_sym(self):
        for model_name in self.models:
            model = self.models[model_name]
            if model.hasAdjointVariables:
                self.assertTrue(is_equal(model.lamb_sym, model.x_sym[model.n_x // 2:]))
            else:
                self.assertTrue(is_equal(model.lamb_sym, SX()))

    def test_all_sym(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertEqual(len(model.all_sym), 8)
            answer = [model.t_sym, model.x_sym, model.y_sym, model.z_sym, model.u_sym, model.p_sym,
                      model.theta_sym, model.u_par]

            for index in range(len(model.all_sym)):
                self.assertTrue(is_equal(model.all_sym[index], answer[index]))

    def test_all_alg(self):
        for model_name in self.models:
            model = self.models[model_name]
            self.assertTrue(is_equal(model.all_alg, vertcat(model.alg, model.alg_z, model.con)))

    def test_replace_variable(self):
        pass

    def test_include_system_equations(self):
        pass

    def test_include_state(self):
        new_state_1 = SX.sym('new_state')
        new_state_2 = SX.sym('new_state_2', 2)
        for model_name in self.models:
            model = self.models[model_name]
            model_n_x = model.n_x
            new_x_0_sym_1 = model.include_state(new_state_1)

            self.assertEqual(model.n_x, model_n_x + 1, )  # Number of state variables has increased
            self.assertEqual(model.x_0_sym.numel(), model_n_x + 1)  # Num. of initial cond for the state has increased
            self.assertEqual(new_x_0_sym_1.numel(), new_state_1.numel())  # The returned initial cond var == size added
            self.assertTrue(is_equal(model.x_sym[-1], new_state_1))  # The added var is the in the x_sym

            new_x_0_sym_2 = model.include_state(new_state_2)  # Number of state variables has increased
            self.assertEqual(model.n_x, model_n_x + 1 + 2)
            self.assertEqual(model.x_0_sym.numel(),
                             model_n_x + 1 + 2)  # Num. of initial cond for the state has increased
            self.assertEqual(new_x_0_sym_2.numel(), new_state_2.numel())  # The returned initial cond var == size added
            self.assertTrue(is_equal(model.x_sym[-3], new_state_1))
            self.assertTrue(is_equal(model.x_sym[-2:], new_state_2))

    def test_include_algebraic(self):
        pass

    def test_include_external_algebraic(self):
        pass

    def test_include_connecting_equations(self):
        pass

    def test_include_control(self):
        pass

    def test_include_parameter(self):
        pass

    def test_include_theta(self):
        pass

    def test_remove_variables_from_vector(self):
        pass

    def test_remove_algebraic(self):
        pass

    def test_remove_external_algebraic(self):
        pass

    def test_remove_connecting_equations(self):
        pass

    def test_remove_control(self):
        pass

    def test_slice_yz_to_y_and_z(self):
        pass

    def test_concat_y_and_z(self):
        pass

    def test_put_values_in_all_sym_format(self):
        pass

    def test_convert_from_time_to_tau(self):
        pass

    def test_convert_expr_from_tau_to_time(self):
        pass

    def test_merge(self):
        pass

    def test_get_dae_system(self):
        pass

    def test_simulate(self):
        pass

    def test_simulate_step(self):
        pass

    def test_simulate_interval(self):
        pass

    def test__create_integrator(self):
        pass

    def test__create_explicit_integrator(self):
        pass

    def test_find_variables_indices_in_vector(self):
        pass

    def test_linearize(self):
        pass
