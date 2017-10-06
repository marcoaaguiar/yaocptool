from __future__ import print_function
import unittest

from casadi import DM, mtimes, inf
from yaocptool.methods import DirectMethod, IndirectMethod
from yaocptool.modelling.model_classes import SystemModel
from yaocptool.modelling.ocp import OptimalControlProblem


def _create_model_and_problem():
    model = SystemModel(name='MIMO_2x2', Nx=2, Nu=2)
    x = model.x_sym
    a = DM([[-1, -2], [5, -1]])
    model.include_system_equations(mtimes(a, x))

    problem = OptimalControlProblem(model, obj={'Q': DM.ones(2, 2), 'R': DM.ones(2, 2)}, x_0=[1, 1])
    return model, problem


class MIMO2x2TestCase(unittest.TestCase):
    @property
    def _create_model_and_problem(self):
        return _create_model_and_problem

    def setUp(self):
        self.model, self.problem = _create_model_and_problem()
        self.obj_tol = 1e-5
        self.obj_value = DM(1.35676)
        self.nlpsol_opts = {
            'ipopt.print_level': 0,
            'print_time': False
        }

    # region TEST MODEL
    def test_number_of_states(self):
        self.assertEqual(self.model.Nx, 2)

    def test_number_of_controls(self):
        self.assertEqual(self.model.Nu, 2)

    # endregion

    def test_problem_x_0(self):
        self.assertEqual(self.problem.x_0.numel(), 2)

    def test_problem_bounds_size(self):
        self.assertEqual(self.problem.x_max.numel(), 2)
        self.assertEqual(self.problem.x_min.numel(), 2)
        self.assertEqual(self.problem.u_max.numel(), 2)
        self.assertEqual(self.problem.u_min.numel(), 2)

    def test_problem_positive_objective(self):
        model, problem = self._create_model_and_problem()
        problem.createCostState()
        self.assertEqual(problem.x_min[-1], -inf)

        model, problem = self._create_model_and_problem()
        problem.positive_objective = True
        problem.createCostState()
        self.assertEqual(problem.x_min[-1], 0)

    # region DIRECT MULTIPLE SHOOTING

    def test_direct_multiple_shooting_explicit_pw_cont_control(self):
        model, problem = self._create_model_and_problem()
        solution_method = DirectMethod(problem, degree=3, degree_control=1,
                                       finite_elements=20,
                                       integrator_type='explicit',
                                       discretization_scheme='multiple-shooting',
                                       nlpsol_opts=self.nlpsol_opts
                                       )
        result = solution_method.solve()
        self.assertAlmostEqual(result.objective, self.obj_value, delta=self.obj_tol)

    def test_direct_multiple_shooting_implicit_pw_cont_control(self):
        model, problem = self._create_model_and_problem()
        solution_method = DirectMethod(problem, degree=3, degree_control=1,
                                       finite_elements=20,
                                       integrator_type='implicit',
                                       discretization_scheme='multiple-shooting',
                                       nlpsol_opts=self.nlpsol_opts
                                       )
        result = solution_method.solve()
        self.assertAlmostEqual(result.objective, self.obj_value, delta=self.obj_tol)

    def test_direct_multiple_shooting_explicit_polynomial_control(self):
        model, problem = self._create_model_and_problem()
        solution_method = DirectMethod(problem, degree=3, degree_control=3,
                                       finite_elements=20,
                                       integrator_type='explicit',
                                       discretization_scheme='multiple-shooting',
                                       nlpsol_opts=self.nlpsol_opts
                                       )
        result = solution_method.solve()
        self.assertAlmostEqual(result.objective, self.obj_value, delta=self.obj_tol)

    def test_direct_multiple_shooting_implicit_polynomial_control(self):
        model, problem = self._create_model_and_problem()
        solution_method = DirectMethod(problem, degree=3, degree_control=3,
                                       finite_elements=20,
                                       integrator_type='implicit',
                                       discretization_scheme='multiple-shooting',
                                       nlpsol_opts=self.nlpsol_opts
                                       )
        result = solution_method.solve()
        self.assertAlmostEqual(result.objective, self.obj_value, delta=self.obj_tol)

    # endregion

    # region DIRECT COLLOCAITON METHOD
    def test_direct_collocation_polynomial_control(self):
        model, problem = self._create_model_and_problem()
        solution_method = DirectMethod(problem, degree=3, degree_control=3,
                                       finite_elements=20,
                                       discretization_scheme='collocation',
                                       nlpsol_opts=self.nlpsol_opts
                                       )
        result = solution_method.solve()
        self.assertAlmostEqual(result.objective, self.obj_value, delta=self.obj_tol)

    def test_direct_collocation_pw_cont_control(self):
        model, problem = self._create_model_and_problem()
        solution_method = DirectMethod(problem, degree=3, degree_control=1,
                                       finite_elements=20,
                                       integrator_type='explicit',
                                       discretization_scheme='collocation',
                                       nlpsol_opts=self.nlpsol_opts
                                       )
        result = solution_method.solve()
        print(result.objective)
        self.assertAlmostEqual(result.objective, self.obj_value, delta=self.obj_tol)

    # endregion
    # region INIDRECT METHOD
    def test_indirect_multiple_collocation(self):
        model, problem = self._create_model_and_problem()
        solution_method = IndirectMethod(problem, degree=3, degree_control=3,
                                         finite_elements=20,
                                         discretization_scheme='collocation',
                                         nlpsol_opts=self.nlpsol_opts
                                         )
        result = solution_method.solve()
        self.assertAlmostEqual(result.objective, 0, delta=self.obj_tol)

    def test_indirect_multiple_shooting_implicit(self):
        model, problem = self._create_model_and_problem()
        solution_method = IndirectMethod(problem, degree=3, degree_control=3,
                                         finite_elements=20,
                                         integrator_type='implicit',
                                         discretization_scheme='multiple-shooting',
                                         nlpsol_opts=self.nlpsol_opts
                                         )
        result = solution_method.solve()
        self.assertAlmostEqual(result.objective, 0, delta=self.obj_tol)

    def test_indirect_multiple_shooting_explicit(self):
        model, problem = self._create_model_and_problem()
        solution_method = IndirectMethod(problem, degree=3, degree_control=3,
                                         finite_elements=20,
                                         integrator_type='explicit',
                                         discretization_scheme='multiple-shooting',
                                         nlpsol_opts=self.nlpsol_opts
                                         )
        result = solution_method.solve()
        self.assertAlmostEqual(result.objective, 0, delta=self.obj_tol)
        # endregion


if __name__ == '__main__':
    unittest.main(exit=False)
