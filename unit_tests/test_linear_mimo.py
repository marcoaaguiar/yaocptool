from __future__ import print_function
import unittest

from casadi import DM, mtimes, inf
from yaocptool.methods import DirectMethod, IndirectMethod
from yaocptool.modelling.model_classes import SystemModel
from yaocptool.modelling.ocp import OptimalControlProblem
from models import create_2x2_mimo



class MIMO2x2TestCase(unittest.TestCase):
    @property
    def _create_model_and_problem(self):
        return create_2x2_mimo

    def setUp(self):
        self.model, self.problem = create_2x2_mimo()
        self.obj_tol = 1e-8
        self.obj_value = 0#DM(0.131427)
        self.nlpsol_opts = {
            'ipopt.print_level': 0,
            'print_time': False
        }

    # region TEST MODEL
    def test_number_of_states(self):
        self.assertEqual(self.model.n_x, 2)

    def test_number_of_controls(self):
        self.assertEqual(self.model.n_u, 2)

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
        problem.create_cost_state()
        self.assertEqual(problem.x_min[-1], -inf)

        model, problem = self._create_model_and_problem()
        problem.positive_objective = True
        problem.create_cost_state()
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
        print(result.objective)
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
        print(result.objective)
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
        print(result.objective)
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
        print(result.objective)
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
        print(result.objective)
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
