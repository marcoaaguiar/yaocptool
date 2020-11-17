# -*- coding: utf-8 -*-
"""
Created on $date

@author: Marco Aurelio Schmitz de Aguiar
"""
import sys
from unittest import TestCase

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock

from casadi import is_equal, DM, MX, inf, repmat

from yaocptool.optimization.abstract_optimization_problem import (
    AbstractOptimizationProblem,
)


class TestAbstractOptimizationProblem(TestCase):
    def test_create_variable(self):
        aop = AbstractOptimizationProblem()
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 3)
        self.assertTrue(is_equal(aop.x, x))

    def test_include_variable(self):
        aop = AbstractOptimizationProblem()
        x = MX.sym("x", 3)
        aop.include_variable(x)
        self.assertTrue(is_equal(aop.x, x))

    def test_include_variable_twice(self):
        # Test to avoid override
        aop = AbstractOptimizationProblem()
        x = MX.sym("x", 3)
        y = MX.sym("y", 5)
        aop.include_variable(x)
        aop.include_variable(y)

        self.assertTrue(is_equal(aop.x[: x.numel()], x))
        self.assertTrue(is_equal(aop.x[x.numel() :], y))

    def test_include_variable_with_bounds(self):
        lb = -DM(range(1, 4))
        ub = DM(range(5, 8))
        aop = AbstractOptimizationProblem()
        x = MX.sym("x", 3)
        aop.include_variable(x, lb=lb, ub=ub)
        self.assertTrue(is_equal(aop.x_lb, lb))
        self.assertTrue(is_equal(aop.x_ub, ub))

    def test_include_variable_lb_greater_than_ub(self):
        ub = -DM(range(1, 4))
        lb = DM(range(5, 8))
        aop = AbstractOptimizationProblem()
        self.assertRaises(
            ValueError, aop.create_variable, name="x", size=3, lb=lb, ub=ub
        )

    def test_include_variable_ub_wrong_size(self):
        ub = -DM(range(1, 5))
        lb = DM(range(5, 8))
        aop = AbstractOptimizationProblem()
        self.assertRaises(
            ValueError, aop.create_variable, name="x", size=3, lb=lb, ub=ub
        )

    def test_include_variable_lb_wrong_size(self):
        ub = -DM(range(1, 4))
        lb = DM(range(5, 10))
        aop = AbstractOptimizationProblem()
        self.assertRaises(
            ValueError, aop.create_variable, name="x", size=3, lb=lb, ub=ub
        )

    def test_include_variable_parameter_in_bound(self):
        aop = AbstractOptimizationProblem()
        p = aop.create_parameter("p")
        self.assertRaises(ValueError, aop.create_variable, name="x", size=3, lb=p, ub=p)

    def test_create_parameter(self):
        aop = AbstractOptimizationProblem()
        p = aop.create_parameter("p")

        self.assertTrue(is_equal(aop.p, p))

    def test_include_parameter(self):
        aop = AbstractOptimizationProblem()
        p = MX.sym("p", 2)
        aop.include_parameter(p)

        self.assertTrue(is_equal(aop.p, p))

    def test_include_parameter_twice(self):
        # Test to avoid override
        aop = AbstractOptimizationProblem()
        p1 = MX.sym("p1", 3)
        p2 = MX.sym("p2", 5)
        aop.include_parameter(p1)
        aop.include_parameter(p2)
        self.assertTrue(is_equal(aop.p[: p1.numel()], p1))
        self.assertTrue(is_equal(aop.p[p1.numel() :], p2))

    def test_set_objective(self):
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        f = x[0] ** 2 + x[1] ** 2
        aop.set_objective(f)

        self.assertTrue(is_equal(aop.f, f))

    def test_set_objective_w_mock(self):
        obj = Mock()
        obj.numel = lambda: 1
        aop = AbstractOptimizationProblem()
        aop.set_objective(obj)

        self.assertEqual(obj, aop.f)

    def test_set_objective_wrong_size(self):
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)

        self.assertRaises(ValueError, aop.set_objective, x ** 2)

    def test_get_problem_dict(self):
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 3)
        p = aop.create_parameter("p", 3)
        f = sum(x[i] ** 2 for i in range(x.numel()))
        g = x[0] - x[1] + 2 * x[2]
        aop.set_objective(f)
        aop.include_inequality(g, lb=-10, ub=20)
        d_res = {"x": x, "p": p, "f": f, "g": g}

        d = aop.get_problem_dict()
        for key in set(d_res.keys()).union(set(d.keys())):
            self.assertTrue(is_equal(d_res[key], d[key]))

    def test_get_solver(self):
        aop = AbstractOptimizationProblem()
        self.assertRaises(NotImplementedError, aop.get_solver)

    def test__create_solver(self):
        aop = AbstractOptimizationProblem()
        self.assertRaises(NotImplementedError, aop.solve, None)

    def test_get_default_call_dict(self):
        aop = AbstractOptimizationProblem()

        lbx = -DM([2, 3, 10])
        ubx = DM([2, 3, 10])

        x = aop.create_variable("x", 3, lb=lbx, ub=ubx)
        p = aop.create_parameter("p", 3)
        f = sum(x[i] ** 2 for i in range(x.numel()))
        g = x[0] - x[1] + 2 * x[2]

        aop.set_objective(f)
        aop.include_inequality(g, lb=-10, ub=20)

        expected = {"lbx": lbx, "ubx": ubx, "lbg": DM(-10), "ubg": DM(20)}

        d = aop.get_default_call_dict()
        for key in set(expected.keys()).union(set(d.keys())):
            self.assertTrue(is_equal(expected[key], d[key]))

    def test_solve(self):
        aop = AbstractOptimizationProblem()
        self.assertRaises(NotImplementedError, aop.solve, None)

    def test_include_inequality(self):
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = x[0] - x[1]
        aop.include_inequality(g)
        self.assertTrue(is_equal(aop.g, g))
        self.assertTrue(is_equal(aop.g_lb, -inf))
        self.assertTrue(is_equal(aop.g_ub, inf))

    def test_include_inequality_with_bounds(self):
        lb = 2
        ub = 3
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = x[0] - x[1]
        aop.include_inequality(g, lb=lb, ub=ub)
        self.assertTrue(is_equal(aop.g, g))
        self.assertTrue(is_equal(aop.g_lb, lb))
        self.assertTrue(is_equal(aop.g_ub, ub))

    def test_include_inequality_with_lb_greater_ub(self):
        lb = 5
        ub = 1
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        self.assertRaises(ValueError, aop.include_inequality, x[0] - x[1], lb=lb, ub=ub)

    def test_include_inequality_scalar_bound(self):
        lb = 1
        ub = 4
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = 2 * x
        aop.include_inequality(g, lb=lb, ub=ub)
        self.assertTrue(is_equal(aop.g_lb, repmat(lb, 2)))
        self.assertTrue(is_equal(aop.g_ub, repmat(ub, 2)))

    def test_include_inequality_ub_wrong_size(self):
        ub = -DM(range(1, 5))
        lb = DM(range(5, 8))
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = x[0] - x[1]
        self.assertRaises(ValueError, aop.include_inequality, g, lb=lb, ub=ub)

    def test_include_inequality_w_external_variable_in_bound(self):
        theta = MX.sym("theta")
        lb = -theta
        ub = theta
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = x[0] - x[1]
        aop.include_inequality(g, lb=lb, ub=ub)
        self.assertTrue(is_equal(aop.g, g))
        self.assertTrue(is_equal(aop.g_lb, lb))
        self.assertTrue(is_equal(aop.g_ub, ub))

    def test_include_inequality_w_external_variable_in_expr(self):
        theta = MX.sym("theta")

        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = theta * x[0] - x[1]
        aop.include_inequality(g)
        self.assertTrue(is_equal(aop.g, g))
        self.assertTrue(is_equal(aop.g_lb, -inf))
        self.assertTrue(is_equal(aop.g_ub, inf))

    def test_include_equality(self):
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = x[0] - x[1]
        aop.include_equality(g)
        self.assertTrue(is_equal(aop.g, g))
        self.assertTrue(is_equal(aop.g_lb, 0))
        self.assertTrue(is_equal(aop.g_ub, 0))

    def test_include_equality_with_bounds(self):
        rhs = 2
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = x[0] - x[1]
        aop.include_equality(g, rhs=rhs)
        self.assertTrue(is_equal(aop.g, g))
        self.assertTrue(is_equal(aop.g_lb, rhs))
        self.assertTrue(is_equal(aop.g_ub, rhs))

    def test_include_equality_scalar_bound(self):
        rhs = 2
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = 2 * x
        aop.include_equality(g, rhs=rhs)
        self.assertTrue(is_equal(aop.g_lb, repmat(rhs, 2)))
        self.assertTrue(is_equal(aop.g_ub, repmat(rhs, 2)))

    def test_include_equality_ub_wrong_size(self):
        rhs = -DM(range(1, 5))
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = x[0] - x[1]
        self.assertRaises(ValueError, aop.include_equality, g, rhs)

    def test_include_equality_w_external_variable_in_bound(self):
        theta = MX.sym("theta")

        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = x[0] - x[1]
        aop.include_equality(g, theta)
        self.assertTrue(is_equal(aop.g, g))
        self.assertTrue(is_equal(aop.g_lb, theta))
        self.assertTrue(is_equal(aop.g_ub, theta))

    def test_include_equality_w_external_variable_in_expr(self):
        theta = MX.sym("theta")

        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)
        g = theta * x[0] - x[1]
        aop.include_equality(g)
        self.assertTrue(is_equal(aop.g, g))
        self.assertTrue(is_equal(aop.g_lb, 0))
        self.assertTrue(is_equal(aop.g_ub, 0))

    def test_include_constraint_inequality(self):
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)

        aop.include_constraint(x + 2 <= 1)

    def test_include_constraint_equality(self):
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)

        aop.include_constraint(x + 2 == 1)

    def test_include_constraint_inequality_w_external_var(self):
        theta = MX.sym("theta")
        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)

        aop.include_constraint(x + 2 <= theta)

    def test_include_constraint_equality_w_external_var(self):
        theta = MX.sym("theta")

        aop = AbstractOptimizationProblem()
        x = aop.create_variable("x", 2)

        aop.include_constraint(x + 2 == theta)
