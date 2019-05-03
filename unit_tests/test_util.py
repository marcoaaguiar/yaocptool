# -*- coding: utf-8 -*-
"""
Created on 

@author: Marco Aurelio Schmitz de Aguiar
"""

import unittest

from casadi import SX, vertcat, is_equal, DM, MX, exp, diag, norm_fro

from yaocptool import find_variables_indices_in_vector, find_variables_in_vector_by_name, \
    remove_variables_from_vector, remove_variables_from_vector_by_indices, create_constant_theta, is_inequality, \
    is_equality, blockdiag, join_thetas, convert_expr_from_tau_to_time, expm, create_polynomial_approximation
from yaocptool.util.util import _create_lagrangian_polynomial_basis


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.x_vector = vertcat(SX.sym('x'),
                                SX.sym('other_name'),
                                SX.sym('the_right_one'),
                                SX.sym('foo'),
                                SX.sym('food'),
                                SX.sym('bar'),
                                SX.sym('var1'),
                                SX.sym('alpha'),
                                SX.sym('cat'))

    def test_find_variables_indices_in_vector(self):
        indices = [3, 4, 8]

        # Try find that work
        var = self.x_vector[indices]
        self.assertEqual(find_variables_indices_in_vector(var, self.x_vector), indices)

        # Try var that is not there
        self.assertEqual(len(find_variables_indices_in_vector(SX.sym('wrong'), self.x_vector)), 0)

    def test_find_variables_in_vector_by_name(self):

        # Only search for one variable
        correct_var = self.x_vector[2]
        correct_var_name = correct_var.name()

        result = find_variables_in_vector_by_name(correct_var_name, self.x_vector)
        self.assertEqual(len(result), 1)
        self.assertTrue(is_equal(result[-1], correct_var))

        # Only search for multiple variables
        correct_var = self.x_vector[[2, 4]]
        correct_var_name = [correct_var[ind].name() for ind in range(correct_var.numel())]

        result = find_variables_in_vector_by_name(correct_var_name, self.x_vector)
        self.assertEqual(len(result), 2)
        for ind in range(correct_var.numel()):
            self.assertTrue(is_equal(result[ind], correct_var[ind]))

        # Search for name using regex
        correct_var = self.x_vector[[3, 4]]
        correct_var_name = 'foo'

        result = find_variables_in_vector_by_name(correct_var_name, self.x_vector)
        self.assertEqual(len(result), 2)
        for ind in range(correct_var.numel()):
            self.assertTrue(is_equal(result[ind], correct_var[ind]))

        # Search for name using exact
        correct_var = self.x_vector[3]
        correct_var_name = correct_var.name()

        result = find_variables_in_vector_by_name(correct_var_name, self.x_vector, exact=True)
        self.assertEqual(len(result), 1)
        self.assertTrue(is_equal(result[-1], correct_var))

        # Search for variable that does not exists with exact
        result = find_variables_in_vector_by_name('wrong', self.x_vector, exact=True)
        self.assertEqual(len(result), 0)
        self.assertEqual(result, [])

        # Search for variable that does not exists with regex
        result = find_variables_in_vector_by_name('wrong', self.x_vector, exact=True)
        self.assertEqual(len(result), 0)
        self.assertEqual(result, [])

    def test_remove_variables_from_vector(self):
        removed_variables_ind = [2, 6, 3]
        remaining_ind = [ind for ind in range(self.x_vector.numel()) if ind not in removed_variables_ind]

        # Assert if the variable is being removed
        var_tor_remove = self.x_vector[removed_variables_ind]
        correct_res = self.x_vector[remaining_ind]
        self.assertTrue(is_equal(remove_variables_from_vector(var_tor_remove, self.x_vector), correct_res))

        # Test remove variable that it is not in the vector
        self.assertRaises(ValueError, remove_variables_from_vector, SX.sym('wrong_var'), self.x_vector)

    def test_remove_variables_from_vector_by_indices(self):
        removed_variables_ind = [2, 6, 3]
        remaining_ind = [ind for ind in range(self.x_vector.numel()) if ind not in removed_variables_ind]

        # Test removing
        correct_res = self.x_vector[remaining_ind]
        self.assertTrue(is_equal(remove_variables_from_vector_by_indices(removed_variables_ind,
                                                                         self.x_vector),
                                 correct_res))

        # Test remove empty list
        self.assertTrue(is_equal(remove_variables_from_vector_by_indices([],
                                                                         self.x_vector),
                                 self.x_vector))

        # Test removing out of bounds
        self.assertRaises(ValueError, remove_variables_from_vector_by_indices, [100], self.x_vector)
        self.assertRaises(ValueError, remove_variables_from_vector_by_indices, [-100], self.x_vector)

    def test_create_constant_theta(self):
        constant = 3
        dimension = 4
        finite_elements = 5

        # test normal use
        res_dict = dict(zip(range(finite_elements), [constant * DM.ones(dimension, 1)] * finite_elements))
        theta = create_constant_theta(constant, dimension, finite_elements)
        for el in range(finite_elements):
            self.assertTrue(is_equal(theta[el], res_dict[el]))

        # Test 0 dimension
        res_dict = dict(zip(range(finite_elements), [constant * DM.ones(0, 1)] * finite_elements))
        theta = create_constant_theta(constant, 0, finite_elements)
        for el in range(finite_elements):
            self.assertTrue(is_equal(theta[el], res_dict[el]))

    def test_join_thetas(self):
        # with 3 dicts with same size
        dimension = 2
        finite_elements = 5

        theta_1 = create_constant_theta(1, dimension, finite_elements)
        theta_2 = create_constant_theta(2, dimension + 1, finite_elements)
        theta_3 = create_constant_theta(3, dimension + 2, finite_elements)
        res = join_thetas(theta_1, theta_2, theta_3)
        self.assertEqual(len(res), finite_elements)
        for el in range(finite_elements):
            self.assertEqual(res[el].shape, (3 * dimension + 3, 1))

        # with a dict and empty dict
        dimension = 2
        finite_elements = 5

        theta_1 = create_constant_theta(1, dimension, finite_elements)
        theta_2 = {}
        res = join_thetas(theta_1, theta_2)
        self.assertEqual(len(res), finite_elements)
        for el in range(finite_elements):
            self.assertEqual(res[el].shape, (dimension, 1))

        # with a dict and None
        dimension = 2
        finite_elements = 5

        theta_1 = create_constant_theta(1, dimension, finite_elements)
        theta_2 = None
        res = join_thetas(theta_1, theta_2)
        self.assertEqual(len(res), finite_elements)
        for el in range(finite_elements):
            self.assertEqual(res[el].shape, (dimension, 1))

        # with a dict and None
        dimension = 2
        finite_elements = 5

        theta_1 = create_constant_theta(1, dimension, finite_elements)
        theta_2 = create_constant_theta(1, dimension, finite_elements + 5)

        self.assertRaises(ValueError, join_thetas, theta_1, theta_2)

    def test_convert_expr_from_tau_to_time(self):
        t = SX.sym('t')
        tau = SX.sym('tau')
        expr = tau
        t_0 = 5
        t_f = 15

        correct_res = (t - t_0) / (t_f - t_0)
        res = convert_expr_from_tau_to_time(expr, t, tau, t_0, t_f)
        self.assertTrue(is_equal(res, correct_res, 10))

    def test_blockdiag(self):
        # Test blockdiag with DM
        correct_res = DM([[1, 1, 0, 0, 0],
                          [1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1]])

        a = DM.ones(2, 2)
        b = DM.ones(3, 3)
        res = blockdiag(a, b)
        self.assertTrue(is_equal(res, correct_res))

        # MX and DM mix
        a = MX.sym('a', 2, 2)
        b = DM.ones(1, 1)
        correct_res = MX.zeros(3, 3)
        correct_res[:2, :2] = a
        correct_res[2:, 2:] = b

        res = blockdiag(a, b)
        self.assertTrue(is_equal(res, correct_res, 30))

        # SX and DM mix
        a = SX.sym('a', 2, 2)
        b = DM.ones(1, 1)
        correct_res = SX.zeros(3, 3)
        correct_res[:2, :2] = a
        correct_res[2:, 2:] = b

        res = blockdiag(a, b)
        self.assertTrue(is_equal(res, correct_res, 30))

        # SX and MX
        a = SX.sym('a', 2, 2)
        b = MX.sym('b', 2, 2)
        self.assertRaises(ValueError, blockdiag, a, b)

    def test_expm(self):
        # Test for eye
        correct_res = diag(exp(DM.ones(3)))
        a = expm(DM.eye(3))
        self.assertAlmostEqual(norm_fro(a - correct_res), 0, 3)

        # Test for -magic(3) (compared with MATLAB solution)
        a = DM([[-8, - 1, - 6],
                [-3, - 5, - 7],
                [-4, - 9, - 2]])

        correct_res = DM([[3.646628887990924, 32.404567030885005, -36.051195612973601],
                          [5.022261973341555, 44.720086474306093, -49.742348141745325],
                          [-8.668890555430160, -77.124653199288772, 85.793544060621244]])
        self.assertAlmostEqual(norm_fro(expm(a) - correct_res), 0, 2)

    def test_is_inequality(self):
        x_vector = vertcat(MX.sym('x'),
                           MX.sym('other_name'),
                           MX.sym('the_right_one'),
                           MX.sym('foo'),
                           MX.sym('food'),
                           MX.sym('bar'),
                           MX.sym('var1'),
                           MX.sym('alpha'),
                           MX.sym('cat'))

        self.assertTrue(is_inequality(x_vector[1] >= 1))
        self.assertTrue(is_inequality(x_vector[1] <= 1))
        self.assertFalse(is_inequality(x_vector[1] == 1))

        self.assertTrue(is_inequality(x_vector >= 1))
        self.assertTrue(is_inequality(x_vector <= 1))
        self.assertFalse(is_inequality(x_vector == 1))

    def test_is_equality(self):
        x_vector = vertcat(MX.sym('x'),
                           MX.sym('other_name'),
                           MX.sym('the_right_one'),
                           MX.sym('foo'),
                           MX.sym('food'),
                           MX.sym('bar'),
                           MX.sym('var1'),
                           MX.sym('alpha'),
                           MX.sym('cat'))

        self.assertTrue(is_equality(x_vector[1] == 1))
        self.assertFalse(is_equality(x_vector[1] >= 1))
        self.assertFalse(is_equality(x_vector[1] <= 1))

        self.assertTrue(is_equality(x_vector == 1))
        self.assertFalse(is_equality(x_vector >= 1))
        self.assertFalse(is_equality(x_vector <= 1))

    def test__create_lagrangian_polynomial_basis(self):
        tau = SX.sym('tau')
        degree = 3

        # Test without point_at_zeo
        l_list = _create_lagrangian_polynomial_basis(tau, degree, False)
        self.assertEqual(len(l_list), degree)

        # Test with point_at_zeo
        l_list = _create_lagrangian_polynomial_basis(tau, degree, True)
        self.assertEqual(len(l_list), degree + 1)

    def test_create_polynomial_approximation(self):
        tau = SX.sym('t')
        size = 3
        degree = 3

        # Pol
        pol, par = create_polynomial_approximation(tau, size, degree, name='func')

        self.assertEqual(pol.shape, (size, 1))
        self.assertEqual(par.shape, (size * degree, 1))

        # Pol point at zero
        pol, par = create_polynomial_approximation(tau, size, degree, name='func', point_at_zero=True)

        self.assertEqual(pol.shape, (size, 1))
        self.assertEqual(par.shape, (size * (degree + 1), 1))

        # Pol degree = 1
        pol, par = create_polynomial_approximation(tau, size, 1, name='func')

        self.assertEqual(pol.shape, (size, 1))
        self.assertEqual(par.shape, (size, 1))

        # Pol size = zero
        size_zero = 0
        pol, par = create_polynomial_approximation(tau, size_zero, degree, name='func')

        self.assertEqual(pol.shape, (size_zero, 1))
        self.assertEqual(par.shape, (size_zero * degree, 1))

        # Pol size = zero, start_at_zero = True
        size_zero = 0
        pol, par = create_polynomial_approximation(tau, size_zero, degree, name='func', point_at_zero=True)

        self.assertEqual(pol.shape, (size_zero, 1))
        self.assertEqual(par.shape, (size_zero * (degree + 1), 1))

        # Pol degree= 1, size = zero
        degree_one = 1
        size_zero = 0
        pol, par = create_polynomial_approximation(tau, size_zero, degree_one, name='func', point_at_zero=False)

        self.assertEqual(pol.shape, (size_zero, 1))
        self.assertEqual(par.shape, (size_zero * degree_one, 1))

        # Pol degree= 1, size = zero, start_at_zero = True
        degree_one = 1
        size_zero = 0
        pol, par = create_polynomial_approximation(tau, size_zero, degree_one, name='func', point_at_zero=True)

        self.assertEqual(pol.shape, (size_zero, 1))
        self.assertEqual(par.shape, (size_zero * (degree + 1), 1))
