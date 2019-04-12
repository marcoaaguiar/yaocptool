# -*- coding: utf-8 -*-
"""
Created on $date

@author: Marco Aurelio Schmitz de Aguiar
"""
import copy
from unittest import TestCase

from casadi import SX, is_equal, substitute, vertcat

from yaocptool.modelling import DAESystem


class TestDAESystem(TestCase):
    def test_is_dae_true(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        t = SX.sym('t')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, t=t)

        self.assertTrue(sys.is_dae)

    def test_is_dae_false(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        t = SX.sym('t')

        ode = -2 * x

        sys = DAESystem(x=x, y=y, ode=ode, t=t)
        self.assertFalse(sys.is_dae)

    def test_is_ode_true(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        p = SX.sym('p', 3)
        t = SX.sym('t')

        ode = -2 * x

        sys = DAESystem(x=x, y=y, p=p, ode=ode, t=t)
        self.assertTrue(sys.is_ode)

    def test_is_ode_false(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        t = SX.sym('t')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, t=t)

        self.assertFalse(sys.is_ode)

    def test_type_dae(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        t = SX.sym('t')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, t=t)

        self.assertEqual(sys.type, 'dae')

    def test_type_ode(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        t = SX.sym('t')

        ode = -2 * x

        sys = DAESystem(x=x, y=y, ode=ode, t=t)

        self.assertEqual(sys.type, 'ode')

    def test_has_parameters_true(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        t = SX.sym('t')
        p = SX.sym('p')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2 + p

        sys = DAESystem(x=x, y=y, p=p, ode=ode, alg=alg, t=t)

        self.assertTrue(sys.has_parameters)

    def test_has_parameters_false(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        t = SX.sym('t')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, t=t)

        self.assertFalse(sys.has_parameters)

    def test_dae_system_dict_dae_with_p(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        p = SX.sym('p')
        t = SX.sym('t')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, t=t, p=p)
        res = {'x': x, 'z': y, 'p': p, 'ode': ode, 'alg': alg, 't': t}
        self.assertEqual(set(res.keys()), set(sys.dae_system_dict.keys()))

        for key in res:
            self.assertTrue(is_equal(res[key], sys.dae_system_dict[key]))

    def test_dae_system_dict_dae_wo_p(self):
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        t = SX.sym('t')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, t=t)
        res = {'x': x, 'z': y, 'ode': ode, 'alg': alg, 't': t}
        self.assertEqual(set(res.keys()), set(sys.dae_system_dict.keys()))

        for key in res:
            self.assertTrue(is_equal(res[key], sys.dae_system_dict[key]))

    def test_dae_system_dict_ode_with_p(self):
        x = SX.sym('x', 2)
        p = SX.sym('p')
        t = SX.sym('t')

        ode = -2 * x

        sys = DAESystem(x=x, ode=ode, t=t, p=p)
        res = {'x': x, 'ode': ode, 't': t, 'p': p}
        self.assertEqual(set(res.keys()), set(sys.dae_system_dict.keys()))

        for key in res:
            self.assertTrue(is_equal(res[key], sys.dae_system_dict[key]))

    def test_dae_system_dict_ode_wo_p(self):
        x = SX.sym('x', 2)
        t = SX.sym('t')

        ode = -2 * x

        sys = DAESystem(x=x, ode=ode, t=t)
        res = {'x': x, 'ode': ode, 't': t}
        self.assertEqual(set(res.keys()), set(sys.dae_system_dict.keys()))

        for key in res:
            self.assertTrue(is_equal(res[key], sys.dae_system_dict[key]))

    def test_has_variable(self):
        # Make system
        x = SX.sym('x')
        y = SX.sym('y')
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = -2 * x + y * p * tau
        alg = y - x - p * t

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Test
        # Variables that it has
        self.assertTrue(sys.has_variable(x))
        self.assertTrue(sys.has_variable(y))
        self.assertTrue(sys.has_variable(p))
        self.assertTrue(sys.has_variable(t))
        self.assertTrue(sys.has_variable(tau))

        # Variable that it does not have
        self.assertFalse(sys.has_variable(SX.sym('foo')))

    def test_depends_on(self):
        # Make system
        x = SX.sym('x')
        y = SX.sym('y')
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = -2 * x + y * p
        alg = y - x - p * t

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Test
        # Variables that it has
        self.assertTrue(sys.depends_on(x))
        self.assertTrue(sys.depends_on(y))
        self.assertTrue(sys.depends_on(p))
        self.assertTrue(sys.depends_on(t))

        # Has tau but does not depend on it (on the equations)
        self.assertFalse(sys.depends_on(tau))

        # Variable that it does not have
        self.assertFalse(sys.depends_on(SX.sym('foo')))

    def test_convert_from_tau_to_time_missing_time(self):
        # Make system
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        p = SX.sym('p')
        t = SX.sym('t')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, t=t, p=p)

        # Test
        self.assertRaises(AttributeError, sys.convert_from_tau_to_time, 0, 5)

    def test_convert_from_tau_to_time_missing_tau(self):
        # Make system
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        p = SX.sym('p')
        tau = SX.sym('tau')

        ode = -2 * x
        alg = y - x[0] + x[1] ** 2

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, p=p)

        # Test
        self.assertRaises(AttributeError, sys.convert_from_tau_to_time, 0, 5)

    def test_convert_from_tau_to_time(self):
        # Make system
        x = SX.sym('x', 2)
        y = SX.sym('y', 1)
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = tau
        alg = 1 - tau

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Test
        res = {'ode': t / 5, 'alg': (1 - t / 5)}
        sys.convert_from_tau_to_time(0, 5)
        self.assertTrue(is_equal(res['ode'], sys.ode, 10))
        self.assertTrue(is_equal(res['alg'], sys.alg, 10))

    def test_substitute_variable_x(self):
        # Make system
        x = SX.sym('x')
        y = SX.sym('y')
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = -2 * x + y * p
        alg = y - x - p

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Test
        new_x = SX.sym('new_x')

        sys.substitute_variable(x, new_x)

        res = {'ode': -2 * new_x + y * p, 'alg': y - new_x - p, 'x': new_x, 'y': y, 'p': p, 't': t, 'tau': tau}
        for key in res:
            self.assertTrue(is_equal(res[key], sys.__dict__[key], 10))

    def test_substitute_variable_y(self):
        # Make system
        x = SX.sym('x')
        y = SX.sym('y')
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = -2 * x + y * p
        alg = y - x - p

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Test
        new_y = SX.sym('new_y')

        sys.substitute_variable(y, new_y)

        res = {'ode': -2 * x + new_y * p, 'alg': new_y - x - p, 'x': x, 'y': new_y, 'p': p, 't': t, 'tau': tau}
        for key in res:
            self.assertTrue(is_equal(res[key], sys.__dict__[key], 10))

    def test_substitute_variable_p(self):
        # Make system
        x = SX.sym('x')
        y = SX.sym('y')
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = -2 * x + y * p
        alg = y - x - p

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Test
        new_p = SX.sym('new_p')

        sys.substitute_variable(p, new_p)

        res = {'ode': -2 * x + y * new_p, 'alg': y - x - new_p, 'x': x, 'y': y, 'p': new_p, 't': t, 'tau': tau}
        for key in res:
            self.assertTrue(is_equal(res[key], sys.__dict__[key], 10))

    def test_join(self):
        # Make system 1
        x = SX.sym('x')
        y = SX.sym('y')
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = -2 * x + y * p * tau
        alg = y - x - p * t

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Make system 2
        x2 = SX.sym('x2')
        y2 = SX.sym('y2')
        p2 = SX.sym('p2')
        t2 = SX.sym('t2')
        tau2 = SX.sym('tau2')

        ode2 = -2 * x2 + y2 * p2 * tau2
        alg2 = y2 - x2 - p2 * t2

        sys2 = DAESystem(x=x2, y=y2, ode=ode2, alg=alg2, tau=tau2, t=t2, p=p2)

        # Test
        res_sys = copy.copy(sys)
        res_sys.join(sys2)

        # check if variables and equations were passed
        for key in ['x', 'y', 'ode', 'alg', 'p']:
            self.assertTrue(is_equal(res_sys.__dict__[key][0], sys.__dict__[key], 30))
            self.assertTrue(is_equal(res_sys.__dict__[key][1], substitute(sys2.__dict__[key],
                                                                          vertcat(sys2.tau, sys2.t),
                                                                          vertcat(sys.tau, sys.t)), 30))

        # check if t and tau was passed (it shouldn't)
        self.assertFalse(is_equal(res_sys.t, sys2.t))
        self.assertFalse(is_equal(res_sys.tau, sys2.tau))

        # if the joined equation still depends on sys2 time varibles
        self.assertFalse(res_sys.depends_on(sys2.t))
        self.assertFalse(res_sys.depends_on(sys2.tau))

    def test_simulate_tf_equal_t0(self):
        # Make system 1
        x = SX.sym('x')
        y = SX.sym('y')
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = -2 * x + y * p * tau
        alg = y - x - p * t

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Test
        self.assertRaises(ValueError, sys.simulate, x_0=1, t_f=0, t_0=0)

    def test_simulate_has_tau(self):
        # Make system 1
        x = SX.sym('x')
        y = SX.sym('y')
        p = SX.sym('p')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = -2 * x + y * p * tau
        alg = y - x - p * t

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t, p=p)

        # Test
        self.assertRaises(AttributeError, sys.simulate, x_0=1, t_f=5, t_0=0)

    def test_simulate(self):
        # Make system 1
        x = SX.sym('x')
        y = SX.sym('y')
        t = SX.sym('t')
        tau = SX.sym('tau')

        ode = 1
        alg = y - (5 - t)

        sys = DAESystem(x=x, y=y, ode=ode, alg=alg, tau=tau, t=t)

        # Test
        res = sys.simulate(x_0=0, t_f=5, t_0=0, y_0=1, integrator_options={'abstol': 1e-10})

        self.assertAlmostEqual(res['xf'], 5)
        self.assertAlmostEqual(res['zf'], 0)

    def test__create_integrator(self):
        # self.fail()
        pass

    def test__create_explicit_integrator(self):
        # self.fail()
        pass
