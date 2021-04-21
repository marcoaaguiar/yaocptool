from unittest import TestCase

from casadi import DM, SX, is_equal, vec, vertcat
from casadi.casadi import depends_on, mtimes
from pytest import raises

from yaocptool.modelling import SystemModel
from yaocptool.modelling.utils import der


class TestSystemModel(TestCase):
    def setUp(self):
        # create ode system
        self.ode_model = SystemModel(name="ode_sys")
        x = self.ode_model.create_state("x", 3)
        u = self.ode_model.create_input("u", 3)
        self.ode_model.include_equations(ode=-x + u)

        self.dae_model = SystemModel(name="dae_sys")
        x = self.dae_model.create_state("x", 3)
        y = self.dae_model.create_algebraic_variable("y", 3)
        u = self.dae_model.create_input("u", 3)
        self.dae_model.include_equations(ode=-x + u, alg=y - x + u ** 2)

    def test_system_type(self):
        assert self.ode_model.system_type == "ode"
        assert self.dae_model.system_type, "dae"

    def test_x_sys_sym(self):
        self.assertTrue(is_equal(self.ode_model.x_sys_sym, self.ode_model.x))

        # with adjoints
        ode_x = self.ode_model.x[:]
        dae_x = self.dae_model.x[:]
        self.ode_model.create_state("lamb", self.ode_model.n_x)
        self.ode_model.has_adjoint_variables = True

        self.dae_model.create_state("lamb", self.dae_model.n_x)
        self.dae_model.has_adjoint_variables = True

        self.assertTrue(is_equal(self.ode_model.x_sys_sym, ode_x))
        self.assertTrue(is_equal(self.dae_model.x_sys_sym, dae_x))

    def test_lamb_sym(self):
        self.assertEqual(self.ode_model.lamb_sym.numel(), 0)
        self.assertEqual(self.dae_model.lamb_sym.numel(), 0)

        # with adjoints
        ode_lamb = self.ode_model.create_state("lamb", self.ode_model.n_x)
        self.ode_model.has_adjoint_variables = True

        dae_lamb = self.dae_model.create_state("lamb", self.dae_model.n_x)
        self.dae_model.has_adjoint_variables = True

        self.assertTrue(is_equal(self.ode_model.lamb_sym, ode_lamb))
        self.assertTrue(is_equal(self.dae_model.lamb_sym, dae_lamb))

    def test_all_sym(self):
        for model in [self.ode_model, self.dae_model]:
            answer = [
                model.t,
                model.x,
                model.y,
                model.p,
                model.theta,
                model.u_par,
            ]
            self.assertEqual(len(model.all_sym), len(answer))

            for index in range(len(model.all_sym)):
                self.assertTrue(is_equal(model.all_sym[index], answer[index]))

    def test_t(self):
        self.assertTrue(is_equal(self.ode_model.t, self.ode_model.t))
        self.assertTrue(is_equal(self.dae_model.t, self.dae_model.t))

    def test_t_setter(self):
        new_t = SX.sym("t")
        self.ode_model.t = new_t
        self.dae_model.t = new_t

        self.assertTrue(is_equal(self.ode_model.t, new_t))
        self.assertTrue(is_equal(self.dae_model.t, new_t))

    def test_tau(self):
        self.assertTrue(is_equal(self.ode_model.tau, self.ode_model.tau))
        self.assertTrue(is_equal(self.dae_model.tau, self.dae_model.tau))

    def test_tau_setter(self):
        new_tau = SX.sym("tau")
        self.ode_model.tau = new_tau
        self.dae_model.tau = new_tau

        self.assertTrue(is_equal(self.ode_model.tau, new_tau))
        self.assertTrue(is_equal(self.dae_model.tau, new_tau))

    def test_x_names(self):
        for model in [self.ode_model, self.dae_model]:
            for ind in range(model.n_x):
                self.assertEqual(model.x[ind].name(), model.x_names[ind])

    def test_y_names(self):
        for model in [self.ode_model, self.dae_model]:
            for ind in range(model.n_y):
                self.assertEqual(model.y[ind].name(), model.y_names[ind])

    def test_u_names(self):
        for model in [self.ode_model, self.dae_model]:
            for ind in range(model.n_u):
                self.assertEqual(model.u[ind].name(), model.u_names[ind])

    def test_print_variables(self):
        for model in [self.ode_model, self.dae_model]:
            model.print_variables()

    def test_create_input(self):
        n_u_initial = self.ode_model.n_u
        n_new_u = 4
        u = self.ode_model.create_input("u", n_new_u)
        self.assertEqual(self.ode_model.n_u, n_u_initial + n_new_u)
        self.assertTrue(is_equal(self.ode_model.u[-n_new_u:], u))

    def test_get_variable_by_name(self):
        model = self.dae_model.get_copy()

        # Variable does not exist:
        self.assertRaises(ValueError, model.get_variable_by_name, "other_var")

        # Multiple vars
        self.assertRaises(ValueError, model.get_variable_by_name, "x")

        # Some var that exists and is unique
        var = model.get_variable_by_name("x_1")
        self.assertTrue(is_equal(model.x[1], var))

    def test_get_variables_by_names(self):
        model = self.dae_model.get_copy()

        # Variable does not exist
        self.assertEqual(model.get_variables_by_names("other_var"), [])

        # Multiple vars
        self.assertGreater(len(model.get_variables_by_names("x")), 0)
        res = model.get_variables_by_names("x")
        for ind in range(model.n_x):
            self.assertTrue(is_equal(res[ind], model.x[ind]))

        # Some var that exists and is unique
        res = model.get_variables_by_names("x_1")
        self.assertEqual(len(res), 1)
        self.assertTrue(is_equal(model.x[1], res[0]))

        # find var with var_type
        model2 = self.dae_model.get_copy()
        p_with_name_starting_with_x = model2.create_parameter("x_ref")
        res = model2.get_variables_by_names("x", var_type="p")
        self.assertEqual(len(res), 1)
        self.assertTrue(is_equal(p_with_name_starting_with_x, res[0]))

        # find var with list of names
        res = model.get_variables_by_names(["x_1", "u_2"])
        self.assertEqual(len(res), 2)
        self.assertTrue(is_equal(model.x[1], res[0]))
        self.assertTrue(is_equal(model.u[2], res[1]))

    def test_has_variable(self):
        self.assertTrue(self.dae_model.has_variable(self.dae_model.x[0]))
        self.assertFalse(self.dae_model.has_variable(SX.sym("x_0")))

    def test_is_parametrized(self):
        model = self.dae_model.get_copy()
        self.assertFalse(model.is_parametrized())

        k = model.create_parameter("k")
        model.parametrize_control(model.u[0], -k * model.x[0], k)
        self.assertTrue(model.is_parametrized())


def test_replace_variable_wrong_size(model):
    original = model.x
    wrong_replacement = SX.sym("new_x_wrong", original.numel() + 3)
    with raises(ValueError):
        model.replace_variable(original, wrong_replacement)


def test_include_models():
    pass


def test_connect():
    pass


def test_all_sym_names():
    pass


def test_slice_yz_to_y_and_z():
    pass


def test_concat_y_and_z():
    pass


def test_put_values_in_all_sym_format():
    pass


def test_convert_from_time_to_tau():
    pass


def test_convert_expr_from_tau_to_time():
    pass


def test_merge():
    pass


def test_get_dae_system():
    pass


def test_simulate():
    pass


def test_simulate_step():
    pass


def test_simulate_interval():
    pass


def test__create_integrator():
    pass


def test__create_explicit_integrator():
    pass


def test_find_variables_indices_in_vector():
    pass


def test_linearize():
    pass


def test_convert_expr_from_time_to_tau():
    pass


def test_get_copy():
    pass


def test_get_deepcopy():
    pass


def test_find_algebraic_variable():
    pass


def test_find_equilibrium():
    pass


def test_include_equations_ode(empty_model: SystemModel):
    x = empty_model.create_state("x")
    ode = -x
    empty_model.include_equations(ode=ode)
    assert empty_model.ode.numel() == x.numel()
    assert is_equal(empty_model.ode, ode, 20)


def test_include_equations_ode_with_x(empty_model: SystemModel):
    x = empty_model.create_state("x")
    ode = -x
    empty_model.include_equations(ode=ode, x=x)
    assert empty_model.ode.numel() == x.numel()
    assert is_equal(empty_model.ode, ode, 20)


def test_include_equations_list(empty_model):
    x = empty_model.create_state("x", 2)
    u = empty_model.create_control("u")
    y = empty_model.create_algebraic_variable("y", 3)

    # test for list input
    ode = [-x - y[:1] + u]
    alg = [2 * x[0] - y[0] + u, 2 * x[1] - y[1] + u, 2 * x[0] - y[2] + u]

    empty_model.include_equations(ode=ode, alg=alg)

    assert is_equal(empty_model.ode, vertcat(*ode), 20)
    assert is_equal(empty_model.alg, vertcat(*alg), 20)


def test_include_equations_ode_multi_dim(empty_model):
    x = empty_model.create_state("x", 2)
    u = empty_model.create_control("u", 2)

    a = DM([[-1, -2], [5, -1]])
    b = DM([[1, 0], [0, 1]])
    ode = mtimes(a, x) + mtimes(b, u)
    empty_model.include_equations(ode=ode)

    assert empty_model.ode.shape == (2, 1)
    assert is_equal(empty_model.ode, ode)


def test_include_equations_der(empty_model):
    x = empty_model.create_state("x")
    u = empty_model.create_control("u")

    empty_model.include_equations(der(x) == -x + u)

    assert empty_model.ode.numel() == 1
    assert is_equal(empty_model.ode, -x + u, 20)
