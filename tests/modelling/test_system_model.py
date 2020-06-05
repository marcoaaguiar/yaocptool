from unittest import TestCase

from casadi import SX, is_equal, vertcat, vec, DM

from yaocptool.modelling import SystemModel
from pytest import raises
from yaocptool.modelling.utils import der
from casadi.casadi import depends_on, mtimes


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
        self.dae_model.include_equations(ode=-x + u, alg=y - x + u**2)

    def test_system_type(self):
        assert self.ode_model.system_type == "ode"
        assert self.dae_model.system_type, "dae"

    def test_n_y(self):
        assert self.ode_model.n_y == 0
        assert self.dae_model.n_y == 3

    def test_n_u(self):
        assert self.ode_model.n_u == 3
        assert self.dae_model.n_u == 3

    def test_n_p(self):
        self.assertEqual(self.ode_model.n_p, 0)
        self.assertEqual(self.dae_model.n_p, 0)

        # with p
        self.ode_model.p = SX.sym("p", 4)
        self.dae_model.p = SX.sym("p", 4)

        self.assertEqual(self.ode_model.n_p, 4)
        self.assertEqual(self.dae_model.n_p, 4)

    def test_n_theta(self):
        self.assertEqual(self.ode_model.n_theta, 0)
        self.assertEqual(self.dae_model.n_theta, 0)

        # with theta
        self.ode_model.theta = SX.sym("theta", 4)
        self.dae_model.theta = SX.sym("theta", 4)

        self.assertEqual(self.ode_model.n_theta, 4)
        self.assertEqual(self.dae_model.n_theta, 4)

    def test_n_u_par(self):
        self.assertEqual(self.ode_model.n_u_par, 3)
        self.assertEqual(self.dae_model.n_u_par, 3)

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

    def test_p(self):
        self.assertTrue(is_equal(self.ode_model.p, self.ode_model.p))
        self.assertTrue(is_equal(self.dae_model.p, self.dae_model.p))

    def test_p_setter(self):
        new_p = SX.sym("p", 2)
        self.ode_model.p = new_p
        self.dae_model.p = new_p

        self.assertTrue(is_equal(self.ode_model.p, new_p))
        self.assertTrue(is_equal(self.dae_model.p, new_p))

    def test_theta(self):
        self.assertTrue(is_equal(self.ode_model.theta, self.ode_model.theta))
        self.assertTrue(is_equal(self.dae_model.theta, self.dae_model.theta))

    def test_theta_setter(self):
        new_theta = SX.sym("theta", 2)
        self.ode_model.theta = new_theta
        self.dae_model.theta = new_theta

        self.assertTrue(is_equal(self.ode_model.theta, new_theta))
        self.assertTrue(is_equal(self.dae_model.theta, new_theta))

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

    def test_p_names(self):
        for model in [self.ode_model, self.dae_model]:
            model.create_parameter("par", 10)
            for ind in range(model.n_p):
                self.assertEqual(model.p[ind].name(), model.p_names[ind])

    def test_theta_names(self):
        for model in [self.ode_model, self.dae_model]:
            model.create_theta("theta", 10)
            for ind in range(model.n_theta):
                self.assertEqual(model.theta[ind].name(),
                                 model.theta_names[ind])

    def test_print_variables(self):
        for model in [self.ode_model, self.dae_model]:
            model.print_variables()

    def test_create_algebraic_variable(self):
        n_y_initial = self.ode_model.n_y
        n_new_y = 4
        y = self.ode_model.create_algebraic_variable("y", n_new_y)
        self.assertEqual(self.ode_model.n_y, n_y_initial + n_new_y)
        self.assertTrue(is_equal(self.ode_model.y[-n_new_y:], y))

    def test_create_control(self):
        n_u_initial = self.ode_model.n_u
        n_new_u = 4
        u = self.ode_model.create_control("u", n_new_u)
        self.assertEqual(self.ode_model.n_u, n_u_initial + n_new_u)
        self.assertTrue(is_equal(self.ode_model.u[-n_new_u:], u))

    def test_create_input(self):
        n_u_initial = self.ode_model.n_u
        n_new_u = 4
        u = self.ode_model.create_input("u", n_new_u)
        self.assertEqual(self.ode_model.n_u, n_u_initial + n_new_u)
        self.assertTrue(is_equal(self.ode_model.u[-n_new_u:], u))

    def test_create_parameter(self):
        n_p_initial = self.ode_model.n_p
        n_new_p = 4
        p = self.ode_model.create_parameter("p", n_new_p)
        self.assertEqual(self.ode_model.n_p, n_p_initial + n_new_p)
        self.assertTrue(is_equal(self.ode_model.p[-n_new_p:], p))

    def test_create_theta(self):
        n_theta_initial = self.ode_model.n_theta
        n_new_theta = 4
        theta = self.ode_model.create_theta("theta", n_new_theta)
        self.assertEqual(self.ode_model.n_theta, n_theta_initial + n_new_theta)
        self.assertTrue(is_equal(self.ode_model.theta[-n_new_theta:], theta))

    def test_include_algebraic(self):
        new_y_1 = SX.sym("new_y")
        new_y_2 = SX.sym("new_y_2", 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_y = model.n_y
            model.include_algebraic(new_y_1, alg=new_y_1 - model.x[0])

            self.assertEqual(
                model.n_y,
                model_n_y + 1,
            )
            self.assertTrue(is_equal(model.y[-1], new_y_1))
            self.assertTrue(is_equal(model.alg[-1], new_y_1 - model.x[0], 10))

            model.include_algebraic(new_y_2)
            self.assertEqual(model.n_y, model_n_y + 1 + 2)
            self.assertTrue(is_equal(model.y[-3], new_y_1))
            self.assertTrue(is_equal(model.y[-2:], new_y_2))

    def test_include_control(self):
        new_u_1 = SX.sym("new_u")
        new_u_2 = SX.sym("new_u_2", 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_u = model.n_u
            model.include_control(new_u_1)

            self.assertEqual(
                model.n_u,
                model_n_u + 1,
            )
            self.assertTrue(is_equal(model.u[-1], new_u_1))

            model.include_control(new_u_2)
            self.assertEqual(model.n_u, model_n_u + 1 + 2)
            self.assertTrue(is_equal(model.u[-3], new_u_1))
            self.assertTrue(is_equal(model.u[-2:], new_u_2))

    def test_include_parameter(self):
        new_p_1 = SX.sym("new_p")
        new_p_2 = SX.sym("new_p_2", 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_p = model.n_p
            model.include_parameter(new_p_1)

            self.assertEqual(model.n_p, model_n_p + 1)
            self.assertTrue(is_equal(model.p[-1], new_p_1))

            model.include_parameter(new_p_2)
            self.assertEqual(model.n_p, model_n_p + 1 + 2)
            self.assertTrue(is_equal(model.p[-3], new_p_1))
            self.assertTrue(is_equal(model.p[-2:], new_p_2))

    def test_include_theta(self):
        new_theta_1 = SX.sym("new_theta")
        new_theta_2 = SX.sym("new_theta_2", 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_theta = model.n_theta
            model.include_theta(new_theta_1)

            self.assertEqual(model.n_theta, model_n_theta + 1)
            self.assertTrue(is_equal(model.theta[-1], new_theta_1))

            model.include_theta(new_theta_2)
            self.assertEqual(model.n_theta, model_n_theta + 1 + 2)
            self.assertTrue(is_equal(model.theta[-3], new_theta_1))
            self.assertTrue(is_equal(model.theta[-2:], new_theta_2))

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

    def test_control_is_parametrized(self):
        model = self.dae_model.get_copy()
        self.assertFalse(model.control_is_parametrized(model.u[0]))

        # error multiple controls are passed
        self.assertRaises(ValueError, model.control_is_parametrized, model.u)

        k = model.create_parameter("k")
        model.parametrize_control(model.u[0], -k * model.x[0], k)
        self.assertTrue(model.control_is_parametrized(model.u[0]))

    def test_is_parametrized(self):
        model = self.dae_model.get_copy()
        self.assertFalse(model.is_parametrized())

        k = model.create_parameter("k")
        model.parametrize_control(model.u[0], -k * model.x[0], k)
        self.assertTrue(model.is_parametrized())

    def test_parametrize_control(self):
        model = self.dae_model.get_copy()

        # wrong size for expr
        k = SX.sym("k", 2)
        self.assertRaises(ValueError, model.parametrize_control, model.u,
                          k * model.t, k)

        # Test parametrize by a time dependent polynomial
        model = self.dae_model.get_copy()
        u_par = SX.sym("u_par", 3, 2)
        u_expr = model.tau * u_par[:, 0] + (1 - model.tau) * u_par[:, 1]
        model.parametrize_control(model.u, u_expr, vec(u_par))
        self.assertTrue(is_equal(model.u_par, vec(u_par)))
        self.assertTrue(is_equal(model.u_expr, u_expr, 30))
        for ind in range(model.n_u):
            self.assertTrue(
                is_equal(model._parametrized_controls[ind], model.u[ind]))

        # Test for list inputs, parametrize by a time dependent polynomial
        model = self.dae_model.get_copy()
        u_par = SX.sym("u_par", 3, 2)
        u_expr = model.tau * u_par[:, 0] + (1 - model.tau) * u_par[:, 1]
        model.parametrize_control(
            [model.u[ind] for ind in range(model.n_u)],
            [u_expr[ind] for ind in range(model.n_u)],
            [vec(u_par)[ind] for ind in range(u_par.numel())],
        )

        self.assertTrue(is_equal(model.u_par, vec(u_par)))
        self.assertTrue(is_equal(model.u_expr, u_expr, 30))
        for ind in range(model.n_u):
            self.assertTrue(
                is_equal(model._parametrized_controls[ind], model.u[ind]))

        # test parametrize a control already parametrized
        model = self.dae_model.get_copy()
        k = SX.sym("k")
        model.parametrize_control(model.u[0], -k * model.x[0], k)
        self.assertRaises(ValueError, model.parametrize_control, model.u[0],
                          k * model.t, k)


def test_replace_variable_state(model: SystemModel):
    model.alg = vertcat(*[model.x])
    # replace x
    original = model.x
    replacement = SX.sym("new_x", original.numel())

    model.replace_variable(original, replacement)

    assert not depends_on(model.ode, original)
    assert depends_on(model.ode, replacement)
    assert not depends_on(model.alg, original)
    assert depends_on(model.alg, replacement)


def test_remove_algebraic(model):
    if model.n_y > 0:
        ind_to_remove = 0
        to_remove = model.y[ind_to_remove]
        to_remove_eq = model.alg[ind_to_remove]

        n_y_original = model.n_y
        n_alg_original = model.alg.numel()

        model.remove_algebraic(to_remove, eq=to_remove_eq)

        # removed var
        assert model.n_y == n_y_original - 1
        for ind in range(model.n_y):
            assert is_equal(model.y[ind], to_remove)

        # removed alg
        assert model.alg.numel() == n_alg_original - 1
        for ind in range(model.alg.numel()):
            assert is_equal(model.alg[ind], to_remove_eq)


def test_remove_control(model):
    if model.n_u > 0:
        ind_to_remove = 0
        to_remove = model.u[ind_to_remove]
        n_u_original = model.n_u

        model.remove_control(to_remove)

        # removed var
        assert model.n_u == n_u_original - 1
        for ind in range(model.n_u):
            assert not is_equal(model.u[ind], to_remove)


def test_remove_parameter(model):
    model.create_parameter("par", 3)
    ind_to_remove = 1
    to_remove = model.p[ind_to_remove]
    n_p_original = model.n_p

    model.remove_parameter(to_remove)

    # removed var
    assert model.n_p == n_p_original - 1
    for ind in range(model.n_p):
        assert not is_equal(model.p[ind], to_remove)


def test_remove_theta(model):
    model.create_theta("par", 3)
    ind_to_remove = 0
    to_remove = model.theta[ind_to_remove]
    n_theta_original = model.n_theta

    model.remove_theta(to_remove)

    # removed var
    assert model.n_theta == n_theta_original - 1
    for ind in range(model.n_theta):
        assert not is_equal(model.theta[ind], to_remove)


def test_replace_variable_u_par(model):
    # replace a u_par
    new_u_par = SX.sym("new_u", model.n_u)
    new_u_expr = new_u_par

    model.replace_variable(model.u_par, new_u_par)

    assert is_equal(model.ode, -model.x + new_u_expr, 30)


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
    x = empty_model.create_state('x')
    ode = -x
    empty_model.include_equations(ode=ode)
    assert empty_model.ode.numel() == x.numel()
    assert is_equal(empty_model.ode, ode, 20)


def test_include_equations_ode_with_x(empty_model: SystemModel):
    x = empty_model.create_state('x')
    ode = -x
    empty_model.include_equations(ode=ode, x=x)
    assert empty_model.ode.numel() == x.numel()
    assert is_equal(empty_model.ode, ode, 20)


def test_include_equations_alg(empty_model):
    y = empty_model.create_algebraic_variable('y')
    alg = -y

    empty_model.include_equations(alg=alg)
    assert is_equal(empty_model.alg, -y, 20)


def test_include_equations_list(empty_model):
    x = empty_model.create_state('x', 2)
    u = empty_model.create_control('u')
    y = empty_model.create_algebraic_variable('y', 3)

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
    ode = (mtimes(a, x) + mtimes(b, u))
    empty_model.include_equations(ode=ode)

    assert empty_model.ode.shape == (2, 1)
    assert is_equal(empty_model.ode, ode)


def test_include_equations_der(empty_model):
    x = empty_model.create_state('x')
    u = empty_model.create_control('u')

    empty_model.include_equations(der(x) == -x + u)

    assert empty_model.ode.numel() == 1
    assert is_equal(empty_model.ode, -x + u, 20)
