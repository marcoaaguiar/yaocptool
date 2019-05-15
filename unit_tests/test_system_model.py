from unittest import TestCase

from casadi import SX, is_equal, vertcat, vec

from yaocptool.modelling import SystemModel


class TestSystemModel(TestCase):
    def setUp(self):
        # create ode system
        self.ode_model = SystemModel(name='ode_sys')
        x = self.ode_model.create_state('x', 3)
        u = self.ode_model.create_input('u', 3)
        self.ode_model.include_system_equations(ode=-x + u)

        self.dae_model = SystemModel(name='dae_sys')
        x = self.dae_model.create_state('x', 3)
        y = self.dae_model.create_algebraic_variable('y', 3)
        u = self.dae_model.create_input('u', 3)
        self.dae_model.include_system_equations(ode=-x + u, alg=y - x + u ** 2)

    def test_system_type(self):
        self.assertEqual(self.ode_model.system_type, 'ode')
        self.assertEqual(self.dae_model.system_type, 'dae')

    def test_n_x(self):
        self.assertEqual(self.ode_model.n_x, 3)
        self.assertEqual(self.dae_model.n_x, 3)

    def test_n_y(self):
        self.assertEqual(self.ode_model.n_y, 0)
        self.assertEqual(self.dae_model.n_y, 3)

    def test_n_u(self):
        self.assertEqual(self.ode_model.n_u, 3)
        self.assertEqual(self.dae_model.n_u, 3)

    def test_n_p(self):
        self.assertEqual(self.ode_model.n_p, 0)
        self.assertEqual(self.dae_model.n_p, 0)

        # with p
        self.ode_model.p_sym = SX.sym('p', 4)
        self.dae_model.p_sym = SX.sym('p', 4)

        self.assertEqual(self.ode_model.n_p, 4)
        self.assertEqual(self.dae_model.n_p, 4)

    def test_n_theta(self):
        self.assertEqual(self.ode_model.n_theta, 0)
        self.assertEqual(self.dae_model.n_theta, 0)

        # with theta
        self.ode_model.theta_sym = SX.sym('theta', 4)
        self.dae_model.theta_sym = SX.sym('theta', 4)

        self.assertEqual(self.ode_model.n_theta, 4)
        self.assertEqual(self.dae_model.n_theta, 4)

    def test_n_u_par(self):
        self.assertEqual(self.ode_model.n_u_par, 3)
        self.assertEqual(self.dae_model.n_u_par, 3)

    def test_x_sys_sym(self):
        self.assertTrue(is_equal(self.ode_model.x_sys_sym, self.ode_model.x_sym))
        self.assertTrue(is_equal(self.dae_model.x_sys_sym, self.dae_model.x_sym))

        # with adjoints
        ode_x = self.ode_model.x[:]
        dae_x = self.dae_model.x[:]
        self.ode_model.create_state('lamb', self.ode_model.n_x)
        self.ode_model.has_adjoint_variables = True

        self.dae_model.create_state('lamb', self.dae_model.n_x)
        self.dae_model.has_adjoint_variables = True

        self.assertTrue(is_equal(self.ode_model.x_sys_sym, ode_x))
        self.assertTrue(is_equal(self.dae_model.x_sys_sym, dae_x))

    def test_lamb_sym(self):
        self.assertEqual(self.ode_model.lamb_sym.numel(), 0)
        self.assertEqual(self.dae_model.lamb_sym.numel(), 0)

        # with adjoints
        ode_lamb = self.ode_model.create_state('lamb', self.ode_model.n_x)
        self.ode_model.has_adjoint_variables = True

        dae_lamb = self.dae_model.create_state('lamb', self.dae_model.n_x)
        self.dae_model.has_adjoint_variables = True

        self.assertTrue(is_equal(self.ode_model.lamb_sym, ode_lamb))
        self.assertTrue(is_equal(self.dae_model.lamb_sym, dae_lamb))

    def test_all_sym(self):
        for model in [self.ode_model, self.dae_model]:
            answer = [model.t_sym, model.x_sym, model.y_sym, model.p_sym,
                      model.theta_sym, model.u_par]
            self.assertEqual(len(model.all_sym), len(answer))

            for index in range(len(model.all_sym)):
                self.assertTrue(is_equal(model.all_sym[index], answer[index]))

    def test_x(self):
        self.assertTrue(is_equal(self.ode_model.x, self.ode_model.x_sym))
        self.assertTrue(is_equal(self.dae_model.x, self.dae_model.x_sym))

    def test_x_setter(self):
        new_x = SX.sym('x', 2)
        self.ode_model.x = new_x
        self.dae_model.x = new_x

        self.assertTrue(is_equal(self.ode_model.x_sym, new_x))
        self.assertTrue(is_equal(self.ode_model.x, new_x))
        self.assertTrue(is_equal(self.dae_model.x_sym, new_x))
        self.assertTrue(is_equal(self.dae_model.x, new_x))

    def test_y(self):
        self.assertTrue(is_equal(self.ode_model.y, self.ode_model.y_sym))
        self.assertTrue(is_equal(self.dae_model.y, self.dae_model.y_sym))

    def test_y_setter(self):
        new_y = SX.sym('y', 2)
        self.ode_model.y = new_y
        self.dae_model.y = new_y

        self.assertTrue(is_equal(self.ode_model.y_sym, new_y))
        self.assertTrue(is_equal(self.ode_model.y, new_y))
        self.assertTrue(is_equal(self.dae_model.y_sym, new_y))
        self.assertTrue(is_equal(self.dae_model.y, new_y))

    def test_u(self):
        self.assertTrue(is_equal(self.ode_model.u, self.ode_model.u_sym))
        self.assertTrue(is_equal(self.dae_model.u, self.dae_model.u_sym))

    def test_u_setter(self):
        new_u = SX.sym('u', 2)
        self.ode_model.u = new_u
        self.dae_model.u = new_u

        self.assertTrue(is_equal(self.ode_model.u_sym, new_u))
        self.assertTrue(is_equal(self.ode_model.u, new_u))
        self.assertTrue(is_equal(self.dae_model.u_sym, new_u))
        self.assertTrue(is_equal(self.dae_model.u, new_u))

    def test_p(self):
        self.assertTrue(is_equal(self.ode_model.p, self.ode_model.p_sym))
        self.assertTrue(is_equal(self.dae_model.p, self.dae_model.p_sym))

    def test_p_setter(self):
        new_p = SX.sym('p', 2)
        self.ode_model.p = new_p
        self.dae_model.p = new_p

        self.assertTrue(is_equal(self.ode_model.p_sym, new_p))
        self.assertTrue(is_equal(self.ode_model.p, new_p))
        self.assertTrue(is_equal(self.dae_model.p_sym, new_p))
        self.assertTrue(is_equal(self.dae_model.p, new_p))

    def test_theta(self):
        self.assertTrue(is_equal(self.ode_model.theta, self.ode_model.theta_sym))
        self.assertTrue(is_equal(self.dae_model.theta, self.dae_model.theta_sym))

    def test_theta_setter(self):
        new_theta = SX.sym('theta', 2)
        self.ode_model.theta = new_theta
        self.dae_model.theta = new_theta

        self.assertTrue(is_equal(self.ode_model.theta_sym, new_theta))
        self.assertTrue(is_equal(self.ode_model.theta, new_theta))
        self.assertTrue(is_equal(self.dae_model.theta_sym, new_theta))
        self.assertTrue(is_equal(self.dae_model.theta, new_theta))

    def test_t(self):
        self.assertTrue(is_equal(self.ode_model.t, self.ode_model.t_sym))
        self.assertTrue(is_equal(self.dae_model.t, self.dae_model.t_sym))

    def test_t_setter(self):
        new_t = SX.sym('t')
        self.ode_model.t = new_t
        self.dae_model.t = new_t

        self.assertTrue(is_equal(self.ode_model.t_sym, new_t))
        self.assertTrue(is_equal(self.ode_model.t, new_t))
        self.assertTrue(is_equal(self.dae_model.t_sym, new_t))
        self.assertTrue(is_equal(self.dae_model.t, new_t))

    def test_tau(self):
        self.assertTrue(is_equal(self.ode_model.tau, self.ode_model.tau_sym))
        self.assertTrue(is_equal(self.dae_model.tau, self.dae_model.tau_sym))

    def test_tau_setter(self):
        new_tau = SX.sym('tau')
        self.ode_model.tau = new_tau
        self.dae_model.tau = new_tau

        self.assertTrue(is_equal(self.ode_model.tau_sym, new_tau))
        self.assertTrue(is_equal(self.ode_model.tau, new_tau))
        self.assertTrue(is_equal(self.dae_model.tau_sym, new_tau))
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
            model.create_parameter('par', 10)
            for ind in range(model.n_p):
                self.assertEqual(model.p[ind].name(), model.p_names[ind])

    def test_theta_names(self):
        for model in [self.ode_model, self.dae_model]:
            model.create_theta('theta', 10)
            for ind in range(model.n_theta):
                self.assertEqual(model.theta[ind].name(), model.theta_names[ind])

    def test_print_variables(self):
        for model in [self.ode_model, self.dae_model]:
            model.print_variables()

    def test_create_state(self):
        n_x_initial = self.ode_model.n_x
        n_new_x = 4
        x = self.ode_model.create_state('x', n_new_x)
        self.assertEqual(self.ode_model.n_x, n_x_initial + n_new_x)
        self.assertTrue(is_equal(self.ode_model.x[-n_new_x:], x))

    def test_create_algebraic_variable(self):
        n_y_initial = self.ode_model.n_y
        n_new_y = 4
        y = self.ode_model.create_algebraic_variable('y', n_new_y)
        self.assertEqual(self.ode_model.n_y, n_y_initial + n_new_y)
        self.assertTrue(is_equal(self.ode_model.y[-n_new_y:], y))

    def test_create_control(self):
        n_u_initial = self.ode_model.n_u
        n_new_u = 4
        u = self.ode_model.create_control('u', n_new_u)
        self.assertEqual(self.ode_model.n_u, n_u_initial + n_new_u)
        self.assertTrue(is_equal(self.ode_model.u[-n_new_u:], u))

    def test_create_input(self):
        n_u_initial = self.ode_model.n_u
        n_new_u = 4
        u = self.ode_model.create_input('u', n_new_u)
        self.assertEqual(self.ode_model.n_u, n_u_initial + n_new_u)
        self.assertTrue(is_equal(self.ode_model.u[-n_new_u:], u))

    def test_create_parameter(self):
        n_p_initial = self.ode_model.n_p
        n_new_p = 4
        p = self.ode_model.create_parameter('p', n_new_p)
        self.assertEqual(self.ode_model.n_p, n_p_initial + n_new_p)
        self.assertTrue(is_equal(self.ode_model.p[-n_new_p:], p))

    def test_create_theta(self):
        n_theta_initial = self.ode_model.n_theta
        n_new_theta = 4
        theta = self.ode_model.create_theta('theta', n_new_theta)
        self.assertEqual(self.ode_model.n_theta, n_theta_initial + n_new_theta)
        self.assertTrue(is_equal(self.ode_model.theta[-n_new_theta:], theta))

    def test_include_system_equations(self):
        self.ode_model.ode = vertcat([])

        ode = -self.ode_model.x + self.ode_model.u
        self.ode_model.include_system_equations(ode=ode)
        self.assertTrue(is_equal(self.ode_model.ode, ode, 20))

        # test for the dae system
        self.dae_model.ode = vertcat([])

        ode = -self.dae_model.x - self.dae_model.y + self.dae_model.u
        self.dae_model.include_system_equations(ode=ode)
        self.assertTrue(is_equal(self.dae_model.ode, ode, 20))

        self.dae_model.alg = vertcat([])

        alg = -self.dae_model.x - self.dae_model.y + self.dae_model.u
        self.dae_model.include_system_equations(alg=alg)
        self.assertTrue(is_equal(self.dae_model.alg, alg, 20))

        # test for list input
        self.dae_model.ode = vertcat([])
        self.dae_model.alg = vertcat([])

        ode = [-self.dae_model.x - self.dae_model.y + self.dae_model.u]
        alg = [-self.dae_model.x - self.dae_model.y + self.dae_model.u]
        self.dae_model.include_system_equations(ode=ode, alg=alg)
        self.assertTrue(is_equal(self.dae_model.ode, vertcat(*ode), 20))
        self.assertTrue(is_equal(self.dae_model.alg, vertcat(*alg), 20))

    def test_include_equations(self):
        self.ode_model.ode = vertcat([])

        ode = -self.ode_model.x + self.ode_model.u
        self.ode_model.include_equations(ode=ode)
        self.assertTrue(is_equal(self.ode_model.ode, ode, 20))

        # test for the dae system
        self.dae_model.ode = vertcat([])

        ode = -self.dae_model.x - self.dae_model.y + self.dae_model.u
        self.dae_model.include_equations(ode=ode)
        self.assertTrue(is_equal(self.dae_model.ode, ode, 20))

        self.dae_model.alg = vertcat([])

        alg = -self.dae_model.x - self.dae_model.y + self.dae_model.u
        self.dae_model.include_equations(alg=alg)
        self.assertTrue(is_equal(self.dae_model.alg, alg, 20))

        # test for list input
        self.dae_model.ode = vertcat([])
        self.dae_model.alg = vertcat([])

        ode = [-self.dae_model.x - self.dae_model.y + self.dae_model.u]
        alg = [-self.dae_model.x - self.dae_model.y + self.dae_model.u]
        self.dae_model.include_equations(ode=ode, alg=alg)
        self.assertTrue(is_equal(self.dae_model.ode, vertcat(*ode), 20))
        self.assertTrue(is_equal(self.dae_model.alg, vertcat(*alg), 20))

    def test_include_state(self):
        new_x_1 = SX.sym('new_x')
        new_x_2 = SX.sym('new_x_2', 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_x = model.n_x
            new_x_0_sym_1 = model.include_state(new_x_1)

            self.assertEqual(model.n_x, model_n_x + 1, )  # Number of state variables has increased
            self.assertEqual(model.x_0_sym.numel(), model_n_x + 1)  # Num. of initial cond for the state has increased
            self.assertEqual(new_x_0_sym_1.numel(), new_x_1.numel())  # The returned initial cond var == size added
            self.assertTrue(is_equal(model.x_sym[-1], new_x_1))  # The added var is the in the x_sym

            new_x_0_sym_2 = model.include_state(new_x_2)  # Number of state variables has increased
            self.assertEqual(model.n_x, model_n_x + 1 + 2)
            self.assertEqual(model.x_0_sym.numel(),
                             model_n_x + 1 + 2)  # Num. of initial cond for the state has increased
            self.assertEqual(new_x_0_sym_2.numel(), new_x_2.numel())  # The returned initial cond var == size added
            self.assertTrue(is_equal(model.x_sym[-3], new_x_1))
            self.assertTrue(is_equal(model.x_sym[-2:], new_x_2))

    def test_include_algebraic(self):
        new_y_1 = SX.sym('new_y')
        new_y_2 = SX.sym('new_y_2', 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_y = model.n_y
            model.include_algebraic(new_y_1, alg=new_y_1 - model.x[0])

            self.assertEqual(model.n_y, model_n_y + 1, )
            self.assertTrue(is_equal(model.y_sym[-1], new_y_1))
            self.assertTrue(is_equal(model.alg[-1], new_y_1 - model.x[0], 10))

            model.include_algebraic(new_y_2)
            self.assertEqual(model.n_y, model_n_y + 1 + 2)
            self.assertTrue(is_equal(model.y_sym[-3], new_y_1))
            self.assertTrue(is_equal(model.y_sym[-2:], new_y_2))

    def test_include_control(self):
        new_u_1 = SX.sym('new_u')
        new_u_2 = SX.sym('new_u_2', 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_u = model.n_u
            model.include_control(new_u_1)

            self.assertEqual(model.n_u, model_n_u + 1, )
            self.assertTrue(is_equal(model.u_sym[-1], new_u_1))

            model.include_control(new_u_2)
            self.assertEqual(model.n_u, model_n_u + 1 + 2)
            self.assertTrue(is_equal(model.u_sym[-3], new_u_1))
            self.assertTrue(is_equal(model.u_sym[-2:], new_u_2))

    def test_include_parameter(self):
        new_p_1 = SX.sym('new_p')
        new_p_2 = SX.sym('new_p_2', 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_p = model.n_p
            model.include_parameter(new_p_1)

            self.assertEqual(model.n_p, model_n_p + 1)
            self.assertTrue(is_equal(model.p_sym[-1], new_p_1))

            model.include_parameter(new_p_2)
            self.assertEqual(model.n_p, model_n_p + 1 + 2)
            self.assertTrue(is_equal(model.p_sym[-3], new_p_1))
            self.assertTrue(is_equal(model.p_sym[-2:], new_p_2))

    def test_include_theta(self):
        new_theta_1 = SX.sym('new_theta')
        new_theta_2 = SX.sym('new_theta_2', 2)
        for model in [self.ode_model, self.dae_model]:
            model_n_theta = model.n_theta
            model.include_theta(new_theta_1)

            self.assertEqual(model.n_theta, model_n_theta + 1)
            self.assertTrue(is_equal(model.theta_sym[-1], new_theta_1))

            model.include_theta(new_theta_2)
            self.assertEqual(model.n_theta, model_n_theta + 1 + 2)
            self.assertTrue(is_equal(model.theta_sym[-3], new_theta_1))
            self.assertTrue(is_equal(model.theta_sym[-2:], new_theta_2))

    def test_replace_variable(self):
        # replace x
        model = self.dae_model.get_copy()
        original = model.x
        replacement = SX.sym('new_x', original.numel())

        model.replace_variable(original, replacement)

        self.assertTrue(is_equal(model.ode, -replacement + model.u, 30))
        self.assertTrue(is_equal(model.alg, model.y - replacement + model.u ** 2, 30))

        # replace a u_par
        model = self.dae_model.get_copy()

        new_u_par = SX.sym('new_u', model.n_u)
        new_u_expr = new_u_par

        model.replace_variable(model.u_par, new_u_par)

        self.assertTrue(is_equal(model.ode, -model.x + new_u_expr, 30))
        self.assertTrue(is_equal(model.alg, model.y - model.x + new_u_expr ** 2, 30))
        self.assertTrue(is_equal(model.u_par, new_u_par, 30))
        self.assertTrue(is_equal(model.u_expr, new_u_expr, 30))

        # Wrong size
        wrong_replacement = SX.sym('new_x_wrong', original.numel() + 3)
        self.assertRaises(ValueError, self.dae_model.replace_variable, original, wrong_replacement)

    def test_remove_state(self):
        model = self.dae_model.get_copy()
        ind_to_remove = 1
        to_remove = model.x[ind_to_remove]
        to_remove_eq = model.ode[ind_to_remove]

        n_x_original = model.n_x
        n_ode_original = model.ode.numel()

        model.remove_state(to_remove, eq=to_remove_eq)

        # removed var
        self.assertEqual(model.n_x, n_x_original - 1)
        for ind in range(model.n_x):
            self.assertFalse(is_equal(model.x[ind], to_remove))

        # removed ode
        self.assertEqual(model.ode.numel(), n_ode_original - 1)
        for ind in range(model.ode.numel()):
            self.assertFalse(is_equal(model.ode[ind], to_remove_eq))

    def test_remove_algebraic(self):
        model = self.dae_model.get_copy()
        ind_to_remove = 1
        to_remove = model.y[ind_to_remove]
        to_remove_eq = model.alg[ind_to_remove]

        n_y_original = model.n_y
        n_alg_original = model.alg.numel()

        model.remove_algebraic(to_remove, eq=to_remove_eq)

        # removed var
        self.assertEqual(model.n_y, n_y_original - 1)
        for ind in range(model.n_y):
            self.assertFalse(is_equal(model.y[ind], to_remove))

        # removed alg
        self.assertEqual(model.alg.numel(), n_alg_original - 1)
        for ind in range(model.alg.numel()):
            self.assertFalse(is_equal(model.alg[ind], to_remove_eq))

    def test_remove_control(self):
        model = self.dae_model.get_copy()
        ind_to_remove = 1
        to_remove = model.u[ind_to_remove]
        n_u_original = model.n_u

        model.remove_control(to_remove)

        # removed var
        self.assertEqual(model.n_u, n_u_original - 1)
        for ind in range(model.n_u):
            self.assertFalse(is_equal(model.u[ind], to_remove))

    def test_remove_parameter(self):
        model = self.dae_model.get_copy()
        model.create_parameter('par', 3)
        ind_to_remove = 1
        to_remove = model.p[ind_to_remove]
        n_p_original = model.n_p

        model.remove_parameter(to_remove)

        # removed var
        self.assertEqual(model.n_p, n_p_original - 1)
        for ind in range(model.n_p):
            self.assertFalse(is_equal(model.p[ind], to_remove))

    def test_remove_theta(self):
        model = self.dae_model.get_copy()
        model.create_theta('par', 3)
        ind_to_remove = 1
        to_remove = model.theta[ind_to_remove]
        n_theta_original = model.n_theta

        model.remove_theta(to_remove)

        # removed var
        self.assertEqual(model.n_theta, n_theta_original - 1)
        for ind in range(model.n_theta):
            self.assertFalse(is_equal(model.theta[ind], to_remove))

    def test_get_variable_by_name(self):
        model = self.dae_model.get_copy()

        # Variable does not exist:
        self.assertRaises(ValueError, model.get_variable_by_name, 'other_var')

        # Multiple vars
        self.assertRaises(ValueError, model.get_variable_by_name, 'x')

        # Some var that exists and is unique
        var = model.get_variable_by_name('x_1')
        self.assertTrue(is_equal(model.x[1], var))

    def test_get_variables_by_names(self):
        model = self.dae_model.get_copy()

        # Variable does not exist
        self.assertEqual(model.get_variables_by_names('other_var'), [])

        # Multiple vars
        self.assertGreater(len(model.get_variables_by_names('x')), 0)
        res = model.get_variables_by_names('x')
        for ind in range(model.n_x):
            self.assertTrue(is_equal(res[ind], model.x[ind]))

        # Some var that exists and is unique
        res = model.get_variables_by_names('x_1')
        self.assertEqual(len(res), 1)
        self.assertTrue(is_equal(model.x[1], res[0]))

        # find var with var_type
        model2 = self.dae_model.get_copy()
        p_with_name_starting_with_x = model2.create_parameter('x_ref')
        res = model2.get_variables_by_names('x', var_type='p')
        self.assertEqual(len(res), 1)
        self.assertTrue(is_equal(p_with_name_starting_with_x, res[0]))

        # find var with list of names
        res = model.get_variables_by_names(['x_1', 'u_2'])
        self.assertEqual(len(res), 2)
        self.assertTrue(is_equal(model.x[1], res[0]))
        self.assertTrue(is_equal(model.u[2], res[1]))

    def test_has_variable(self):
        self.assertTrue(self.dae_model.has_variable(self.dae_model.x[0]))
        self.assertFalse(self.dae_model.has_variable(SX.sym('x_0')))

    def test_control_is_parametrized(self):
        model = self.dae_model.get_copy()
        self.assertFalse(model.control_is_parametrized(model.u[0]))

        # error multiple controls are passed
        self.assertRaises(ValueError, model.control_is_parametrized, model.u)

        k = model.create_parameter('k')
        model.parametrize_control(model.u[0], -k * model.x[0], k)
        self.assertTrue(model.control_is_parametrized(model.u[0]))

    def test_is_parametrized(self):
        model = self.dae_model.get_copy()
        self.assertFalse(model.is_parametrized())

        k = model.create_parameter('k')
        model.parametrize_control(model.u[0], -k * model.x[0], k)
        self.assertTrue(model.is_parametrized())

    def test_parametrize_control(self):
        model = self.dae_model.get_copy()

        # wrong size for expr
        k = SX.sym('k', 2)
        self.assertRaises(ValueError, model.parametrize_control, model.u, k * model.t, k)

        # Test parametrize by a time dependent polynomial
        model = self.dae_model.get_copy()
        u_par = SX.sym('u_par', 3, 2)
        u_expr = model.tau * u_par[:, 0] + (1 - model.tau) * u_par[:, 1]
        model.parametrize_control(model.u, u_expr, vec(u_par))
        self.assertTrue(is_equal(model.u_par, vec(u_par)))
        self.assertTrue(is_equal(model.u_expr, u_expr, 30))
        for ind in range(model.n_u):
            self.assertTrue(is_equal(model._parametrized_controls[ind], model.u[ind]))

        # Test for list inputs, parametrize by a time dependent polynomial
        model = self.dae_model.get_copy()
        u_par = SX.sym('u_par', 3, 2)
        u_expr = model.tau * u_par[:, 0] + (1 - model.tau) * u_par[:, 1]
        model.parametrize_control([model.u[ind] for ind in range(model.n_u)],
                                  [u_expr[ind] for ind in range(model.n_u)],
                                  [vec(u_par)[ind] for ind in range(u_par.numel())])

        self.assertTrue(is_equal(model.u_par, vec(u_par)))
        self.assertTrue(is_equal(model.u_expr, u_expr, 30))
        for ind in range(model.n_u):
            self.assertTrue(is_equal(model._parametrized_controls[ind], model.u[ind]))

        # test parametrize a control already parametrized
        model = self.dae_model.get_copy()
        k = SX.sym('k')
        model.parametrize_control(model.u[0], -k * model.x[0], k)
        self.assertRaises(ValueError, model.parametrize_control, model.u[0], k * model.t, k)

    def test_include_models(self):
        pass

    def test_connect(self):
        pass

    def test_all_sym_names(self):
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

    def test_convert_expr_from_time_to_tau(self):
        pass

    def test_get_copy(self):
        pass

    def test_get_deepcopy(self):
        pass

    def test_find_algebraic_variable(self):
        pass

    def test_find_equilibrium(self):
        pass
