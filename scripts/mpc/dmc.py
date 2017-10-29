from numbers import Number
from matplotlib import pyplot as plt
from casadi import mtimes, vertcat, DM

from yaocptool.optimization import QuadraticOptimizationProblem
from yaocptool.modelling import SystemModel


class DMC:
    def __init__(self, model, **kwargs):
        self.model = model

        self.N_U = 10
        self.N = 20
        self.Ts = 1.

        self.Q = DM(1.)
        self.R = DM(1.)
        self.y_ref = DM(0.)

        self.x_k = 1
        self.u_k = 0

        self._g_i = vertcat([])

        for (k, v) in kwargs.items():
            setattr(self, k, v)
        if not 'plant' in kwargs:
            self.plant = model

        self._initialize()

    def _initialize(self):
        self._create_g_i()
        self._create_g_matrix()

        if isinstance(self.y_ref, (Number, DM)):
            self.y_ref = vertcat(*[self.y_ref] * self.N)

        self._create_qp()

    def _get_free_response(self, x_0, u_0):
        x_list = []
        x_k = x_0
        for k in range(self.N):
            x_next = self.model.simulate(x_0=x_k, t_f=self.Ts, p=u_0)['xf']
            x_list.append(x_next)
            x_k = x_next
        return vertcat(*x_list)

    def _create_g_i(self):
        g = []
        for i in range(self.N):
            g_i = self.model.simulate(x_0=0., t_f=i * self.Ts, p=1.)['xf']
            g.append(g_i)
        self.g_i = vertcat(*g)

    def _create_g_matrix(self):
        G = DM.zeros(self.N, self.N_U)
        for col in range(self.N_U):
            G[col:, col] = self.g_i[:self.N - col]
        self.G = G

    def _create_qp(self):
        qp = QuadraticOptimizationProblem(name='dmc_qp')
        delta_u = qp.create_variable('delta_u', self.N_U)
        y = qp.create_variable('delta_u', self.N)

        f = qp.create_parameter('f', self.N)

        error = y - self.y_ref
        J = self.Q * mtimes(error.T, error) + self.R * mtimes(delta_u.T, delta_u)

        qp.set_objective(J)
        qp.include_equality(y - mtimes(self.G, delta_u) - f)
        self.qp = qp

    def solve_iteration(self, x_k, u_k):
        f = self._get_free_response(x_0=x_k, u_0=u_k)
        res = self.qp.solve(p=f)
        delta_u = res['x'][:self.N_U]
        return delta_u[0]

    def run(self, iterations=10, disturbance=0):
        if not isinstance(disturbance, list):
            disturbance = [disturbance] * iterations

        x_list = [self.x_k]
        u_list = []

        for k in range(iterations):
            delta_u = self.solve_iteration(x_k=self.x_k, u_k=self.x_k)
            self.u_k = self.u_k + delta_u
            self.x_k = self.plant.simulate(x_0=self.x_k, t_f=self.Ts, p=self.u_k + disturbance[k])['xf']
            u_list.append(self.u_k)
            x_list.append(self.x_k)

        return x_list, u_list


if __name__ == '__main__':
    # PARAMETERS
    Ts = 1
    iterations = 40

    model = SystemModel(n_x=1, n_u=1)
    x = model.x_sym
    u = model.u_sym
    model.include_system_equations(ode=[-x + u])

    plant = SystemModel(n_x=1, n_u=1)
    x = plant.x_sym
    u = plant.u_sym
    plant.include_system_equations(ode=[-1.5*x + u])

    dmc = DMC(model, plant=plant, x_k=1., u_k=1., R=10.)
    x_res, u_res = dmc.run(iterations)
    x_res2, u_res2 = dmc.run(iterations, disturbance=1.)
    x_res.extend(x_res2[1:])
    u_res.extend(u_res2)
    plt.step(range(iterations*2+1), x_res, where='post')
    plt.step(range(iterations*2), u_res, where='post')
    # plt.show()