from yaocptool.modelling import SystemModel, OptimalControlProblem


class VanDerPol(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, **kwargs)

        x_1 = self.create_state('x_0')
        x_2 = self.create_state('x_1')
        u = self.create_control('u')

        ode = [(1 - x_2 ** 2) * x_1 - x_2 + u,
               x_1]

        self.include_system_equations(ode)


class VanDerPolDAE(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, **kwargs)

        x_1 = self.create_state('x_0')
        x_2 = self.create_state('x_1')
        y = self.create_algebraic_variable('y')
        u = self.create_control('u')

        ode = [y + u,
               x_1]
        alg = [(1 - x_2 ** 2) * x_1 - x_2 - y]

        self.include_system_equations(ode, alg)


def get_model(name='dae_system'):
    model = VanDerPol(name=name, model_name_as_prefix=True)
    return model


class VanDerPolStabilization(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        OptimalControlProblem.__init__(self, model, name=model.name + '_stabilization', **kwargs)
        self.t_f = 10
        self.L = model.x[0] ** 2 + model.x[1] ** 2 + model.u ** 2
        self.x_0 = [0, 1]
