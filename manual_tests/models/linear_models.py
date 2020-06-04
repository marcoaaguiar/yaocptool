from casadi import mtimes, DM

from yaocptool.modelling import SystemModel, OptimalControlProblem


class MIMO2x2(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, **kwargs)

        a = DM([[-1, -2], [5, -1]])
        b = DM([[1, 0], [0, 1]])

        x = self.create_state("x", 2)
        u = self.create_control("u", 2)

        self.include_equations(ode=mtimes(a, x) + mtimes(b, u))


class MIMO2x2DAE(SystemModel):
    def __init__(self, name="dae_system", **kwargs):
        SystemModel.__init__(self, name=name, model_name_as_prefix=True, **kwargs)

        x = self.create_state("x", 2)
        y = self.create_algebraic_variable("y", 2)
        u = self.create_control("u", 2)
        a = self.create_parameter("a")

        ode = [-a * x[0] + y[0], -x[1] + y[1] + u[0]]
        alg = [-y[0] - x[1] + u[1], -y[1] - x[0]]

        self.include_equations(ode=ode, alg=alg)


class StabilizationMIMO2x2(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        OptimalControlProblem.__init__(
            self,
            model,
            name=model.name + "_stabilization",
            obj={"Q": DM.eye(2), "R": DM.eye(2)},
            x_0=[1, 1],
            **kwargs
        )


class StabilizationMIMO2x2WithInequality(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        OptimalControlProblem.__init__(
            self,
            model,
            name=model.name + "_stabilization",
            obj={"Q": DM.eye(2), "R": DM.eye(2)},
            x_0=[1, 1],
            **kwargs
        )
        # self.include_time_inequality(+model.u + model.x[0], when='end')

