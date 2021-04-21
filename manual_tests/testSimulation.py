from yaocptool.modelling import SystemModel

# create model
model = SystemModel(name="dae_system")

x = model.create_state("x", 2)
y = model.create_algebraic_variable("y", 2)
u = model.create_control("u")
a = model.create_parameter("a")
b = model.create_theta("b")

model.include_equations(
    ode=[-a * x[0] + b * y[0], -x[1] + y[1] + u[0]],
    alg=[-y[0] - x[1] ** 2, -y[1] - x[0] ** 1],
)

x_0 = [1, 2]
sim_result = model.simulate(
    x_0, range(1, 10), 0.5, u=1.0, p=[1], theta=dict(zip(range(0, 9), [0] * 9))
)
# sim_result.plot([{'x': 'all'}, {'y': 'all'}, {'u': 'all'}])

# Include data at the end of the sim_result
copy_of_sim_result = sim_result.get_copy()
sim_result2 = model.simulate(
    x_0, range(20, 30), 19, u=1.0, p=[1], theta=dict(zip(range(0, 10), [0] * 10))
)
copy_of_sim_result.extend(sim_result2)
# copy_of_sim_result.plot([{'x': 'all'}, {'y': 'all'}, {'u': 'all'}])

sim_result3 = model.simulate(
    x_0, range(-20, -10), -21, u=1.0, p=[1], theta=dict(zip(range(0, 10), [0] * 10))
)
copy_of_sim_result.extend(sim_result3)
copy_of_sim_result.plot([{"x": ["x_0", 1]}, {"y": "all"}, {"u": "all"}])

# sim_result.plot([{'x': 'all'}, {'y': 'all'}, {'u': 'all'}])
