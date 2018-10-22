# yaocptool
(YAOCPTool) Yet Another Optimal Control Tool

## How to install

The package is not yet listed in PyPI (pip install), however it can be installed directly from GitHub using the following command:

For MASTER branch:
```commandline
pip install https://github.com/marcoaaguiar/yaocptool/archive/master.zip
```
For DEV branch:
```commandline
pip install https://github.com/marcoaaguiar/yaocptool/archive/dev.zip
```

If you are not installing using pip install, but cloning the repository you will need to install the following packages
```commandline
pip install casadi numpy scipy sobol_seq matplotlib
```


## Documentation

The documentation can be accessed [here!](https://marcoaaguiar.github.io/yaocptool/)
For PDF version click [here!](https://marcoaaguiar.github.io/yaocptool/build/latex/YAOCPTool.pdf)


## How to use
The objective of this tool is to make easier to use the state-of-the-art CasADi, at the same time allowing for researchers to propose new methods.

Trust me, it is easier!

### Creating a model

```python
from yaocptool.modelling import SystemModel

model = SystemModel(name="simple_model")
x = model.create_state("x") # vector of state variables
u = model.create_control("u") # vector of control variables

# Include the dynamic equation
ode = [-x + u]
model.include_system_equations(ode=ode)

# Print model information
print(model)
```

### Creating a Optimal Control Problem (OCP)

```python
from yaocptool.modelling import OptimalControlProblem

problem = OptimalControlProblem(model, x_0=[1], t_f=10, obj={"Q": 1, "R": 1})
```

### Creating a Solver for the OCP
```python
# Initialize a DirectMethod to solve the OCP using collocation
solution_method = DirectMethod(problem, finite_elements=20, discretization_scheme='collocation')

# Solve the problem and get the result
result = solution_method.solve()

# Make one plot with the element x[0] (the first state) and one plot with the control u[0]
result.plot([{'x':[0]}, {'u':[0]}])

```
