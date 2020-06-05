import pytest
from yaocptool.modelling import SystemModel
from yaocptool.modelling.mixins.state_mixin import StateMixin
from casadi.casadi import depends_on, is_equal, SX, vertcat


@pytest.fixture
def model():
    return StateMixin()


def test_n_x(model: StateMixin):
    assert model.n_x == model.x.numel()


def test_create_state(model: StateMixin):
    n_x_initial = model.n_x
    n_new_x = 4
    x = model.create_state("x", n_new_x)
    assert model.n_x == n_x_initial + n_new_x
    assert is_equal(model.x[-n_new_x:], x)


def test_remove_state(model: StateMixin):
    x = model.create_state('x', 5)
    model.include_ode_equation(ode=vertcat(
        *[i * x_i for i, x_i in enumerate(x.nz)]))

    ind_to_remove = 3
    to_remove = model.x[ind_to_remove]
    to_remove_eq = model.ode[ind_to_remove]

    n_x_original = model.n_x
    n_ode_original = model.ode.numel()

    model.remove_state(to_remove)

    # removed var
    assert model.n_x == n_x_original - 1
    for ind in range(model.n_x):
        assert not is_equal(model.x[ind], to_remove)

    # removed ode
    assert model.ode.numel() == n_ode_original - 1
    for ind in range(model.ode.numel()):
        assert not is_equal(model.ode[ind], to_remove_eq)


def test_include_state(model):
    new_x_1 = SX.sym("new_x")
    new_x_2 = SX.sym("new_x_2", 2)

    model_n_x = model.n_x
    new_x_0_1 = model.include_state(new_x_1)

    assert model.n_x == model_n_x + 1
    assert model.x_0.numel() == model_n_x + 1
    assert new_x_0_1.numel() == new_x_1.numel()
    assert is_equal(model.x[-1], new_x_1)

    model_n_x = model.n_x
    new_x_0_2 = model.include_state(new_x_2)

    assert model.n_x == model_n_x + 2
    assert model.x_0.numel(), model_n_x + 2
    assert new_x_0_2.numel() == new_x_2.numel()
    assert is_equal(model.x[-3], new_x_1)
    assert is_equal(model.x[-2:], new_x_2)
