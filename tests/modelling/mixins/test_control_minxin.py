from casadi.casadi import is_equal, SX
from yaocptool.modelling.mixins.control_mixin import ControlMixin
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def model():
    return ControlMixin()


def test_include_control(model: ControlMixin):
    new_u_1 = SX.sym("new_u")
    new_u_2 = SX.sym("new_u_2", 2)

    model_n_u = model.n_u
    model.include_control(new_u_1)

    assert model.n_u == model_n_u + 1
    assert (is_equal(model.u[-1], new_u_1))

    model.include_control(new_u_2)
    assert model.n_u == model_n_u + 1 + 2
    assert is_equal(model.u[-3], new_u_1)
    assert is_equal(model.u[-2:], new_u_2)


def test_create_control(model: ControlMixin):
    n_u_initial = model.n_u
    n_new_u = 4
    u = model.create_control("u", n_new_u)
    assert model.n_u == n_u_initial + n_new_u
    assert is_equal(model.u[-n_new_u:], u)


def test_remove_control(model: ControlMixin):
    model.create_control('u', 4)
    ind_to_remove = 2
    to_remove = model.u[ind_to_remove]
    n_u_original = model.n_u

    model.remove_control(to_remove)

    # removed var
    assert model.n_u == n_u_original - 1
    for ind in range(model.n_u):
        assert not is_equal(model.u[ind], to_remove)


def test_control_is_parametrized(model: ControlMixin):
    model.replace_variable = MagicMock()
    model.create_control('u', 4)
    assert not model.control_is_parametrized(model.u[0])

    # error multiple controls are passed
    with pytest.raises(ValueError):
        model.control_is_parametrized(model.u)

    k = SX.sym('k')
    model.x = SX.sym('x')
    model.parametrize_control(model.u[0], -k * model.x[0], k)

    assert model.control_is_parametrized(model.u[0])
    model.replace_variable.assert_called()
