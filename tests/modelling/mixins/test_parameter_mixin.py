import pytest
from casadi.casadi import is_equal, SX
from yaocptool.modelling.mixins.parameter_mixin import ParameterMixin


@pytest.fixture
def model():
    return ParameterMixin()


def test_n_p(model):
    assert model.n_p == 0

    model.p = SX.sym("p", 4)
    assert model.n_p == 4


def test_n_theta(model):
    assert model.n_theta == 0

    model.theta = SX.sym("theta", 4)
    assert model.n_theta == 4


def test_p_names(model):
    model.create_parameter("par", 10)

    for ind in range(model.n_p):
        assert model.p[ind].name() == model.p_names[ind]


def test_theta_names(model):
    model.create_theta("theta", 10)

    for ind in range(model.n_theta):
        assert model.theta[ind].name() == model.theta_names[ind]


def test_create_parameter(model):
    n_p_initial = model.n_p
    n_new_p = 4
    p = model.create_parameter("p", n_new_p)
    assert model.n_p == n_p_initial + n_new_p
    assert is_equal(model.p[-n_new_p:], p)


def test_create_theta(model):
    n_theta_initial = model.n_theta
    n_new_theta = 4
    theta = model.create_theta("theta", n_new_theta)
    assert model.n_theta == n_theta_initial + n_new_theta
    assert is_equal(model.theta[-n_new_theta:], theta)


def test_include_parameter(model):
    new_p_1 = SX.sym("new_p")
    new_p_2 = SX.sym("new_p_2", 2)

    model_n_p = model.n_p
    model.include_parameter(new_p_1)

    assert model.n_p == model_n_p + 1
    assert is_equal(model.p[-1], new_p_1)

    model.include_parameter(new_p_2)
    assert model.n_p == model_n_p + 1 + 2
    assert is_equal(model.p[-3], new_p_1)
    assert is_equal(model.p[-2:], new_p_2)


def test_include_theta(model):
    new_theta_1 = SX.sym("new_theta")
    new_theta_2 = SX.sym("new_theta_2", 2)

    model_n_theta = model.n_theta
    model.include_theta(new_theta_1)

    assert model.n_theta == model_n_theta + 1
    assert is_equal(model.theta[-1], new_theta_1)

    model.include_theta(new_theta_2)
    assert model.n_theta == model_n_theta + 1 + 2
    assert is_equal(model.theta[-3], new_theta_1)
    assert is_equal(model.theta[-2:], new_theta_2)


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
