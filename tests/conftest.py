import pytest

from yaocptool.modelling.system_model import SystemModel


@pytest.fixture
def empty_model():
    return SystemModel()


@pytest.fixture
def siso_model():
    m = SystemModel(name='siso_model')
    x = m.create_state('x')
    u = m.create_control('u')
    m.include_equations(ode=[-x + u], x=x)
    return m


@pytest.fixture
def mimo_model():
    m = SystemModel(name='mimo_model')
    x = m.create_state('x')
    u = m.create_control('u')
    m.include_equations(ode=[-x + u], x=x)
    return m


@pytest.fixture
def dae_model():
    m = SystemModel(name='dae_model')
    x = m.create_state('x')
    y = m.create_algebraic_variable('y')
    u = m.create_control('u')

    m.include_equations(ode=[-x + u], x=x, alg=[y - x + u**2])
    return m


@pytest.fixture(params=['siso_model', 'mimo_model', 'dae_model'])
def model(request):
    """Hack to be have a fixture of fixtures
    https://github.com/pytest-dev/pytest/issues/349#issuecomment-189370273
    """
    return request.getfixturevalue(request.param)
