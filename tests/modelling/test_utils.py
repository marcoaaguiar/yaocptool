import pytest
from yaocptool.modelling.utils import der, Derivative, EqualityEquation
from casadi.casadi import is_equal, SX


def test_equality_equation_init():
    ee = EqualityEquation(1, 2)
    assert ee.lhs == 1
    assert ee.rhs == 2


def test_derivative_init():
    x = SX.sym('x')
    derivative = Derivative(x)
    assert derivative.inner is x


def test_derivative_eq():
    x = SX.sym('x')
    derivative = Derivative(x)

    eq = derivative == -x
    assert isinstance(eq, EqualityEquation)
    assert eq.lhs is derivative
    assert is_equal(eq.rhs, -x, 10)


def test_der():
    x = SX.sym('x')
    derivative = der(x)
    assert isinstance(derivative, Derivative)
    assert derivative.inner is x
