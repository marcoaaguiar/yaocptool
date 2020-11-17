from casadi import SX, is_equal

from yaocptool.modelling.utils import Derivative, EqualityEquation, der


def test_derivative_init():
    x = SX.sym("x")
    derivative = Derivative(x)
    assert derivative.inner is x


def test_derivative_eq():
    x = SX.sym("x")
    derivative = Derivative(x)

    eq = derivative == -x
    assert isinstance(eq, EqualityEquation)
    assert eq.lhs is derivative
    assert is_equal(eq.rhs, -x, 10)


def test_der():
    x = SX.sym("x")
    derivative = der(x)
    assert isinstance(derivative, Derivative)
    assert derivative.inner is x


def test_der():
    x = SX.sym("x")
    derivative = der(x)
    assert isinstance(derivative, Derivative)
    assert derivative.inner is x
