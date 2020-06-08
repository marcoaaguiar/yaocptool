from casadi.casadi import depends_on, is_equal, SX
import pytest
from yaocptool.modelling.mixins.algebraic_mixin import AlgebraicMixin


@pytest.fixture
def model():
    return AlgebraicMixin()


def test_n_y(model):
    assert model.n_y == 0

    model.create_algebraic_variable('y', 3)
    assert model.n_y == 3


def test_include_algebraic(model):
    new_y_1 = SX.sym("new_y")
    new_y_2 = SX.sym("new_y_2", 2)

    model_n_y = model.n_y
    alg = new_y_1 - 3
    model.include_algebraic(new_y_1, alg=alg)

    assert model.n_y == model_n_y + 1
    assert is_equal(model.y[-1], new_y_1)
    assert is_equal(model.alg[-1], alg, 10)

    model.include_algebraic(new_y_2)
    assert model.n_y == model_n_y + 1 + 2
    assert is_equal(model.y[-3], new_y_1)
    assert is_equal(model.y[-2:], new_y_2)


def test_create_algebraic_variable(model):
    n_y_initial = model.n_y
    n_new_y = 4
    y = model.create_algebraic_variable("y", n_new_y)
    assert model.n_y == n_y_initial + n_new_y
    assert is_equal(model.y[-n_new_y:], y)


def test_remove_algebraic(model):
    y = model.create_algebraic_variable('y', 4)
    model.include_equations(alg=[i * y_i for i, y_i in enumerate(y.nz)])

    ind_to_remove = 3
    to_remove = model.y[ind_to_remove]
    to_remove_eq = model.alg[ind_to_remove]

    n_y_original = model.n_y
    n_alg_original = model.alg.numel()

    model.remove_algebraic(to_remove, eq=to_remove_eq)

    # removed var
    assert model.n_y == n_y_original - 1
    for ind in range(model.n_y):
        assert not is_equal(model.y[ind], to_remove)

    # removed alg
    assert model.alg.numel() == n_alg_original - 1
    for ind in range(model.alg.numel()):
        assert not is_equal(model.alg[ind], to_remove_eq)


def test_include_equations_alg(empty_model):
    y = empty_model.create_algebraic_variable('y')
    alg = -y

    empty_model.include_equations(alg=alg)
    assert is_equal(empty_model.alg, -y, 20)


def test_replace_variable_state(model: AlgebraicMixin):
    y = model.create_algebraic_variable('y', 3)
    model.include_equations(alg=[-y])

    # replace y
    original = model.y[1]
    replacement = SX.sym("new_y", original.numel())

    model.replace_variable(original, replacement)

    assert not depends_on(model.alg, original)
    assert depends_on(model.alg, replacement)
