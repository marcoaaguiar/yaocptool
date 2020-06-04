def test_create_state(empty_model):
    x = empty_model.create_state('x', 2)
    assert empty_model.n_x == 2
    for x_i in x.nz:
        assert x_i in empty_model._ode
        assert empty_model._ode[x_i] is None


def test_include_equations(empty_model):
    x = empty_model.create_state('x')
    empty_model.include_equations(ode=[-x])
    assert empty_model.ode.numel() == 1
