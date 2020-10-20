from yaocptool.modelling.mixins import (
    AlgebraicMixin,
    ControlMixin,
    ParameterMixin,
    StateMixin,
)


class SystemModel(StateMixin, AlgebraicMixin, ControlMixin, ParameterMixin):
    pass
