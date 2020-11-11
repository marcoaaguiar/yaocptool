from typing import Dict, List, Optional, TypeVar, Type, Union, Tuple, overload

from casadi2.casadi import collocation_points

Typ = TypeVar("Typ")

class SymMixin:
    @classmethod
    def sym(
        cls: Type[OneDMT],
        name: str,
        size1: Optional[Union[Sparsity, int, Tuple[int, int]]] = ...,
        size2: Optional[int] = ...,
    ) -> OneDMT: ...
    def name(self) -> str: ...
    def dep(self: Typ, n: int) -> Typ: ...
    def op(self: Typ) -> int: ...

class NumericMixin:
    @property
    def T(self: Typ) -> Typ: ...
    shape: Tuple[int, int]
    @overload
    def __init__(
        self: Typ,
        arg1: Union[Sparsity, float, List[float], List[List[float]], DM, Typ],
    ): ...
    @overload
    def __init__(
        self: Typ,
        arg1: Sparsity,
        arg2: Typ,
    ): ...
    @overload
    def __init__(
        self: Typ,
        arg1: int,
        arg2: int,
    ): ...
    @classmethod
    def ones(
        cls: Type[OneDMT],
        arg1: Union[int, Tuple[int, int], Sparsity],
        arg2: Optional[int] = ...,
    ) -> OneDMT: ...
    @classmethod
    def zeros(
        cls: Type[OneDMT],
        arg1: Union[int, Tuple[int, int], Sparsity],
        arg2: Optional[int] = ...,
    ) -> OneDMT: ...
    @classmethod
    def eye(
        cls: Type[OneDMT],
        arg1: Union[int, Tuple[int, int], Sparsity],
        arg2: Optional[int] = ...,
    ) -> OneDMT: ...
    def numel(self) -> int: ...
    def size1(self) -> int: ...
    def size2(self) -> int: ...
    def reshape(self: Typ, size: Tuple[int, int]) -> Typ: ...
    def __getitem__(
        self: Typ,
        s: Union[int, List[int], slice, Tuple[Union[int, List[int], slice], ...]],
    ) -> Typ: ...
    def __setitem__(
        self: Typ,
        s: Union[int, List[int], slice, Tuple[Union[int, List[int], slice], ...]],
        value: Union[float, DM, SX, MX],
    ) -> Typ: ...
    def __add__(self: Typ, other: Union[Typ, float, DM]) -> Typ: ...
    def __radd__(self: Typ, other: Union[Typ, float, DM]) -> Typ: ...
    def __sub__(self: Typ, other: Union[Typ, float, DM]) -> Typ: ...
    def __rsub__(self: Typ, other: Union[Typ, float, DM]) -> Typ: ...
    def __mul__(self: Typ, other: Union[Typ, float, DM]) -> Typ: ...
    def __rmul__(self: Typ, other: Union[Typ, float, DM]) -> Typ: ...
    def __truediv__(self: Typ, other: Union[Typ, float, DM]) -> Typ: ...
    def __neg__(self: Typ) -> Typ: ...

class SX(SymMixin, NumericMixin):
    pass

class MX(SymMixin, NumericMixin):
    pass

class DM(NumericMixin):
    pass

class Function:
    def __init__(
        self, arg1: Union[str, Function], arg2: List[OneDMT], arg3: List[OneDMT]
    ) -> Union[Tuple[OneDMT], OneDMT]: ...
    def __call__(
        self,
        *args: Union[int, List[float], OneDMT],
        **kwargs: Union[int, List[float], OneDMT],
    ) -> Union[Tuple[OneDMT], OneDMT]: ...
    @overload
    def map(
        self, n: int, parallelization: str, max_num_threads: int = ...
    ) -> Function: ...
    @overload
    def map(
        self, n: int, reduce_in: bool, reduce_out: bool, opts: dict
    ) -> Function: ...
    @overload
    def map(
        self,
        name: str,
        parallelization: str,
        n: int,
        reduce_in: bool,
        reduce_out: bool,
        opts: dict,
    ) -> Function: ...
    @overload
    def map(
        self,
        name: str,
        parallelization: str,
        n: int,
        reduce_in: str,
        reduce_out: str,
        opts: dict,
    ) -> Function: ...

class Sparsity(NumericMixin):
    pass

# Create a generic variable that can be 'Parent', or any subclass.
OneDMT = TypeVar("OneDMT", DM, SX, MX)
OneMT = TypeVar("OneMT", Sparsity, DM, SX, MX)
AnyMT = TypeVar("AnyMT", bound=Union[Sparsity, DM, SX, MX])
@overload
def vertcat(*args: Union[DM, List[float], float]) -> DM: ...
@overload
def vertcat(*args: OneMT) -> OneMT: ...
def horzcat(*args: OneMT) -> OneMT: ...
def veccat(*args: OneMT) -> OneMT: ...
def diagcat(*args: OneMT) -> OneMT: ...
def vec(a: OneDMT) -> OneMT: ...
@overload
def repmat(A: Union[OneMT, float], n: int, m: int = ...) -> OneMT: ...
@overload
def repmat(A: Union[OneMT, float], rc: Tuple[int, int] = ...) -> OneMT: ...
def sum1(x: OneMT) -> OneMT: ...
def sum2(x: OneMT) -> OneMT: ...
def mtimes(x: Union[OneMT, List[OneMT]], y: Optional[OneMT] = ...) -> OneMT: ...
def substitute(
    ex: Union[OneMT, List[OneMT]],
    v: Union[OneMT, List[OneMT]],
    vdef: Union[OneMT, List[OneMT]],
) -> OneDMT: ...
def gradient(
    ex: OneDMT,
    arg: OneDMT,
) -> OneDMT: ...
def jacobian(ex: OneDMT, arg: OneDMT, opts: Optional[dict] = ...) -> OneDMT: ...
def dot(
    x: OneDMT,
    y: OneDMT,
) -> OneDMT: ...
def is_equal(x: OneDMT, y: OneDMT, depth: Optional[int] = ...) -> bool: ...
def depends_on(f: OneDMT, arg: OneDMT) -> bool: ...
def collocation_points(order: int, scheme: str = ...) -> List[float]: ...
def integrator(
    name: str, solver: str, dae: Dict[str, OneDMT], opts: dict
) -> Function: ...
def rootfinder(
    name: str, solver: str, rfp: Dict[str, OneDMT], opts: dict
) -> Function: ...
def nlpsol(
    name: str, solver: str, nlp: Dict[str, Union[MX, SX]], opts: dict
) -> Function: ...

inf: float
OP_LT: int
OP_LE: int
OP_EQ: int
