from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    Type,
    TypedDict,
    Union,
    Tuple,
    overload,
)

Typ = TypeVar("Typ")
NumericMT = TypeVar("NumericMT", DM, SX, MX)
MixedNumericMT = TypeVar("MixedNumericMT", bound=Union[DM, SX, MX])
SymbolicMT = TypeVar("SymbolicMT", SX, MX)

class NumericMixin:
    shape: Tuple[int, int]
    @property
    def T(self: Typ) -> Typ: ...
    @overload
    def __init__(self): ...
    @overload
    def __init__(
        self,
        arg1: Union[
            Sparsity,
            float,
            List[float],
            List[List[float]],
            DM,
            List[DM],
            NumericMT,
            range,
        ],
    ): ...
    @overload
    def __init__(
        self,
        arg1: Sparsity,
        arg2: NumericMT,
    ): ...
    @overload
    def __init__(
        self,
        arg1: int,
        arg2: int,
    ): ...
    @classmethod
    def ones(
        cls: Type[Typ],
        arg1: Union[int, Tuple[int, int], Sparsity],
        arg2: Optional[int] = ...,
    ) -> Typ: ...
    @classmethod
    def zeros(
        cls: Type[Typ],
        arg1: Union[int, Tuple[int, int], Sparsity],
        arg2: Optional[int] = ...,
    ) -> Typ: ...
    @classmethod
    def eye(
        cls: Type[Typ],
        arg1: Union[int, Tuple[int, int], Sparsity],
        arg2: Optional[int] = ...,
    ) -> Typ: ...
    @classmethod
    def inf(
        cls: Type[Typ],
        arg1: Union[int, Tuple[int, int], Sparsity],
        arg2: Optional[int] = ...,
    ) -> Typ: ...
    def numel(self) -> int: ...
    def size1(self) -> int: ...
    def size2(self) -> int: ...
    def is_constant(self) -> bool: ...
    @overload
    def reshape(self: Typ, rows: int, columns: int) -> Typ: ...
    @overload
    def reshape(self: Typ, size: Tuple[int, int]) -> Typ: ...
    def __getitem__(
        self: Typ,
        s: Union[int, List[int], slice, Tuple[Union[int, List[int], slice], ...]],
    ) -> Typ: ...
    def __setitem__(
        self,
        s: Union[int, List[int], slice, Tuple[Union[int, List[int], slice], ...]],
        value: Union[float, DM, NumericMT],
    ) -> Typ: ...
    def __float__(self) -> float: ...
    def __neg__(self: Typ) -> Typ: ...
    def __add__(self: Typ, other: Union[Typ, int, float, DM]) -> Typ: ...
    def __radd__(self: Typ, other: Union[Typ, int, float, DM]) -> Typ: ...
    def __sub__(self: Typ, other: Union[Typ, int, float, DM]) -> Typ: ...
    def __rsub__(self: Typ, other: Union[Typ, int, float, DM]) -> Typ: ...
    def __mul__(self: Typ, other: Union[Typ, int, float, DM]) -> Typ: ...
    def __rmul__(self: Typ, other: Union[Typ, int, float, DM]) -> Typ: ...
    def __truediv__(self: Typ, other: Union[Typ, int, float, DM, int]) -> Typ: ...
    def __pow__(self: Typ, other: Union[Typ, int, float, DM, int]) -> Typ: ...
    def __lt__(self: Typ, other: Union[Typ, int, float, DM, int]) -> Typ: ...
    def __le__(self: Typ, other: Union[Typ, int, float, DM, int]) -> Typ: ...
    def __eq__(self: Typ, other: Union[Typ, int, float, DM, int]) -> Typ: ...  # type: ignore
    def __ne__(self: Typ, other: Union[Typ, int, float, DM, int]) -> Typ: ...  # type: ignore
    def __gt__(self: Typ, other: Union[Typ, int, float, DM, int]) -> Typ: ...
    def __ge__(self: Typ, other: Union[Typ, int, float, DM, int]) -> Typ: ...

class SymMixin:
    @classmethod
    def sym(
        cls: Type[Typ],
        name: str,
        size1: Optional[Union[Sparsity, int, Tuple[int, int]]] = ...,
        size2: Optional[int] = ...,
    ) -> Typ: ...
    def name(self) -> str: ...
    def dep(self: Typ, n: int) -> Typ: ...
    def op(self: Typ) -> int: ...

class DenseMixin:
    @property
    def nz(self: Typ) -> Iterable[Typ]: ...

class SX(NumericMixin, SymMixin, DenseMixin):
    pass

class MX(NumericMixin, SymMixin, DenseMixin):
    def to_DM(self) -> DM: ...

class DM(NumericMixin, DenseMixin):
    def __lt__(self: Typ, other: Any) -> DM: ...  # type: ignore
    def __le__(self: Typ, other: Any) -> DM: ...  # type: ignore
    def __eq__(self: Typ, other: Any) -> DM: ...  # type: ignore
    def __ne__(self: Typ, other: Any) -> DM: ...  # type: ignore
    def __gt__(self: Typ, other: Any) -> DM: ...  # type: ignore

class Sparsity(NumericMixin):
    pass

# Create a generic variable that can be 'Parent', or any subclass.

NlpSolReturn = TypedDict(
    "NlpSolReturn", {"f": DM, "g": DM, "lam_g": DM, "lam_p": DM, "lam_x": DM, "x": DM}
)
FunctionCallArgT = TypeVar("FunctionCallArgT", DM, MX)

class Function:
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, fname: str): ...
    @overload
    def __init__(self, other: Function): ...
    @overload
    def __init__(
        self,
        name: str,
        d: Dict[str, SymbolicMT],
        name_in: List[str] = ...,
        name_out: List[str] = ...,
        opts: Dict[str, Any] = ...,
    ): ...
    @overload
    def __init__(
        self,
        name: str,
        ex_in: List[SymbolicMT],
        ex_out: List[SymbolicMT],
        name_in: List[str] = ...,
        name_out: List[str] = ...,
        opts: Dict[str, Any] = ...,
    ): ...
    @overload
    def __call__(self, args: FunctionCallArgT, /) -> FunctionCallArgT: ...
    @overload
    def __call__(
        self,
        arg1: FunctionCallArgT,
        arg2: FunctionCallArgT,
        /,
        *args: Union[FunctionCallArgT, DM],
    ) -> List[FunctionCallArgT]: ...
    @overload
    def __call__(
        self, **kwargs: Union[DM, FunctionCallArgT]
    ) -> Dict[str, FunctionCallArgT]: ...
    # call
    # call
    # call
    # call
    @overload
    def call(
        self, args: List[Union[DM, FunctionCallArgT]]
    ) -> List[FunctionCallArgT]: ...
    @overload
    def call(self, kwarg: Mapping[str, Union[float, DM]]) -> Dict[str, DM]: ...
    @overload
    def call(
        self, kwarg: Mapping[str, Union[float, DM, FunctionCallArgT]]
    ) -> Dict[str, FunctionCallArgT]: ...
    #
    #
    #
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
    def stats(self) -> Dict[str, Any]: ...

@overload
def vertcat(
    arg1: SymbolicMT,
    arg2: Union[float, List[float], DM, SymbolicMT],
    arg3: Union[float, List[float], DM, SymbolicMT] = ...,
) -> SymbolicMT: ...
@overload
def vertcat(
    arg1: Union[float, List[float], DM, SymbolicMT],
    arg2: SymbolicMT,
    arg3: Union[float, List[float], DM, SymbolicMT] = ...,
) -> SymbolicMT: ...
@overload
def vertcat(*args: Union[float, List[float], DM]) -> DM: ...
@overload
def vertcat(*args: SymbolicMT) -> SymbolicMT: ...

# horzcat
def horzcat(*args: Union[float, List[float], DM, NumericMT]) -> NumericMT: ...
def veccat(*args: NumericMT) -> NumericMT: ...
def diagcat(*args: NumericMT) -> NumericMT: ...
def vertsplit(arg: NumericMT, n: int) -> List[NumericMT]: ...
def horzsplit(arg: NumericMT, n: int) -> List[NumericMT]: ...
def vec(a: NumericMT) -> NumericMT: ...
def diag(a: NumericMT) -> NumericMT: ...
@overload
def repmat(A: float, n: int, m: int = ...) -> NumericMT: ...
@overload
def repmat(A: NumericMT, n: int, m: int = ...) -> NumericMT: ...
@overload
def repmat(A: NumericMT, rc: Tuple[int, int] = ...) -> NumericMT: ...
def reshape(A: NumericMT, n: int, m: int = ...) -> NumericMT: ...
def sum1(x: NumericMT) -> NumericMT: ...
def sum2(x: NumericMT) -> NumericMT: ...
@overload
def mtimes(x: NumericMT, y: NumericMT = ...) -> NumericMT: ...
@overload
def mtimes(x: Union[DM, SymbolicMT], y: Union[DM, SymbolicMT] = ...) -> SymbolicMT: ...
@overload
def mtimes(x: List[NumericMT]) -> NumericMT: ...
def substitute(
    ex: SymbolicMT, v: SymbolicMT, vdef: Union[SymbolicMT, DM, float]
) -> SymbolicMT: ...
def gradient(
    ex: NumericMT,
    arg: NumericMT,
) -> NumericMT: ...
def jacobian(
    ex: NumericMT, arg: NumericMT, opts: Optional[dict] = ...
) -> NumericMT: ...
def hessian(ex: NumericMT, arg: NumericMT) -> Tuple[NumericMT, NumericMT]: ...
def dot(
    x: Union[float, NumericMT],
    y: Union[float, NumericMT],
) -> NumericMT: ...
def is_equal(
    x: Union[float, DM, SymbolicMT],
    y: Union[float, DM, SymbolicMT],
    depth: Optional[int] = ...,
) -> bool: ...
def depends_on(f: Union[DM, SymbolicMT], arg: SymbolicMT) -> bool: ...
def collocation_points(order: int, scheme: str = ...) -> List[float]: ...
def integrator(
    name: str, solver: str, dae: Dict[str, SymbolicMT], opts: dict
) -> Function: ...
def rootfinder(
    name: str, solver: str, rfp: Union[Function, Dict[str, SymbolicMT]], opts: dict
) -> Function: ...
def nlpsol(
    name: str, solver: str, nlp: Dict[str, SymbolicMT], opts: dict
) -> Function: ...
def qpsol(
    name: str, solver: str, nlp: Dict[str, Union[MX, SX]], opts: dict
) -> Function: ...
@overload
def fmax(arg1: Union[float, DM], arg2: Union[float, DM]) -> DM: ...
@overload
def fmax(
    arg1: Union[float, DM, SymbolicMT], arg2: Union[float, DM, SymbolicMT]
) -> SymbolicMT: ...
def fmin(arg1: Union[float, NumericMT], arg2: Union[float, NumericMT]) -> NumericMT: ...
def fabs(arg1: Union[float, NumericMT]) -> NumericMT: ...
def inv(arg1: Union[float, NumericMT]) -> NumericMT: ...
def sqrt(arg1: Union[float, NumericMT]) -> NumericMT: ...
def mmax(arg1: Union[float, NumericMT]) -> NumericMT: ...

inf: float
pi: float
OP_LT: int
OP_LE: int
OP_EQ: int
