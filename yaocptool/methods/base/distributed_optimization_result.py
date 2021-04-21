from dataclasses import InitVar, dataclass, field
from typing import Dict, Union

import pandas as pd
from casadi import DM

from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.modelling.network.node import Node


@dataclass
class DistibutedOptimizationResult:
    results: Dict[Node, OptimizationResult]
    objective: Union[DM, float]
    number_of_iterations: Dict[str, int]
    times: Dict[str, float]
    errors: InitVar[Dict[Node, DM]]
    constraint_violations: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self, errors):
        df = None
        for error in errors.values():
            node_df = pd.DataFrame(error.T.full())

            df = pd.concat([df, node_df], axis="rows") if df is not None else node_df
        if df is None:
            raise ValueError("No error given, cannot comput constraint_violations")
        self.constraint_violations = df

    def as_dataframe(self):
        df = None
        for result in self.results.values():
            res_df = result.dataset.as_dataframe()
            df = (
                pd.merge(
                    df,
                    res_df,
                    how="outer",
                    left_index=True,
                    right_index=True,
                )
                if df is not None
                else res_df
            )
        if df is None:
            raise ValueError("No results to make a dataframe")
        return df
