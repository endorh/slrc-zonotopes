from __future__ import annotations

from typing import Sequence, TypedDict

import numpy as np
from scipy.optimize import (
    milp as scipy_milp,
    Bounds,
    LinearConstraint,
    OptimizeResult,
)

__all__ = [
    'milp',
    'SciPyMILPOptions',
    'Bounds',
    'LinearConstraint',
    'OptimizeResult',
]

class SciPyMILPOptions(TypedDict):
    """
    :ivar disp: bool (default: False)
        Set to True if indicators of optimization status are to be printed to the console during optimization.
    :ivar node_limit: int, optional
        The maximum number of nodes (linear program relaxations) to solve before stopping.
        Default is no maximum number of nodes.
    :ivar time_limit: float, optional
        The maximum number of seconds allotted to solve the problem.
        Default is no time limit.
    :ivar min_rel_gap: float, optional
        Termination criterion for MIP solver: solver will terminate when the gap between the
        primal objective value and the dual objective bound, scaled by the primal objective value,
        is <= mip_rel_gap.
    """
    disp: bool | None
    node_limit: int | None
    time_limit: float | None
    mip_rel_gap: float | None


def milp(
    c: int | float | Sequence[int | float], *,
    integrality: int | tuple[int] | None = None,
    bounds: Bounds | tuple[float | int, float | int] = (0, np.inf),
    constraints: LinearConstraint | Sequence[LinearConstraint] | None = None,
    options: SciPyMILPOptions | None = None,
    presolve: bool = False,
    posround: bool = True,
    poscheck: bool = ...,
) -> OptimizeResult:
    """
    Wrapper for SciPy's MILP solver, which is itself a wrapper for HiGHS [1].
    The algorithm is deterministic, and it typically finds the global optimum of
    moderately challenging mixed-integer linear programs (when it exists).

    Unless `presolve` is explicitly set to `True`, disables SciPy's MILP
    pre-solver, as we have encountered examples where it incorrectly labels
    some problems as infeasible.

    Additionally, unless `posround` is explicitly set to `False`, performs
    rounding of integral variables after optimization, ensuring the rounded
    solution satisfies all bounds and constraints.

    These bounds and constraints checks can be requested even for cases where
    no rounding occurs by setting `poscheck=True`, or it can be disabled
    by setting `poscheck=False`.

    Note that these checks cannot guarantee that the rounded solution will be optimal,
    but it typically should, for well-behaved enough problems.

    :param c:
        Functional to minimize. Vector, 1D-array or Sequence of floats.
        Specify a negated functional to maximize it instead.
    :param integrality:
        Integrality constraints per variable (or for all variables if a scalar is given).
        - `0`: Continuous variable; no integrality constraint.
        - `1`: Integer variable; decision variable must be an integer within bounds. (see `posround` for details)
        - `2`: Semi-continuous variable; decision variable must be within bounds or take value 0.
        - `3`: Semi-integer variable; decision variable must be an integer within bounds or take value 0. (see `posround` for details)
    :param bounds:
        `Bounds` instance, or tuple of bounds, which may be scalars or vectors to define bounds for
        each variable.
        Defaults to (0, np.inf).
    :param constraints:
        `LinearConstraint` or sequence of `LinearConstraint`s to be enforced by the solver.
        Their `keep_feasible` flag is ignored.
    :param options:
        Additional options for the HiGHS solver:
        - `disp`: bool (default: False)
            Set to True if indicators of optimization status are to be printed to the console during optimization.
        - `node_limit`: int, optional
            The maximum number of nodes (linear program relaxations) to solve before stopping. Default is no maximum number of nodes.
        - `time_limit`: float, optional
            The maximum number of seconds allotted to solve the problem. Default is no time limit.
        - `mip_rel_gap`: float, optional
            Termination criterion for MIP solver: solver will terminate when the gap between the primal objective value and the dual objective bound, scaled by the primal objective value, is <= mip_rel_gap.
    :param presolve:
        Presolve attempts to identify trivial infeasibilities, identify trivial unboundedness,
        and simplify the problem before sending it to the main solver.
        Unfortunately, it sometimes dismisses feasible problems as infeasible, so it is disabled
        by default.
    :param posround:
        Round variables with integrality requirements after optimization.
        By default, the rounded solution is checked against bounds and constraints,
        unless `poscheck` is explicitly set to `False`.
    :param poscheck:
        Ensure the solution satisfies all bounds and constraints after optimization.
        By default, only checks solutions that have been rounded according to the
        `posround` parameter.

    :return:
        `OptimizeResult` instance with the following mandatory properties:
        - status: int
            An integer representing the exit status of the algorithm.
            - `0`: Optimal solution found.
            - `1`: Iteration or time limit reached.
            - `2`: Problem is infeasible.
            - `3`: Problem is unbounded.
            - `4`: Other; see message for details.
            - `5`: HiGHS Solution discarded due to violation of constraints after rounding or
                   sanity test. See `posround` and `poscheck` parameters.
        - `success`: bool
            True when an optimal feasible solution is found and False otherwise.
        - `message`: str
            A string descriptor of the exit status of the algorithm.

        The following attributes will also be present, but the values may be None, depending on the solution status:
        - `x`: ndarray
            The values of the decision variables that minimize the objective function while satisfying the constraints.
        - `fun`: float
            The optimal value of the objective function `c@x`.
        - `mip_node_count`: int
            The number of subproblems or "nodes" solved by the MILP solver.
        - `mip_dual_bound`: float
            The MILP solver's final estimate of the lower bound on the optimal solution.
        - `mip_gap`: float
            The difference between the primal objective value and the dual objective bound, scaled by the primal objective value.

    ## References
    [1] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J. "HiGHS - high performance software for linear optimization." https://highs.dev/
    [2] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised simplex method." Mathematical Programming Computation, 10 (1), 119-142, 2018. DOI: 10.1007/s12532-017-0130-5
    """

    if options is None:
        options = {}
    result: OptimizeResult = scipy_milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options | dict(presolve=presolve),
    )
    if result.success:
        X = np.zeros_like(result.x)
        x = result.x

        check_required = poscheck
        if posround and integrality is not None:
            if not hasattr(integrality, '__len__'):
                integrality = np.broadcast_to((integrality,), x.shape)
            for i, it in enumerate(integrality):
                if it in {1, 3}:
                    X[i] = np.round(x[i])
                    if check_required is ...:
                        check_required = True
                else:
                    X[i] = x[i]
            result.x = X
            result.fun = c@X
        else:
            X = x

        if check_required is not ... and check_required:
            failed_bounds = False
            if bounds is not None:
                if isinstance(bounds, Bounds):
                    lb, ub = bounds.lb, bounds.ub
                else:
                    lb, ub = bounds
                failed_bounds = np.any(np.logical_or(X < lb, ub < X))

            failed_constraints = False
            if constraints is not None:
                if not isinstance(constraints, (tuple, list)):
                    constraints = (constraints,)
                for c in constraints:
                    if not isinstance(c, LinearConstraint):
                        c = LinearConstraint(*c)
                    sl, su = c.residual(X)
                    if np.any(np.logical_or(sl < 0, su < 0)):
                        failed_constraints = True
                        break

            if failed_bounds or failed_constraints:
                result.success = False
                result.message = "False positive: SciPy milp returned a solution which does not satisfy all bounds and constraints."
                result.status = 5

    return result
