"""
Mixed Integer Linear Programming (MILP) Optimizer using GEKKO.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import time
from datetime import datetime
from gekko import GEKKO

from app.models.optimizers.base_optimizer import BaseOptimizer


class MILPOptimizer(BaseOptimizer):
    """MILP optimizer using GEKKO. Supports continuous, binary, and integer variables."""

    def __init__(self, name: str = "MILP", solver: str = 'APOPT',
                 remote: bool = True, time_limit: Optional[float] = None,
                 mip_gap: Optional[float] = None, max_iter: Optional[int] = None):
        super().__init__(name, solver)

        self._model = GEKKO(remote=remote)
        self._model.options.SOLVER = {'APOPT': 1, 'BPOPT': 2, 'IPOPT': 3}.get(solver.upper(), 1)

        if time_limit:
            self._model.options.MAX_TIME = time_limit
        if max_iter:
            self._model.options.MAX_ITER = max_iter
        if mip_gap and solver.upper() == 'APOPT':
            self._model.solver_options = [f'minlp_gap_tol {mip_gap}']

        self._vars: Dict[str, Any] = {}

    def add_variable(self, name: str, var_type: str = 'continuous',
                     lb: Optional[float] = None, ub: Optional[float] = None,
                     initial: Optional[float] = None) -> Any:
        """
        Add a decision variable to the model.

        Args:
            name: Variable name
            var_type: 'continuous', 'binary', or 'integer'
            lb: Lower bound
            ub: Upper bound
            initial: Initial value

        Returns:
            GEKKO variable object
        """
        init = initial if initial is not None else 0

        if var_type == 'binary':
            var = self._model.Var(value=init, lb=0, ub=1, integer=True, name=name)
        elif var_type == 'integer':
            var = self._model.Var(value=init, lb=lb or 0, ub=ub, integer=True, name=name)
        else:
            var = self._model.Var(value=init, lb=lb, ub=ub, integer=False, name=name)

        self._vars[name] = var
        return var

    def add_constraint(self, constraint_obj: Any) -> None:
        """
        Add a constraint to the model.

        Args:
            constraint_obj: GEKKO constraint object (e.g., result of x + y <= 10)
        """
        self._model.Equation(constraint_obj)

    def set_objective(self, expression: Any, sense: str = 'minimize') -> None:
        """
        Set the objective function.

        Args:
            expression: GEKKO expression to optimize
            sense: 'minimize' or 'maximize'
        """
        self.optimization_sense = sense.lower()
        if self.optimization_sense == 'minimize':
            self._model.Minimize(expression)
        elif self.optimization_sense == 'maximize':
            self._model.Maximize(expression)
        else:
            raise ValueError(f"Invalid sense: {sense}. Must be 'minimize' or 'maximize'")
        self.objective = expression

    def solve(self, disp: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Solve the optimization problem.

        Args:
            disp: Display solver output
            **kwargs: Additional solver options

        Returns:
            Dictionary with keys: status, objective_value, variables, solver_time, message
        """
        if not self._vars:
            return {
                'status': 'error',
                'message': 'No variables defined',
                'objective_value': None,
                'variables': None,
                'solver_time': 0.0
            }

        # Set additional solver options
        for key, value in kwargs.items():
            try:
                setattr(self._model.options, key.upper(), value)
            except AttributeError:
                pass

        start = time.time()
        try:
            self._model.solve(disp=disp)
            solve_time = time.time() - start

            # Determine solution status
            status = 'optimal' if self._model.options.APPSTATUS == 1 else \
                'infeasible' if self._model.options.APPSTATUS == 0 else 'feasible'

            # Extract variable values
            variables = {
                name: (var.value[0] if hasattr(var.value, '__iter__') else var.value)
                for name, var in self._vars.items()
            } if status == 'optimal' else None

            # Get objective value (GEKKO returns negative for maximization)
            raw_objective = self._model.options.OBJFCNVAL if status == 'optimal' else None

            if raw_objective is not None and self.optimization_sense == 'maximize':
                objective = -raw_objective
            else:
                objective = raw_objective

            self.result = {
                'status': status,
                'objective_value': objective,
                'variables': variables,
                'solver_time': solve_time,
                'message': f"APPSTATUS: {self._model.options.APPSTATUS}"
            }
            self.solved_at = datetime.now()

        except Exception as e:
            self.result = {
                'status': 'error',
                'objective_value': None,
                'variables': None,
                'solver_time': time.time() - start,
                'message': str(e)
            }

        return self.result

    def get_variable_value(self, var_name: str) -> Optional[float]:
        """
        Get the optimized value of a variable.

        Args:
            var_name: Name of the variable

        Returns:
            Variable value or None if not found
        """
        if var_name in self._vars:
            var = self._vars[var_name]
            value = var.value[0] if hasattr(var.value, '__iter__') else var.value
            return float(value)
        return None

    def get_values(self, var_names: List[str]) -> np.ndarray:
        """
        Get multiple variable values as array.

        Args:
            var_names: List of variable names

        Returns:
            Numpy array of variable values
        """
        return np.array([self.get_variable_value(name) for name in var_names])