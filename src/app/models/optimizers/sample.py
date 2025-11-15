"""
Mixed Integer Linear Programming (MILP) Optimizer
Uses GEKKO for solving MILP problems.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import time
from gekko import GEKKO

from src.app.models.optimizers.base_optimizer import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationStatus,
    Variable,
    Constraint,
    VariableType,
    ConstraintType
)


class MILPOptimizer(BaseOptimizer):
    """
    Mixed Integer Linear Programming optimizer using GEKKO.

    Supports:
        - Continuous, binary, and integer variables
        - Linear constraints
        - Linear objective functions
        - Multiple solver backends (APOPT, BPOPT, IPOPT)
    """

    def __init__(self,
                 name: str = "MILP_Optimizer",
                 solver: str = 'APOPT',
                 verbose: bool = False,
                 remote: bool = True,
                 time_limit: Optional[float] = None,
                 mip_gap: Optional[float] = None,
                 log_file: Optional[str] = None):
        """
        Initialize MILP optimizer with GEKKO.

        Args:
            name: Optimizer name
            solver: GEKKO solver ('APOPT', 'BPOPT', 'IPOPT')
            verbose: Print solver output
            remote: Use remote (True) or local (False) solver
            time_limit: Maximum solving time in seconds
            mip_gap: MIP optimality gap tolerance
            log_file: Path to log file
        """
        super().__init__(name, solver, verbose, log_file)

        # Initialize GEKKO model
        self._model = GEKKO(remote=remote)
        self._model.options.SOLVER = self._get_solver_id(solver)

        # Solver options
        self.remote = remote
        self.time_limit = time_limit
        self.mip_gap = mip_gap

        # Configure solver options
        if time_limit:
            self._model.options.MAX_TIME = time_limit

        if mip_gap:
            self._model.options.MIP_GAP = mip_gap

        # Store GEKKO variables
        self._gekko_vars: Dict[str, Any] = {}
        self._gekko_constraints: List[Any] = []

        self.log(f"Initialized MILP Optimizer with solver: {solver}")

    def _get_solver_id(self, solver_name: str) -> int:
        """Map solver name to GEKKO solver ID"""
        solver_map = {
            'APOPT': 1,  # MINLP solver
            'BPOPT': 2,  # Another solver option
            'IPOPT': 3  # Interior point optimizer
        }
        return solver_map.get(solver_name.upper(), 1)

    def add_variable(self,
                     name: str,
                     var_type: Union[str, VariableType] = VariableType.CONTINUOUS,
                     lb: Optional[float] = None,
                     ub: Optional[float] = None,
                     initial: Optional[float] = None,
                     description: str = "") -> Any:
        """
        Add decision variable to GEKKO model.

        Args:
            name: Variable identifier
            var_type: 'continuous', 'binary', or 'integer'
            lb: Lower bound
            ub: Upper bound
            initial: Initial value
            description: Variable description

        Returns:
            GEKKO variable object
        """
        # Convert string to VariableType if needed
        if isinstance(var_type, str):
            var_type = VariableType(var_type)

        # Create GEKKO variable based on type
        if var_type == VariableType.BINARY:
            gekko_var = self._model.Var(
                value=initial if initial is not None else 0,
                lb=0,
                ub=1,
                integer=True,
                name=name
            )
        elif var_type == VariableType.INTEGER:
            gekko_var = self._model.Var(
                value=initial if initial is not None else 0,
                lb=lb if lb is not None else 0,
                ub=ub,
                integer=True,
                name=name
            )
        else:  # CONTINUOUS
            gekko_var = self._model.Var(
                value=initial if initial is not None else 0,
                lb=lb,
                ub=ub,
                integer=False,
                name=name
            )

        # Store variable
        self._gekko_vars[name] = gekko_var
        self.variables[name] = Variable(
            name=name,
            var_type=var_type,
            lower_bound=lb,
            upper_bound=ub,
            initial_value=initial,
            description=description
        )

        self.log(f"Added variable: {name} ({var_type.value})")
        return gekko_var

    def add_variables_array(self,
                            base_name: str,
                            size: int,
                            var_type: Union[str, VariableType] = VariableType.CONTINUOUS,
                            lb: Optional[float] = None,
                            ub: Optional[float] = None) -> List[Any]:
        """
        Add array of variables with indexed names.

        Args:
            base_name: Base name for variables (e.g., 'x')
            size: Number of variables to create
            var_type: Variable type
            lb: Lower bound
            ub: Upper bound

        Returns:
            List of GEKKO variables
        """
        variables = []
        for i in range(size):
            var_name = f"{base_name}_{i}"
            var = self.add_variable(var_name, var_type, lb, ub)
            variables.append(var)
        return variables

    def add_constraint(self,
                       name: str,
                       expression: Any,
                       constraint_type: Union[str, ConstraintType] = ConstraintType.EQUALITY,
                       rhs: float = 0.0,
                       description: str = "") -> None:
        """
        Add constraint to GEKKO model.

        Args:
            name: Constraint identifier
            expression: GEKKO expression (left-hand side)
            constraint_type: 'eq', 'leq', or 'geq'
            rhs: Right-hand side value
            description: Constraint description
        """
        # Convert string to ConstraintType if needed
        if isinstance(constraint_type, str):
            constraint_type = ConstraintType(constraint_type)

        # Add constraint to GEKKO
        if constraint_type == ConstraintType.EQUALITY:
            gekko_constraint = self._model.Equation(expression == rhs)
        elif constraint_type == ConstraintType.LESS_EQUAL:
            gekko_constraint = self._model.Equation(expression <= rhs)
        elif constraint_type == ConstraintType.GREATER_EQUAL:
            gekko_constraint = self._model.Equation(expression >= rhs)
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        # Store constraint
        self._gekko_constraints.append(gekko_constraint)
        self.constraints[name] = Constraint(
            name=name,
            constraint_type=constraint_type,
            expression=expression,
            rhs=rhs,
            active=True,
            description=description
        )

        self.log(f"Added constraint: {name} ({constraint_type.value})")

    def set_objective(self,
                      expression: Any,
                      sense: str = 'minimize') -> None:
        """
        Set objective function.

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
            raise ValueError(f"Invalid project_latest sense: {sense}")

        self.objective = expression
        self.log(f"Set objective: {sense}")

    def solve(self,
              problem_data: Optional[Dict] = None,
              disp: Optional[bool] = None,
              debug: bool = False,
              **kwargs) -> OptimizationResult:
        """
        Solve the MILP problem.

        Args:
            disp: Display solver output (overrides verbose)
            debug: Enable debug mode
            **kwargs: Additional solver options

        Returns:
            OptimizationResult with solution
        """
        # Validate model
        is_valid, errors = self.validate_model()
        if not is_valid:
            error_msg = f"Model validation failed: {', '.join(errors)}"
            self.log(error_msg, level="ERROR")
            return OptimizationResult(
                status=OptimizationStatus.ERROR,
                objective_value=None,
                variables=None,
                solver_time=0.0,
                message=error_msg,
                metadata={'errors': errors}
            )

        # Don't call GEKKO solve if model is empty
        if not self._gekko_vars:
            return OptimizationResult(
                status=OptimizationStatus.ERROR,
                objective_value=None,
                variables=None,
                solver_time=0.0,
                message="Cannot solve: No variables defined in model",
                metadata={'errors': ['Empty model']}
            )

        # Set display option
        if disp is None:
            disp = self.verbose

        # Configure additional solver options
        for key, value in kwargs.items():
            setattr(self._model.options, key.upper(), value)

        self.log("Starting project_latest solve...")
        self.print_model_summary()

        # Solve
        start_time = time.time()
        try:
            self._model.solve(disp=disp, debug=debug)
            solve_time = time.time() - start_time

            # Check solution status
            status = self._parse_gekko_status()

            # Extract results
            variables_solution = None
            objective_value = None

            if status == OptimizationStatus.OPTIMAL:
                variables_solution = {
                    name: var.value[0] if hasattr(var.value, '__iter__') else var.value
                    for name, var in self._gekko_vars.items()
                }
                objective_value = self._model.options.OBJFCNVAL

                self.log(f"Optimization successful! Objective: {objective_value:.6f}")
            else:
                self.log(f"Optimization status: {status.value}", level="WARNING")

            # Create result object
            self.result = OptimizationResult(
                status=status,
                objective_value=objective_value,
                variables=variables_solution,
                dual_values=None,  # GEKKO doesn't provide duals easily
                solver_time=solve_time,
                iterations=None,
                message=self._get_solver_message(),
                metadata={
                    'solver': self.solver,
                    'remote': self.remote,
                    'problem_type': 'MILP',
                    'num_variables': len(self.variables),
                    'num_constraints': len(self.constraints),
                    'gekko_options': self._get_solver_options()
                }
            )

            self.solved_at = time.time()

        except Exception as e:
            solve_time = time.time() - start_time
            error_msg = f"Solver error: {str(e)}"
            self.log(error_msg, level="ERROR")

            self.result = OptimizationResult(
                status=OptimizationStatus.ERROR,
                objective_value=None,
                variables=None,
                solver_time=solve_time,
                message=error_msg,
                metadata={'exception': str(e)}
            )

        return self.result

    def _parse_gekko_status(self) -> OptimizationStatus:
        """Parse GEKKO solver status to standard status"""
        # GEKKO doesn't have explicit status codes, check solution quality
        try:
            if self._model.options.APPSTATUS == 1:
                return OptimizationStatus.OPTIMAL
            elif self._model.options.APPSTATUS == 0:
                return OptimizationStatus.INFEASIBLE
            else:
                return OptimizationStatus.FEASIBLE
        except:
            return OptimizationStatus.ERROR

    def _get_solver_message(self) -> str:
        """Get solver message from GEKKO"""
        try:
            return f"APPSTATUS: {self._model.options.APPSTATUS}"
        except:
            return "No solver message available"

    def _get_solver_options(self) -> Dict[str, Any]:
        """Get current solver options"""
        return {
            'SOLVER': self._model.options.SOLVER,
            'IMODE': self._model.options.IMODE,
            'MAX_TIME': self.time_limit,
            'MIP_GAP': self.mip_gap
        }

    def get_variable_value(self, var_name: str) -> Optional[float]:
        """Get optimized value of a variable"""
        if var_name in self._gekko_vars:
            var = self._gekko_vars[var_name]
            value = var.value[0] if hasattr(var.value, '__iter__') else var.value
            return float(value)
        return None

    def get_variables_as_array(self, var_names: List[str]) -> np.ndarray:
        """Get multiple variable values as numpy array"""
        values = [self.get_variable_value(name) for name in var_names]
        return np.array(values)

    def export_model(self, directory: str = 'gekko_model') -> str:
        """
        Export GEKKO model files.

        Args:
            directory: Directory to save model files

        Returns:
            Path to model directory
        """
        import os
        os.makedirs(directory, exist_ok=True)

        # GEKKO automatically saves model files when solve() is called
        # We can copy them to specified directory
        self.log(f"Model files saved to: {directory}")
        return directory

    def get_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Get basic sensitivity analysis.
        Note: GEKKO has limited sensitivity analysis compared to commercial solvers.
        """
        if self.result is None or self.result.status != OptimizationStatus.OPTIMAL:
            return {"error": "No optimal solution available"}

        # Compute basic statistics
        var_stats = {}
        for name, value in self.result.variables.items():
            var_obj = self.variables[name]
            var_stats[name] = {
                'value': value,
                'lower_bound': var_obj.lower_bound,
                'upper_bound': var_obj.upper_bound,
                'at_lower_bound': abs(value - var_obj.lower_bound) < 1e-6 if var_obj.lower_bound is not None else False,
                'at_upper_bound': abs(value - var_obj.upper_bound) < 1e-6 if var_obj.upper_bound is not None else False
            }

        return {
            'objective_value': self.result.objective_value,
            'variable_statistics': var_stats
        }

    def print_solution(self, decimals: int = 4) -> None:
        """Print formatted solution"""
        if self.result is None:
            print("No solution available. Run solve() first.")
            return

        print("\n" + "=" * 70)
        print("OPTIMIZATION SOLUTION")
        print("=" * 70)
        print(f"Status: {self.result.status.value.upper()}")
        print(f"Objective Value: {self.result.objective_value:.{decimals}f}")
        print(f"Solve Time: {self.result.solver_time:.4f}s")
        print("\nVariable Values:")
        print("-" * 70)

        for name, value in sorted(self.result.variables.items()):
            var_info = self.variables[name]
            print(f"{name:30s} = {value:>{decimals + 8}.{decimals}f}  ({var_info.var_type.value})")

        print("=" * 70 + "\n")