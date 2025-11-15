"""
Base Optimizer Class
Abstract interface for optimization techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple, List, Union
from datetime import datetime


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers."""

    def __init__(self, name: str = "optimizer", solver: Optional[str] = None):
        self.name = name
        self.solver = solver
        self.objective = None
        self.optimization_sense = 'minimize'
        self.result: Optional[Dict[str, Any]] = None
        self._model = None
        self.created_at = datetime.now()
        self.solved_at: Optional[datetime] = None

    @abstractmethod
    def add_variable(self, name: str, var_type: str = 'continuous',
                     lb: Optional[float] = None, ub: Optional[float] = None,
                     initial: Optional[float] = None) -> Any:
        """Add decision variable to the model."""
        pass

    @abstractmethod
    def add_constraint(self, name: str, expression: Any,
                       constraint_type: str = 'eq', rhs: float = 0.0) -> None:
        """Add constraint to the model."""
        pass

    @abstractmethod
    def set_objective(self, expression: Any, sense: str = 'minimize') -> None:
        """Set objective function."""
        pass

    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """Solve the optimization problem. Returns dict with status, objective_value, variables, solver_time."""
        pass

    @abstractmethod
    def get_variable_value(self, var_name: str) -> Optional[float]:
        """Get optimized value of a variable."""
        pass

    def reset(self) -> None:
        """Reset the optimizer to initial state."""
        self.objective = None
        self.result = None
        self._model = None
        self.solved_at = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', solver={self.solver})"